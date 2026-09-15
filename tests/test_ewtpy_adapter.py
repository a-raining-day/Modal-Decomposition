"""
``EWTpy`` 适配层测试 —— ewtpy 作为**可选**第三方后端的接入契约。

覆盖:

* **导入期不依赖 ewtpy**: ``import Modal_Decomposition`` 不触发 ewtpy 的导入;
* 装了 ewtpy 时: 结果与直接调用 ``ewtpy.EWT1D`` 逐位一致 (转发正确),
  且返回规范的 ``DecompositionResult``;
* 没装 ewtpy 时: ``decompose`` 抛 ``ImportError``, 消息含可照抄的安装命令,
  且**不被缓存**(装上后再调用即可成功);
* 与自研 ``EWT`` 互不影响 (同一个进程里两者都能用)。

对应文档: ``docs/EWT_Native_Report.md``; 可选依赖声明见 ``pyproject.toml``。
"""

import subprocess
import sys
import textwrap

import numpy as np
import pytest

import src.Modal_Decomposition as MD
from src.Modal_Decomposition.EWT import EWT
from src.Modal_Decomposition.EWTpy import EWTpy

FS = 1000.0


@pytest.fixture
def signal():
    rng = np.random.default_rng(0)
    t = np.arange(2048) / FS
    return (np.sin(2 * np.pi * 50 * t) + 0.5 * np.sin(2 * np.pi * 150 * t)
            + 0.05 * rng.standard_normal(2048))


ewtpy = pytest.importorskip("ewtpy", reason="可选后端: 未安装 ewtpy 时跳过对照测试")


# --------------------------------------------------------------------------- #
# 注册与惰性导入
# --------------------------------------------------------------------------- #
def test_registered_and_exposed():
    """注册键 EWTpy 与自研 EWT 并存, 且 facade 同时暴露两者。"""
    assert "EWT" in MD.Class.__dict__ and "EWTpy" in MD.Class.__dict__
    assert MD.Class.EWT is EWT
    assert MD.Class.EWTpy is EWTpy
    assert EWT.name == "EWT" and EWTpy.name == "EWTpy"
    # facade docstring 由 Name/Reference 表拼装
    assert "ewtpy" in MD.Function.EWTpy.__doc__


def test_module_import_does_not_import_ewtpy():
    """
    导入本库/本模块不导入 ewtpy —— 可选依赖只在 decompose 时才被拉起。

    在**子进程**里验证 (本进程可能已被其他测试 import 过 ewtpy)。
    """
    code = textwrap.dedent(
        """
        import sys
        sys.path.insert(0, "src")
        import Modal_Decomposition as MD
        import Modal_Decomposition.EWTpy as E
        # 先清掉可能已存在的记录, 再确认导入本库不会拉起 ewtpy
        print("ewtpy" in sys.modules, E.EWTpy.name, "EWT" in MD.Class.__dict__)
        """
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.split() == ["False", "EWTpy", "True"]


# --------------------------------------------------------------------------- #
# 与 ewtpy 的一致性
# --------------------------------------------------------------------------- #
def test_matches_raw_ewtpy(signal):
    """
    适配层只是参数转发 + 布局归一: 模态/滤波器组/边界与直接调用 EWT1D 逐位一致。

    ``Res`` 是唯一的有意差异 —— 按本库契约定为 ``S − ΣIMFs`` (ewtpy 原样放在
    ``info["ewtpy_residual"]``), 见模块 docstring。
    """
    ewt, mfb, boundaries = ewtpy.EWT1D(signal, 5, 0, "locmax", 0, "average", 10, 5)
    ewt = np.asarray(ewt).T
    r = EWTpy(N=5).decompose(signal, fs=FS)
    assert np.allclose(r.IMFs, ewt[:-1], rtol=0, atol=0)
    assert np.allclose(r.info["ewtpy_residual"], ewt[-1], rtol=0, atol=0)
    assert np.allclose(r.info["mfb"], np.asarray(mfb).T, rtol=0, atol=0)
    assert np.allclose(r.info["boundaries"], boundaries, rtol=0, atol=0)
    assert r.info["backend"] == "ewtpy"


def test_result_contract(signal):
    """返回规范结果对象, 且满足库内 reconstruct 契约 (与其它方法一致)。"""
    r = EWTpy(N=5).decompose(signal, fs=FS)
    assert r.IMFs.ndim == 2 and r.IMFs.shape[1] == signal.size
    assert r.Res.shape == signal.shape
    assert r.config.__class__.__name__ == "EWTpyConfig"
    assert r.config.N == 5 and r.config.reg == "average"
    # 残差按库内契约定: 带和 + 残差 == 输入 (ewtpy 原始约定并不满足, 见 info)
    assert np.allclose(r.IMFs.sum(axis=0) + r.Res, signal, rtol=0, atol=1e-12)
    assert r.info["residual_convention"] == "S - sum(IMFs)"
    # ewtpy 原始约定的带和偏差被记录下来 (非零, 正是规范化的原因)
    assert r.info["ewtpy_band_sum_error"] > 0.0
    raw = np.asarray(ewtpy.EWT1D(signal, 5, 0, "locmax", 0, "average", 10, 5)[0]).T
    assert r.info["ewtpy_band_sum_error"] == pytest.approx(
        float(np.linalg.norm(raw.sum(axis=0) - signal) / np.linalg.norm(signal))
    )


def test_defaults_match_ewtpy_defaults(signal):
    """默认参数与 ewtpy 的默认值一致 (可直接互相对照)。"""
    a = EWTpy().decompose(signal, fs=FS)
    b = ewtpy.EWT1D(signal, 5, 0, "locmax", 0, "average", 10, 5)
    assert np.allclose(a.IMFs, np.asarray(b[0]).T[:-1], rtol=0, atol=0)


def test_facade_function_works(signal):
    """Function.EWTpy(S, **params) 走同一路径。"""
    a = MD.Function.EWTpy(signal, N=5)
    b = EWTpy(N=5).decompose(signal)
    assert np.allclose(a.IMFs, b.IMFs, rtol=0, atol=0)


# --------------------------------------------------------------------------- #
# 未安装 ewtpy 时的行为
# --------------------------------------------------------------------------- #
def test_missing_ewtpy_raises_with_hint(signal, monkeypatch):
    """
    ewtpy 不可用时: ImportError + 安装命令, 且 EWT (自研) 仍照常工作。

    ``sys.modules[name] = None`` 是 CPython 的标准手法: 之后 ``import name``
    必定抛 ImportError, 无需真的卸载包。
    """
    from src.Modal_Decomposition.Base.Cache import cache
    from src.Modal_Decomposition.Base.ConstDefine import CACHE_KEY

    key = CACHE_KEY["ewtpy"]
    saved_module = sys.modules.get(key)
    cache.remove(key)                      # 确保不会被缓存命中
    monkeypatch.setitem(sys.modules, key, None)
    try:
        with pytest.raises(ImportError) as ei:
            EWTpy(N=5).decompose(signal, fs=FS)
        msg = str(ei.value)
        assert "ewtpy" in msg and "pip install" in msg
        # 失败不被缓存: 键仍未注册, 装上后可重试
        assert not cache.check(key)
        # 自研 EWT 不受影响
        assert EWT(num_imfs=5).decompose(signal, fs=FS).IMFs.shape[1] == signal.size
    finally:
        cache.remove(key)
        if saved_module is not None:
            monkeypatch.setitem(sys.modules, key, saved_module)


def test_ewt_does_not_need_ewtpy(signal):
    """自研 EWT 与 ewtpy 完全解耦: 同一进程内两者并存且各自可用。"""
    a = EWT(num_imfs=5).decompose(signal, fs=FS)
    b = EWTpy(N=5).decompose(signal, fs=FS)
    assert a.config.__class__.__name__ == "EWTConfig"
    assert b.config.__class__.__name__ == "EWTpyConfig"
    assert b.info["backend"] == "ewtpy" and "backend" not in a.info
    # 边界量纲不同是本库与 ewtpy 的已知差异: EWT 给 Hz (0…fs/2), EWTpy 给下标
    nb = np.asarray(a.info["boundaries"], dtype=np.float64)
    eb = np.asarray(b.info["boundaries"], dtype=np.float64)
    assert nb[0] == 0.0 and nb[-1] == FS / 2.0
    assert not np.isclose(eb[-1], FS / 2.0)


def test_library_works_with_ewtpy_blocked():
    """
    在**真的无法导入 ewtpy** 的子进程里: 整库照常导入, 自研 EWT 与其它方法照常
    工作, 只有 EWTpy 报出带安装命令的 ImportError。

    这是"ewtpy 已改为可选"的端到端验证 (比逐个 find_spec 打桩更强): 用
    meta_path 拦截器让 ``import ewtpy`` 必定失败。
    """
    code = textwrap.dedent(
        """
        import sys

        class _BlockEwtpy:
            def find_spec(self, name, path=None, target=None):
                if name == "ewtpy" or name.startswith("ewtpy."):
                    raise ModuleNotFoundError(f"No module named {name!r} (blocked by test)")
                return None

        sys.meta_path.insert(0, _BlockEwtpy())
        sys.path.insert(0, "src")

        import numpy as np
        import Modal_Decomposition as MD

        assert "EWTpy" in MD.Class.__dict__, "EWTpy 未注册"
        assert "EWT" in MD.Class.__dict__, "EWT 未注册"

        S = np.sin(2 * np.pi * 5 * np.arange(256) / 100.0) + 0.1 * np.cos(2 * np.pi * 20 * np.arange(256) / 100.0)

        r = MD.Class.EWT(num_imfs=3).decompose(S, fs=100.0)
        assert r.IMFs.shape[1] == 256
        assert np.allclose(r.IMFs.sum(axis=0) + r.Res, S, atol=1e-12)

        r2 = MD.Class.EMD().decompose(S)
        assert r2.IMFs.shape[1] == 256

        try:
            MD.Class.EWTpy(N=3).decompose(S)
        except ImportError as exc:
            assert "pip install" in str(exc), str(exc)
        else:
            raise AssertionError("ewtpy 缺失时 EWTpy 应当报 ImportError")

        print("NO_EWTPY_OK")
        """
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip().endswith("NO_EWTPY_OK"), out.stdout
