"""
``Modal_Decomposition.Utils.Slepian`` 模块测试。

覆盖 Utils 层统一契约（公开符号可达 / 签名默认值 / 非法 ``mod`` 被拒 / 返回契约
（形状、``float64``、元组长度）/ 与参考实现（scipy）数值一致 / 边界输入 /
确定性），以及本模块专有项：

* ``slepian(N, halfBW, K)`` 返回 ``(K, N)`` 的 DPSS 锥，行**正交归一**：
  ``tapers @ tapers.T ≈ I``（atol 1e-8，实测 ~2.5e-15）；
* ``return_ratios=True`` 返回 ``(tapers, ratios)``，``ratios`` 形状 ``(K,)``、
  取值 ``(0, 1]`` 且**单调不增**（集中比降序）；
* ``mod="auto"`` 解析到 ``SLEPIAN_BACKEND``（当前 ``"scipy"``），其数值与
  ``mod="numpy"`` 逐元素一致（实测最大差 ~4.6e-15，**无符号差异**，相关系数为 +1）；
* ``nTapers=None`` 自动取 ``max(1, floor(2*halfBW - 1))``，并被截断到 ``N``；
* 进程内 LRU 缓存：``clear_cache()`` 归零统计，调用后 ``misses``/``size``/``bytes``
  增长，第二次同参命中 ``hits``；命中返回**副本**（改写返回值不污染缓存）。

实测行为（与模块 docstring 的差异，均按真实行为断言）：

1. ``SLEPIAN_*`` 常量在 ``Utils.Slepian`` **模块**命名空间里（从
   ``Define.ConstDefine`` 导入），``Utils`` 包顶层**不导出**它们 —— 故从模块取；
2. 非法 ``mod`` 抛 **``KeyError``**（``resolve_backend`` 的实现）而非 ``ValueError``，
   测试接受 ``(KeyError, ValueError)``；
3. 显式 ``mod="scipy"`` / ``mod="C"`` / ``mod=None`` 当前**分发失效**：
   ``resolve_backend`` 在 ``NW < N/2``（scipy）与 C 可用（C）时返回 ``None``、
   ``None`` 直接命中 ``case _`` 抛 ``KeyError``，``slepian`` 随即抛
   ``ValueError("unsupported backend None")``。用例以运行时 ``pytest.xfail``
   记录该缺陷（修好后会自动转为通过），C **核心**本身正常（与 numpy 差 ~4e-15），
   故另有用例直接验证 C 核心；
4. ``mod="auto"`` 恒等于 ``SLEPIAN_BACKEND``，未按 docstring 做规模判断；
5. scipy 后端要求 ``NW < N/2``，故默认路径下 ``N=2..6``（``halfBW=3.0``）抛
   ``ValueError("NW must be less than M/2.")`` —— 小 ``N`` 请显式 ``mod="numpy"``。

规模约束：本文件所有数组 ``N <= 256``，不分配大数组。
"""

import inspect

import numpy as np
import pytest

import Modal_Decomposition.Utils as utils
import Modal_Decomposition.Utils.Slepian as slepian_module
from Modal_Decomposition.Utils import (
    available_backends,
    cache_info,
    clear_cache,
    resolve_backend,
    slepian,
)
from Modal_Decomposition.Utils._Slepian.C import available as _c_available
from Modal_Decomposition.Utils._Slepian.C import generate_slepian_c
from Modal_Decomposition.Utils.Slepian import (
    SLEPIAN_BACKEND,
    SLEPIAN_CACHE,
    SLEPIAN_CACHE_MAX_BYTES,
    SLEPIAN_CACHE_SIZE,
    SLEPIAN_SMALL_N_ORDER,
)

#: 全文件统一的小规模参数（N 远小于 256 上限）。
N, NW, K = 64, 2.5, 3

#: 可用的公开后端名（"auto" 之外）。
BACKENDS = ("numpy", "scipy", "C")

#: 分布矩阵允许的绝对误差（正交归一）。
ATOL = 1e-8


@pytest.fixture(autouse=True)
def _fresh_cache():
    """每个用例前后清空进程内缓存，避免用例间计数/条目互相污染。"""
    clear_cache()
    yield
    clear_cache()


def gram_error(tapers: np.ndarray) -> float:
    """``tapers @ tapers.T - I`` 的最大绝对偏差。"""
    k = tapers.shape[0]
    return float(np.abs(tapers @ tapers.T - np.eye(k)).max())


def reference(mod: str = "numpy") -> np.ndarray:
    """``(N, NW, K)`` 上的参考锥。"""
    return slepian(N, NW, K, mod=mod)


def scipy_available() -> bool:
    return bool(available_backends().get("scipy", False))


def c_available() -> bool:
    return bool(available_backends().get("C", False))


# --------------------------------------------------------------------------- #
# 1. 公开面 / 签名 / 常量 / 后端表
# --------------------------------------------------------------------------- #
def test_public_symbols_reachable_from_utils():
    """5 个公开符号都能从 ``Utils`` 顶层取到（该包无 ``__all__``，按属性存在性断言）。"""
    for name in ("slepian", "resolve_backend", "available_backends",
                 "cache_info", "clear_cache"):
        assert hasattr(utils, name), f"Utils 顶层缺少公开符号 {name}"
        assert callable(getattr(utils, name)), f"{name} 不可调用"

    # ``Utils.Slepian`` 是子模块（包无 __all__，导入即成为属性）。
    assert inspect.ismodule(utils.Slepian)
    # 顶层符号就是模块里的同一个对象（不是包装器）。
    assert utils.slepian is slepian_module.slepian
    assert utils.resolve_backend is slepian_module.resolve_backend


def test_module_constants_exist_and_typed():
    """``SLEPIAN_*`` 常量齐备且类型合理（它们在模块命名空间，不在 Utils 顶层）。"""
    assert isinstance(SLEPIAN_BACKEND, str)
    assert SLEPIAN_BACKEND in BACKENDS
    assert isinstance(SLEPIAN_CACHE, bool)
    assert isinstance(SLEPIAN_CACHE_SIZE, int) and SLEPIAN_CACHE_SIZE >= 1
    assert isinstance(SLEPIAN_CACHE_MAX_BYTES, int) and SLEPIAN_CACHE_MAX_BYTES > 0
    assert isinstance(SLEPIAN_SMALL_N_ORDER, list)
    assert SLEPIAN_SMALL_N_ORDER, "小 N 后端顺序不应为空"
    assert set(SLEPIAN_SMALL_N_ORDER) <= set(BACKENDS)


def test_slepian_signature_defaults():
    """签名与文档一致：``nTapers=None``、``mod="auto"``、``return_ratios=False``。"""
    params = inspect.signature(slepian).parameters
    assert list(params)[:3] == ["N", "halfBW", "nTapers"]
    assert params["nTapers"].default is None
    assert params["mod"].default == "auto"
    assert params["return_ratios"].default is False
    assert params["norm"].default is None
    assert params["sym"].default is True
    assert params["cache"].default is None


def test_available_backends_mapping():
    """后端表是 ``{numpy, scipy, C} -> bool``，numpy 恒可用。"""
    backends = available_backends()
    assert set(backends) == set(BACKENDS)
    assert all(isinstance(v, bool) for v in backends.values())
    assert backends["numpy"] is True


def test_resolve_backend_auto_is_available():
    """``resolve_backend(...,"auto")`` 返回一个确实可用的后端名。"""
    name = resolve_backend(N, "auto")
    assert name in available_backends(), f"auto 解析到未知后端 {name!r}"
    assert available_backends()[name] is True, f"auto 解析到不可用后端 {name!r}"
    assert name == SLEPIAN_BACKEND          # 当前实现：auto 恒等于默认后端


# --------------------------------------------------------------------------- #
# 2. 返回契约：形状 / dtype / 正交归一 / 集中比
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("mod", ["numpy", "auto"])
def test_shape_and_orthonormality(mod):
    """``(N=64, nw=2.5, K=3)`` → ``(3, 64)``，且 ``tapers @ tapers.T ≈ I``。"""
    if mod == "auto" and not scipy_available():
        pytest.skip("scipy 未安装（auto → 默认后端不可用）")

    tapers = slepian(N, NW, K, mod=mod)
    assert isinstance(tapers, np.ndarray)
    assert tapers.shape == (K, N)
    assert tapers.dtype == np.float64
    assert np.all(np.isfinite(tapers))

    # 行正交 + 单位能量（行范数 1）。
    assert gram_error(tapers) < ATOL
    assert np.allclose(np.linalg.norm(tapers, axis=1), 1.0, atol=ATOL)


@pytest.mark.parametrize("mod", ["numpy", "auto"])
def test_return_ratios_contract(mod):
    """``return_ratios=True`` → ``(tapers, ratios)``，``ratios`` 为 ``(K,)`` 且单调不增。"""
    if mod == "auto" and not scipy_available():
        pytest.skip("scipy 未安装")

    out = slepian(N, NW, K, mod=mod, return_ratios=True)
    assert isinstance(out, tuple) and len(out) == 2
    tapers, ratios = out
    assert isinstance(tapers, np.ndarray) and tapers.shape == (K, N)
    assert isinstance(ratios, np.ndarray) and ratios.shape == (K,)

    assert np.all(np.isfinite(ratios))
    assert np.all(ratios > 0.0) and np.all(ratios <= 1.0)
    # 集中比降序（行序即阶数序）。
    assert np.all(np.diff(ratios) <= 0.0), f"ratios 非单调不增: {ratios}"


def test_ratios_agree_across_backends():
    """numpy 与默认（scipy）后端的集中比一致。"""
    if not scipy_available():
        pytest.skip("scipy 未安装")
    _, r_np = slepian(N, NW, K, mod="numpy", return_ratios=True)
    _, r_ref = slepian(N, NW, K, mod="auto", return_ratios=True)
    assert np.allclose(r_np, r_ref, atol=ATOL)


# --------------------------------------------------------------------------- #
# 3. 与参考实现一致（numpy vs scipy）
# --------------------------------------------------------------------------- #
def test_numpy_matches_scipy_reference():
    """``mod="auto"``（= scipy）与 ``mod="numpy"`` 逐元素一致，且**无符号差异**。"""
    if not scipy_available():
        pytest.skip("scipy 未安装")
    assert resolve_backend(N, "auto") == "scipy", "本用例假定默认后端为 scipy"

    t_np = slepian(N, NW, K, mod="numpy")
    t_sp = slepian(N, NW, K, mod="auto")

    assert np.allclose(t_np, t_sp, atol=ATOL)
    # 逐行相关性：全为 +1 说明两后端符号约定相同（故上面的直接 allclose 成立）。
    corr = np.array([
        float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
        for a, b in zip(t_np, t_sp)
    ])
    assert np.allclose(corr, 1.0, atol=1e-10), f"后端间存在符号/方向差异: {corr}"
    # 两条判据都成立：绝对量一致；若将来出现符号差，abs 判据仍可用于诊断。
    assert np.allclose(np.abs(t_np), np.abs(t_sp), atol=ATOL)


def test_c_core_matches_numpy():
    """C 核心（ctypes 直调）与 numpy 后端数值一致；未编译则跳过。"""
    if not c_available():
        pytest.skip("C 后端未编译")
    t_c = generate_slepian_c(N, NW, K)
    t_np = slepian(N, NW, K, mod="numpy")
    assert t_c.shape == (K, N)
    assert np.allclose(t_c, t_np, atol=ATOL)
    assert gram_error(t_c) < ATOL


def test_c_backend_via_public_api():
    """``slepian(mod="C")``：未编译跳过；编译但分发缺陷以 ``xfail`` 记录。"""
    if not c_available():
        pytest.skip("C 后端未编译")

    try:
        t_c = slepian(N, NW, K, mod="C")
    except (ValueError, KeyError) as exc:
        pytest.xfail(
            "已知缺陷: resolve_backend('C') 在 C 可用时返回 None，"
            f"slepian 随即抛 {type(exc).__name__}: {exc}"
        )
    assert np.allclose(t_c, slepian(N, NW, K, mod="numpy"), atol=ATOL)


def test_scipy_backend_via_explicit_mod():
    """``slepian(mod="scipy")``：未安装跳过；分发缺陷以 ``xfail`` 记录。"""
    if not scipy_available():
        pytest.skip("scipy 未安装")

    try:
        t_sp = slepian(N, NW, K, mod="scipy")
    except (ValueError, KeyError) as exc:
        pytest.xfail(
            "已知缺陷: resolve_backend('scipy') 在 NW < N/2 时返回 None，"
            f"slepian 随即抛 {type(exc).__name__}: {exc}"
        )
    assert np.allclose(t_sp, slepian(N, NW, K, mod="numpy"), atol=ATOL)


def test_mod_none_uses_default_backend():
    """``mod=None`` 按文档应走 ``SLEPIAN_BACKEND``；当前抛 ``KeyError``（xfail 记录）。"""
    try:
        got = slepian(N, NW, K, mod=None)
    except (KeyError, ValueError) as exc:
        pytest.xfail(
            f"已知缺陷: mod=None 未按文档回落到 SLEPIAN_BACKEND（{type(exc).__name__}: {exc}）"
        )
    assert np.allclose(got, slepian(N, NW, K, mod="auto"), atol=ATOL)


def test_invalid_mod_rejected():
    """非法 ``mod`` 明确报错（实测为 ``KeyError``，非 ``ValueError``）。"""
    for bad in ("bogus", "", "NUMPY", "numpy "):
        with pytest.raises((KeyError, ValueError)):
            slepian(N, NW, K, mod=bad)


def test_mod_string_equals_module_constant(monkeypatch):
    """``mod="auto"`` / ``mod="numpy"`` / ``mod=SLEPIAN_BACKEND`` 三种写法结果一致。"""
    # 把默认后端切成 numpy（可用且行为稳定），验证"常量 == 字符串"的等价性。
    monkeypatch.setattr(slepian_module, "SLEPIAN_BACKEND", "numpy")
    assert resolve_backend(N, "auto") == "numpy"

    t_auto = slepian(N, NW, K, mod="auto")
    t_name = slepian(N, NW, K, mod="numpy")
    t_const = slepian(N, NW, K, mod=slepian_module.SLEPIAN_BACKEND)
    assert np.array_equal(t_auto, t_name)
    assert np.array_equal(t_name, t_const)


# --------------------------------------------------------------------------- #
# 4. 边界：小 N / 单锥 / 自动条数 / 极端 halfBW / 非法参数
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n", [1, 2, 3])
def test_tiny_N_single_taper(n):
    """``N=1/2/3`` 且 ``K=1``：形状 ``(1, N)``、单位能量、有限（numpy 后端）。"""
    tapers = slepian(n, 3.0, 1, mod="numpy")
    assert tapers.shape == (1, n)
    assert np.all(np.isfinite(tapers))
    assert np.allclose(np.linalg.norm(tapers, axis=1), 1.0, atol=ATOL)


def test_default_backend_small_N_behaviour():
    """默认（scipy）后端：``N=1`` 可用；``N=2..6`` 抛明确的 ``ValueError``。"""
    if SLEPIAN_BACKEND != "scipy" or not scipy_available():
        pytest.skip("默认后端不是 scipy")

    assert slepian(1, 3.0, 1).shape == (1, 1)
    for n in range(2, 7):
        with pytest.raises(ValueError, match="NW must be less than"):
            slepian(n, 3.0, 1)


def test_single_taper():
    """``K=1`` 返回 ``(1, 64)`` 单位范数锥（零交叉数为 0 型）。"""
    tapers = slepian(N, NW, 1, mod="numpy")
    assert tapers.shape == (1, N)
    assert np.isclose(np.linalg.norm(tapers[0]), 1.0, atol=ATOL)
    assert np.all(np.isfinite(tapers))


@pytest.mark.parametrize("half_bw", [0.5, 1.0, 2.5, 3.0, 8.0])
def test_auto_ntapers_formula(half_bw):
    """``nTapers=None`` → ``min(N, max(1, floor(2*halfBW - 1)))``。"""
    expected = min(N, max(1, int(np.floor(2.0 * half_bw - 1))))
    tapers = slepian(N, half_bw, None, mod="numpy")
    assert tapers.shape == (expected, N), f"halfBW={half_bw}"
    assert gram_error(tapers) < ATOL


def test_ntapers_is_capped_to_N():
    """``K`` 超过 ``N`` 时截断到 ``N`` 行（不报错、不崩）。"""
    tapers = slepian(N, NW, 10 ** 6, mod="numpy")
    assert tapers.shape == (N, N)
    assert gram_error(tapers) < ATOL


@pytest.mark.parametrize("half_bw, expected_k", [(0.5, 1), (8.0, 15)])
def test_extreme_half_bw(half_bw, expected_k):
    """极端 ``halfBW``（0.5 → 1 条；8.0 → 15 条）仍正交归一（numpy 后端）。"""
    tapers = slepian(N, half_bw, None, mod="numpy")
    assert tapers.shape == (expected_k, N)
    assert gram_error(tapers) < ATOL


@pytest.mark.parametrize("kwargs, match", [
    ({"N": 0, "halfBW": 2.5}, "N must be >= 1"),
    ({"N": -1, "halfBW": 2.5}, "N must be >= 1"),
    ({"N": 64, "halfBW": 0.0}, "halfBW must be > 0"),
    ({"N": 64, "halfBW": -1.0}, "halfBW must be > 0"),
    ({"N": 64, "halfBW": float("nan")}, "halfBW must be > 0"),
    ({"N": 64, "halfBW": float("inf")}, "halfBW must be > 0"),
    ({"N": 64, "halfBW": 2.5, "nTapers": 0}, "nTapers must be >= 1"),
    ({"N": 64, "halfBW": 2.5, "nTapers": -2}, "nTapers must be >= 1"),
    ({"N": 64, "halfBW": 2.5, "norm": 1}, "norm must be one of"),
    ({"N": 64, "halfBW": 2.5, "sym": "yes"}, "sym must be a bool"),
])
def test_invalid_arguments_raise_value_error(kwargs, match):
    """非法参数抛带明确信息的 ``ValueError``（早于任何大分配）。"""
    with pytest.raises(ValueError, match=match):
        slepian(mod="numpy", **kwargs)


def test_norm_and_sym_options():
    """``norm=None`` 与 ``norm=2`` 等价；``"approximate"`` 峰值归一；``sym=False`` 可用。"""
    t_default = slepian(N, NW, K, mod="numpy")
    t_two = slepian(N, NW, K, mod="numpy", norm=2)
    assert np.array_equal(t_default, t_two)

    t_approx = slepian(N, NW, K, mod="numpy", norm="approximate")
    assert t_approx.shape == (K, N)
    peak = np.abs(t_approx).max(axis=1)
    assert np.all(peak <= 1.0 + 1e-12)
    assert peak[0] > 0.99                       # 一阶锥的峰接近 1
    # 不再满足单位能量（与 norm=None 判据相反）。
    assert not np.allclose(np.linalg.norm(t_approx, axis=1), 1.0, atol=1e-3)

    t_periodic = slepian(N, NW, K, mod="numpy", sym=False)
    assert t_periodic.shape == (K, N)
    assert np.all(np.isfinite(t_periodic))


# --------------------------------------------------------------------------- #
# 5. 进程内缓存
# --------------------------------------------------------------------------- #
def test_cache_info_structure():
    """``cache_info()`` 返回稳定结构（版本戳 + 计数）。"""
    info = cache_info()
    assert set(info) == {"version", "size", "bytes", "hits", "misses",
                         "evictions", "skipped"}
    assert isinstance(info["version"], str) and info["version"]
    for key in ("size", "bytes", "hits", "misses", "evictions", "skipped"):
        assert isinstance(info[key], int) and info[key] >= 0
    # autouse fixture 刚清过缓存。
    assert info["size"] == 0 and info["bytes"] == 0
    assert info["hits"] == 0 and info["misses"] == 0


def test_cache_fills_and_hits():
    """``clear_cache()`` 后归零；一次调用 → miss/size/bytes 增长；再调 → hit。"""
    clear_cache()
    info0 = cache_info()
    assert (info0["size"], info0["bytes"]) == (0, 0)
    assert (info0["hits"], info0["misses"]) == (0, 0)

    first = slepian(N, NW, K, mod="numpy")
    info = cache_info()
    assert info["misses"] >= 1
    assert info["size"] >= 1
    assert info["bytes"] > 0

    second = slepian(N, NW, K, mod="numpy")
    info2 = cache_info()
    assert info2["hits"] >= 1
    assert np.array_equal(first, second)


def test_cache_returns_independent_copy():
    """命中返回副本：改写返回值不污染缓存主副本。"""
    original = reference()
    original_copy = original.copy()

    cached = reference()
    cached[0, 0] = 123.0                        # 只改调用方拿到的副本
    again = reference()
    assert np.array_equal(again, original_copy)


def test_cache_can_be_disabled():
    """``cache=False``：结果正确且不写入缓存。"""
    before = cache_info()
    tapers = slepian(32, 2.0, 2, mod="numpy", cache=False)
    after = cache_info()
    assert tapers.shape == (2, 32)
    assert gram_error(tapers) < ATOL
    assert after["size"] == before["size"] == 0
    assert after["misses"] == before["misses"]


# --------------------------------------------------------------------------- #
# 6. 确定性
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("mod", ["numpy", "auto"])
def test_deterministic_bitwise(mod):
    """同参数两次（禁用缓存、真正重算）逐位相同。"""
    if mod == "auto" and not scipy_available():
        pytest.skip("scipy 未安装")

    a = slepian(N, NW, K, mod=mod, cache=False)
    b = slepian(N, NW, K, mod=mod, cache=False)
    assert np.array_equal(a, b)
    assert a.tobytes() == b.tobytes()           # 逐位相同（含符号）
