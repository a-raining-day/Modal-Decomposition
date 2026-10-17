"""
``Utils/Mirror_NEW.mirror`` 与 ``Utils/Peaks.find_peaks(mod=..., backend=...)``
的回归测试。

运行方式 (二选一)::

    .venv\\Scripts\\python.exe -m pytest tests/test_mirror_new.py -v
    .venv\\Scripts\\python.exe tests/test_mirror_new.py

说明
----
仓库当前处于重构中间态, 连包 ``__init__`` 都还跑不起来::

    Define/ClassDefine.py:11   from ..Base import Global_Seed
    Base/Chunk.py:17           from ..Define import SIZE, DEFAULT_NUMPY_TYPE

``Define`` 与 ``Base`` 互相 import, 于是 ``import Modal_Decomposition`` 直接失败
(与 ``Mirror_NEW`` 无关)。为了在循环解开之前也能测到真代码, 本文件先试正常
import, 失败则 ``_bootstrap`` 注册最小桩: 常量取自真实的 ``Define/ConstDefine.py``
(该文件只依赖 numpy/typing, 可单独加载), ``Base.Chunk`` / ``Base.Memory`` /
``Base.Cache`` 也尽量加载真件。被测试的 ``Mirror_NEW.py`` / ``Peaks.py`` /
``Check.py`` / ``Poly.py`` / ``Monotonicity.py`` 始终是磁盘上那一份, 不做改写。
"""
import importlib
import importlib.util
import sys
import types
import warnings
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

_BOOTSTRAPPED = False


class _CacheShim:
    """最小 cache 桩 (真 Base.Cache 不可用时的兜底), 只需 import_module。"""

    @staticmethod
    def import_module(name, verbose=False):
        try:
            return importlib.import_module(name)
        except ImportError:
            return None


def _bootstrap():
    """注册最小桩包, 使 Utils 叶子模块可以脱离破损的包 __init__ 加载。"""
    spec = importlib.util.spec_from_file_location(
        "_md_constdefine", _SRC / "Modal_Decomposition" / "Define" / "ConstDefine.py"
    )
    const = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(const)

    def _stub_pkg(name, path):
        module = types.ModuleType(name)
        module.__path__ = [str(path)]
        sys.modules[name] = module
        return module

    _stub_pkg("Modal_Decomposition", _SRC / "Modal_Decomposition")
    define = _stub_pkg("Modal_Decomposition.Define", _SRC / "Modal_Decomposition" / "Define")
    base = _stub_pkg("Modal_Decomposition.Base", _SRC / "Modal_Decomposition" / "Base")
    _stub_pkg("Modal_Decomposition.Utils", _SRC / "Modal_Decomposition" / "Utils")

    for name in ("SIZE", "SAFE_MODS", "SAFE_ARGS", "DEFAULT_NUMPY_TYPE"):
        setattr(define, name, getattr(const, name))

    try:
        chunk = importlib.import_module("Modal_Decomposition.Base.Chunk")
        base.Chunk, base.CHUNK_MOD = chunk.Chunk, chunk.CHUNK_MOD
        base.CHUNK_MOD_ARGS = chunk.CHUNK_MOD_ARGS
        # temp_memmap 不在 Base.Chunk 的导出里; 取不到就留 None (只有 memmap 路径会调用)
        base.temp_memmap = getattr(chunk, "temp_memmap", None)
    except Exception as exc:
        print(f"[note] Base.Chunk 真件加载失败, 用最小桩: {type(exc).__name__}: {exc}")
        base.CHUNK_MOD_ARGS = str       # 仅用于 `CHUNK_MOD_ARGS | None` 这类注解
        base.Chunk = object
        base.CHUNK_MOD = None
        base.temp_memmap = None

    try:
        base.should_use_memmap = importlib.import_module(
            "Modal_Decomposition.Base.Memory"
        ).should_use_memmap
    except Exception as exc:
        print(f"[note] Base.Memory 真件加载失败, 用最小桩: {type(exc).__name__}: {exc}")
        base.should_use_memmap = lambda nbytes, extra=0: False

    try:
        base.cache = importlib.import_module("Modal_Decomposition.Base.Cache").cache
    except Exception as exc:
        print(f"[note] Base.Cache 真件加载失败, 用最小桩: {type(exc).__name__}: {exc}")
        base.cache = _CacheShim

    base.Global_Seed = None
    base.Seed = None


def _import(name: str):
    """正常 import; 包初始化失败时启用一次最小桩再 import。"""
    global _BOOTSTRAPPED

    if not _BOOTSTRAPPED:
        try:
            return importlib.import_module(name)
        except Exception:
            for key in [k for k in sys.modules if k.startswith("Modal_Decomposition")]:
                del sys.modules[key]
            _bootstrap()
            _BOOTSTRAPPED = True

    return importlib.import_module(name)


mirror = _import("Modal_Decomposition.Utils.Mirror").mirror
find_peaks = _import("Modal_Decomposition.Utils.Peaks").find_peaks

if _BOOTSTRAPPED:
    print("[note] 包 __init__ 不可用 (Define <-> Base 循环 import), 已用最小桩加载 Utils 叶子模块")


# --------------------------------------------------------------------------- #
# 工具
# --------------------------------------------------------------------------- #
def _signal(n: int = 600, periods: float = 6.0) -> np.ndarray:
    """``sin`` 扫过 ``periods`` 个整周期: 极值数恰为 ``2 * periods`` 个 (默认 12 个)。"""
    return np.sin(np.linspace(0.0, 2.0 * np.pi * periods, n))


def _extrema(S: np.ndarray):
    """(极大 idx/val, 极小 idx/val, 归并后的 (时间, 幅值))。"""
    max_idx, max_arr = find_peaks(S, mod="max", backend="numpy")
    min_idx, min_arr = find_peaks(S, mod="min", backend="numpy")
    order = np.argsort(np.concatenate([max_idx, min_idx]))
    idx = np.concatenate([max_idx, min_idx])[order]
    return max_idx, max_arr, min_idx, min_arr, idx.astype(np.float64), S[idx]


def _raises(exc, fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except exc:
        return True
    except Exception as other:
        raise AssertionError(f"expected {exc.__name__}, got {type(other).__name__}: {other}")
    raise AssertionError(f"expected {exc.__name__}, nothing raised")


# --------------------------------------------------------------------------- #
# find_peaks: 峰型 / 后端 / 取值符号
# --------------------------------------------------------------------------- #
def test_find_peaks_min_matches_negated_max():
    S = _signal()
    idx_min, val_min = find_peaks(S, mod="min", backend="numpy")
    idx_ref, _ = find_peaks(-S, mod="max", backend="numpy")

    assert np.array_equal(idx_min, idx_ref), "min 峰型应与对 -S 找极大同索引"
    assert np.array_equal(val_min, S[idx_min]), "min 的取值必须是原信号幅值"
    assert val_min.max() < 0.0, "极小值应当落在负半轴"
    assert idx_min.dtype == np.int64


def test_find_peaks_min_filters_use_the_peak_side():
    S = _signal()
    for kw in ({"distance": 9}, {"height": -0.5}, {"threshold": 1e-6}):
        a_idx, a_val = find_peaks(S, mod="min", backend="numpy", **kw)
        b_idx, b_val = find_peaks(-S, mod="max", backend="numpy", **kw)
        assert np.array_equal(a_idx, b_idx), f"min + {kw} 索引应与 -S 上的极大一致"
        assert np.allclose(a_val, S[a_idx])


def test_find_peaks_backends_agree():
    try:
        import scipy.signal  # noqa: F401
    except Exception as exc:  # scipy 可能装了却自身坏掉 (本机 _codata.py 解析失败)
        print(f"      (skip: scipy.signal 不可用 -> {type(exc).__name__}: {str(exc)[:60]})")
        return

    S = _signal()
    for mod in ("max", "min"):
        for kw in ({}, {"height": -0.5}, {"height": 0.5}, {"threshold": 1e-6}):
            a_idx, a_val = find_peaks(S, mod=mod, backend="numpy", **kw)
            b_idx, b_val = find_peaks(S, mod=mod, backend="scipy", **kw)
            assert np.array_equal(a_idx, b_idx), f"numpy/scipy 不一致: mod={mod} {kw}"
            assert np.allclose(a_val, b_val)


def test_find_peaks_rejects_bad_mod_and_backend():
    S = _signal()
    _raises(ValueError, find_peaks, S, mod="numpy")       # 旧语义 (后端写在 mod) 应当报错
    _raises(ValueError, find_peaks, S, backend="bogus")
    _raises(ValueError, find_peaks, S, mod="bogus")


# --------------------------------------------------------------------------- #
# mirror: mod="mirror"
# --------------------------------------------------------------------------- #
def test_mirror_matches_np_pad_reflect():
    for n in (2, 3, 4, 7, 16, 45, 100):
        S = np.sin(np.arange(n) * 0.7)
        for e in (0, 1, 2, 3, n - 1, n, n + 5):
            if e < 0:
                continue
            got = mirror(S, nbsym=e)
            if e == 0:
                ref = S
            elif n == 1:
                ref = np.pad(S, (e, e), mode="edge")
            else:
                ref = np.pad(S, (e, e), mode="reflect")
            assert got.shape == ref.shape, f"n={n} e={e}: {got.shape} != {ref.shape}"
            assert np.array_equal(got, ref), f"n={n} e={e}: 与 np.pad 不一致"


def test_mirror_single_sample_and_empty():
    got = mirror(np.array([2.0]), nbsym=3)
    assert np.array_equal(got, np.pad(np.array([2.0]), (3, 3), mode="edge"))
    _raises(ValueError, mirror, np.array([]), nbsym=3)


# --------------------------------------------------------------------------- #
# mirror: mod="extrema"
# --------------------------------------------------------------------------- #
def test_extrema_table_layout_and_symmetry():
    S = _signal()
    for e in (0, 1, 2, 3):
        max_idx, max_arr, min_idx, min_arr, Te, Se = _extrema(S)
        L = Te.size
        out, out_is_max = mirror(S, mod="extrema", nbsym=e)

        assert out.shape == (2, L + 2 * e), f"nbsym={e}: 长度应为 L + 2*nbsym"
        assert out_is_max.shape == (L + 2 * e,) and out_is_max.dtype == bool
        assert int(out_is_max[e:e + L].sum()) == max_idx.size, "中段掩码的极大个数应与 find_peaks 一致"
        mid = slice(e, e + L)
        assert np.allclose(out[0, mid], Te) and np.allclose(out[1, mid], Se), "中段应为检出的极值"

        for j in range(e):                                    # 左端以首极值为对称中心
            assert out[0, e - 1 - j] == 2.0 * Te[0] - Te[1 + j]
            assert out[1, e - 1 - j] == Se[1 + j]
        for j in range(e):                                    # 右端以末极值为对称中心
            assert out[0, L + e + j] == 2.0 * Te[-1] - Te[L - 2 - j]
            assert out[1, L + e + j] == Se[L - 2 - j]

        assert np.all(np.diff(out[0]) > 0), "时间行应严格递增"


def test_extrema_max_min_placement_is_alternating():
    S = _signal()
    out, _ = mirror(S, mod="extrema", nbsym=2)
    value = out[1, 2:-2]
    sign = np.sign(value)
    assert np.all(np.diff(sign) != 0), "极大 / 极小应当交替出现"


# --------------------------------------------------------------------------- #
# mirror: polynomial / parallel / BLCSE
# --------------------------------------------------------------------------- #
def test_polynomial_shape_and_time_order():
    S = _signal()
    for e in (2, 3, 4):
        _, _, _, _, Te, Se = _extrema(S)
        L = Te.size
        out, _ = mirror(S, mod="polynomial", nbsym=e)
        assert out.shape == (2, L + 2 * e)
        assert np.allclose(out[0, e:e + L], Te) and np.allclose(out[1, e:e + L], Se)
        assert np.all(np.diff(out[0]) > 0), f"nbsym={e}: 外推时间应保持递增"
        assert np.all(np.isfinite(out))


def test_parallel_shape_and_slope():
    S = _signal()
    e = 3
    _, _, _, _, Te, Se = _extrema(S)
    L = Te.size
    out, _ = mirror(S, mod="parallel", nbsym=e)
    assert out.shape == (2, L + 2 * e)

    step = np.abs(np.diff(Te)).mean()
    slope_l = (Se[1] - Se[0]) / (Te[1] - Te[0])
    for j in range(e):                                        # 左端由远到近
        assert out[0, j] == Te[0] - (e - j) * step
        assert out[1, j] == Se[0] - (e - j) * slope_l * step
    slope_r = (Se[-1] - Se[-2]) / (Te[-1] - Te[-2])
    for j in range(e):                                        # 右端由近到远
        assert out[0, L + e + j] == Te[-1] + (j + 1) * step
        assert out[1, L + e + j] == Se[-1] + (j + 1) * slope_r * step


def test_blcse_adds_one_pair_per_side():
    S = _signal()
    for e in (1, 2, 5):
        max_idx, max_arr, min_idx, min_arr, Te, Se = _extrema(S)
        L = Te.size
        out, out_is_max = mirror(S, mod="BLCSE", nbsym=e)
        assert out.shape == (2, L + 4), "BLCSE 每端固定补一对"
        assert int(out_is_max[2:2 + L].sum()) == max_idx.size, "中段掩码的极大个数应与 find_peaks 一致"
        assert np.allclose(out[0, 2:2 + L], Te) and np.allclose(out[1, 2:2 + L], Se)
        assert np.all(np.diff(out[0]) > 0), "时间行应严格递增"

        n_use = min(e, max_arr.size, min_arr.size)
        left = sorted([out[1, 0], out[1, 1]])
        expect = sorted([min_arr[:n_use].mean(), max_arr[:n_use].mean()])
        assert np.allclose(left, expect), "左端一对幅值应为端点附近同类极值均值"


# --------------------------------------------------------------------------- #
# T 的处理
# --------------------------------------------------------------------------- #
def test_T_none_equals_index_axis():
    S = _signal()
    a, a_type = mirror(S, mod="extrema", nbsym=2)
    b, b_type = mirror(S, T=np.arange(S.size, dtype=np.float64), mod="extrema", nbsym=2)
    assert np.allclose(a, b) and np.array_equal(a_type, b_type), "T=None 应当等价于样本序号时间轴"


def test_T_scaled_axis_keeps_order():
    S = _signal()
    T = np.arange(S.size) * 0.004
    out, _ = mirror(S, T, mod="extrema", nbsym=2)
    assert np.all(np.diff(out[0]) > 0)
    first_extremum = int(_extrema(S)[4][0])
    assert np.isclose(out[0, 2], T[first_extremum]), "中段首列应取该极值在 T 上的时刻"


def test_T_decreasing_is_rearranged():
    S = _signal()
    T = np.arange(S.size, dtype=np.float64)
    up, _ = mirror(S, T, mod="extrema", nbsym=2)
    down, _ = mirror(S, T[::-1].copy(), mod="extrema", nbsym=2)
    L = up.shape[1] - 4
    # T 反转后同一样本的时间变成 (N-1-t): 幅值序列整体倒序, 时间序列是倒序后再镜像
    assert np.allclose(down[1, 2:2 + L], up[1, 2:2 + L][::-1]), "递减时间轴应按时间重排幅值"
    assert np.allclose(down[0, 2:2 + L], (S.size - 1) - up[0, 2:2 + L][::-1]), "时间应当镜像"
    assert np.all(np.diff(down[0]) > 0), "重排后时间行仍应严格递增"


def test_T_diagnostics_only_when_verbose():
    S = _signal()
    T = np.arange(S.size, dtype=np.float64) ** 1.2          # 非均匀 + 非单调性无关
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        mirror(S, T, mod="extrema", nbsym=2)
        quiet = len(caught)
        mirror(S, T, mod="extrema", nbsym=2, verbose=True)
        loud = len(caught)

    assert quiet == 0, "verbose=False 连诊断都不该做"
    assert loud >= 1, "verbose=True 应给出 T 的诊断告警"


# --------------------------------------------------------------------------- #
# dtype / 报错矩阵
# --------------------------------------------------------------------------- #
def test_dtype_rules():
    S32 = _signal().astype(np.float32)
    assert mirror(S32, nbsym=2).dtype == np.float32
    assert mirror(S32, mod="extrema", nbsym=2)[0].dtype == np.float32, "float32 不应被抬成 float64"
    assert mirror(S32, dtype=np.dtype(np.float64), nbsym=2).dtype == np.float64

    Si = np.array([0, 1, 0, 2, 0, 3, 0], dtype=np.int64)
    assert mirror(Si, nbsym=2).dtype == Si.dtype, "mirror 模式保持整数 dtype"

    _raises(ValueError, mirror, S32, mod="extrema", nbsym=2, dtype=np.int64)


def test_error_matrix():
    S = _signal()
    _raises(ValueError, mirror, S, mod="extrema", nbsym=10 ** 6)      # 极值不够
    _raises(ValueError, mirror, np.arange(50.0), mod="extrema")       # 单调信号无极值
    _raises(ValueError, mirror, S[:2], mod="extrema")                 # 点数不足以检极值
    _raises(ValueError, mirror, S, mod="BLCSE", nbsym=0)
    _raises(ValueError, mirror, S, mod="polynomial", nbsym=2, poly_order=3)
    _raises(ValueError, mirror, S, T=np.arange(5.0), mod="extrema")   # T 长度不符
    _raises(ValueError, mirror, S, mod="extrema", peaks_backend="bogus")
    _raises(ValueError, mirror, S, mod="bogus")
    _raises(ValueError, mirror, S, nbsym=-1)
    _raises(TypeError, mirror, S, nbsym=1.5)
    _raises(TypeError, mirror, S, verbose="yes")


def test_find_peaks_both_matches_single_modes():
    S = _signal()
    idx, val, is_max = find_peaks(S, mod="both", backend="numpy")
    max_idx, max_val = find_peaks(S, mod="max", backend="numpy")
    min_idx, min_val = find_peaks(S, mod="min", backend="numpy")

    assert idx.size == max_idx.size + min_idx.size
    assert np.all(np.diff(idx) > 0), "both 的索引应按样本序严格升序"
    assert np.array_equal(idx[is_max], max_idx), "掩码为 True 的应是极大"
    assert np.array_equal(idx[~is_max], min_idx), "掩码为 False 的应是极小"
    assert np.array_equal(val, S[idx]), "取值恒为原信号幅值"
    assert np.array_equal(val[is_max], max_val) and np.array_equal(val[~is_max], min_val)
    assert idx.dtype == np.int64 and is_max.dtype == bool
    assert np.all(val[is_max] > 0) and np.all(val[~is_max] < 0)


def test_find_peaks_both_filters_are_per_type():
    S = _signal()
    for kw in ({"height": 0.5}, {"height": -0.5}, {"distance": 20}, {"threshold": 1e-6}):
        idx, _, is_max = find_peaks(S, mod="both", backend="numpy", **kw)
        max_idx, _ = find_peaks(S, mod="max", backend="numpy", **kw)
        min_idx, _ = find_peaks(S, mod="min", backend="numpy", **kw)
        assert np.array_equal(idx[is_max], max_idx), f"极大侧 {kw} 应与单峰型一致"
        assert np.array_equal(idx[~is_max], min_idx), f"极小侧 {kw} 应与单峰型一致"


def test_find_peaks_both_short_signal_is_empty():
    idx, val, is_max = find_peaks(np.array([1.0, 2.0]), mod="both")
    assert idx.size == 0 and val.size == 0 and is_max.size == 0


def test_mirror_extrema_table_comes_from_both():
    """Mirror 的极值表应与 find_peaks(both) 逐位一致 (换成 both 后不回归)。"""
    S = _signal()
    idx, val, is_max = find_peaks(S, mod="both", backend="numpy")
    out, out_is_max = mirror(S, mod="extrema", nbsym=2)
    mid = slice(2, 2 + idx.size)
    assert np.allclose(out[0, mid], idx)
    assert np.allclose(out[1, mid], val)
    assert np.array_equal(out_is_max[mid], is_max), "中段掩码应与 find_peaks(both) 一致"


def test_blcse_endpoint_pair_stays_outside():
    """回归: 端点对曾与中段首/末列撞时间 (重复时间戳 -> 样条不可用)。"""
    S = np.array([0.0, -1.0, 1.0, -1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    out, _ = mirror(S, mod="BLCSE", nbsym=1)
    assert np.all(np.diff(out[0]) > 0), "BLCSE 时间行必须严格递增"


def test_tables_strictly_increasing_on_random_signals():
    """回归: 不规则极值下四模式的时间行都要严格递增 (旧 BLCSE 200 条里 56 条回折)。"""
    rng = np.random.default_rng(0)
    checked = 0
    for _ in range(120):
        S = np.cumsum(rng.standard_normal(int(rng.integers(60, 400))))
        for m in ("extrema", "polynomial", "parallel", "BLCSE"):
            try:
                out, out_is_max = mirror(S, mod=m, nbsym=3)
            except ValueError:
                continue
            assert np.all(np.diff(out[0]) > 0), f"{m}: 时间行必须严格递增"
            assert out_is_max.shape == (out.shape[1],)
            checked += 1
    assert checked > 200, "样本太少, 回归没跑到"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed, failed = 0, []
    for fn in tests:
        try:
            fn()
            print(f"[ok]   {fn.__name__}")
            passed += 1
        except Exception as exc:
            print(f"[FAIL] {fn.__name__}: {type(exc).__name__}: {exc}")
            failed.append(fn.__name__)
    print(f"\n{passed} passed, {len(failed)} failed" + (f" -> {failed}" if failed else ""))
    sys.exit(1 if failed else 0)
