"""
tests for ``Modal_Decomposition.Utils.Peaks``.

覆盖:
1. 正确性 —— scipy 后端与 scipy.signal.find_peaks 直调一致; numpy 后端与
   scipy 在无平台信号上索引一致 (height/threshold/distance 过滤亦一致);
2. 统一契约 —— (indices:int64, properties:dict) / 短输入 / 常量 / NaN;
3. 极端输入 —— 2-D、复数、未知 mod、不支持参数、distance<1;
4. numba 后端 —— 未安装时 ImportError, 安装时与 numpy 一致;
5. 性能烟测 —— 各后端耗时 (数字供 docs/PeaksReport.md);
6. Cache 注册 —— Utils.get_peaks 与 scipy.signal 惰性注册。
"""

import importlib.util
import time

import numpy as np
import pytest

from src.Modal_Decomposition.Utils.Peaks import find_peaks

_HAS_NUMBA = importlib.util.find_spec("numba") is not None


def _synth(n: int = 200, seed: int = 0) -> np.ndarray:
    """无平台的严格尖峰信号: 峰位 20, 70, 120, 170 附近。"""
    rng = np.random.default_rng(seed)
    x = 0.05 * rng.standard_normal(n)
    for pos, amp in ((20, 3.0), (70, 2.2), (120, 4.0), (170, 1.5)):
        x[pos - 2:pos + 3] = [0.2, 0.9, amp, 0.7, 0.1]
    return x


# --------------------------------------------------------------------------- #
# 1. 正确性
# --------------------------------------------------------------------------- #
def test_scipy_backend_matches_direct_call():
    from scipy import signal as ss

    S = _synth()
    idx, props = find_peaks(S, mod="scipy", height=1.0, distance=10)
    ref_idx, ref_props = ss.find_peaks(S, height=1.0, distance=10)
    assert np.array_equal(idx, ref_idx)
    assert props.keys() == ref_props.keys()


@pytest.mark.parametrize(
    "kwargs",
    [{}, {"height": 1.0}, {"distance": 15}, {"threshold": 0.5},
     {"height": 1.2, "distance": 8, "threshold": 0.2}],
)
def test_numpy_backend_matches_scipy_on_strict_peaks(kwargs):
    from scipy import signal as ss

    S = _synth()
    idx, props = find_peaks(S, mod="numpy", **kwargs)
    ref_idx, _ = ss.find_peaks(S, **kwargs)
    assert np.array_equal(idx, ref_idx)
    assert props["peak_heights"].tolist() == S[idx].tolist()


def test_numpy_backend_contract():
    S = _synth()
    idx, props = find_peaks(S, mod="numpy")
    assert idx.dtype == np.int64
    assert idx.ndim == 1
    assert set(props) == {"peak_heights"}


def test_numpy_backend_manual_positions():
    """手工构造: 峰位应为 1、3、5。"""
    S = np.array([0, 1, 0, 2, 0, 3, 0], dtype=np.float64)
    idx, props = find_peaks(S, mod="numpy")
    assert idx.tolist() == [1, 3, 5]
    assert props["peak_heights"].tolist() == [1.0, 2.0, 3.0]


# --------------------------------------------------------------------------- #
# 2. 短输入 / 常量 / NaN
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n", [0, 1, 2])
def test_short_input_returns_empty(n):
    idx, props = find_peaks(np.arange(n, dtype=np.float64), mod="numpy")
    assert idx.size == 0 and idx.dtype == np.int64 and props == {}


def test_constant_signal_no_peaks():
    for mod in ("scipy", "numpy"):
        idx, _ = find_peaks(np.full(20, 3.0), mod=mod)
        assert idx.size == 0


def test_nan_is_never_a_peak():
    from scipy import signal as ss

    S = _synth()
    S[120] = np.nan  # 原峰位处注入 NaN
    idx_np, _ = find_peaks(S, mod="numpy")
    idx_sp, _ = ss.find_peaks(S)
    assert 120 not in idx_np.tolist()
    assert 120 not in idx_sp.tolist()
    assert idx_np.tolist() == idx_sp.tolist()


# --------------------------------------------------------------------------- #
# 3. 极端输入
# --------------------------------------------------------------------------- #
def test_rejects_2d_and_complex():
    with pytest.raises(ValueError, match="1-D"):
        find_peaks(np.zeros((3, 3)), mod="numpy")
    with pytest.raises(ValueError, match="real numeric"):
        find_peaks(np.zeros(5, dtype=np.complex128), mod="numpy")


def test_unknown_mod_raises():
    with pytest.raises(ValueError, match="Unknown mod"):
        find_peaks(_synth(), mod="bogus")


def test_numpy_unsupported_kwarg_raises():
    with pytest.raises(NotImplementedError, match="prominence"):
        find_peaks(_synth(), mod="numpy", prominence=1.0)


def test_numpy_bad_distance_raises():
    with pytest.raises(ValueError, match="distance"):
        find_peaks(_synth(), mod="numpy", distance=0)


def test_int_input_works():
    S = np.array([0, 2, 0, 3, 0], dtype=np.int32)
    idx, _ = find_peaks(S, mod="numpy")
    assert idx.tolist() == [1, 3]


# --------------------------------------------------------------------------- #
# 4. numba 后端
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not _HAS_NUMBA, reason="numba not installed")
def test_numba_backend_matches_numpy():
    S = _synth()
    idx_nb, _ = find_peaks(S, mod="numba", distance=10)
    idx_np, _ = find_peaks(S, mod="numpy", distance=10)
    assert np.array_equal(idx_nb, idx_np)


@pytest.mark.skipif(_HAS_NUMBA, reason="numba installed")
def test_numba_backend_missing_raises_importerror():
    with pytest.raises(ImportError):
        find_peaks(_synth(), mod="numba")


# --------------------------------------------------------------------------- #
# 5. 性能烟测 (数字供 docs/PeaksReport.md)
# --------------------------------------------------------------------------- #
def _median_seconds(fn, repeats=5):
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def test_performance_smoke():
    n = 1_000_000
    t = np.linspace(0, 200, n)
    S = np.sin(2 * np.pi * t) + 0.2 * np.sin(2 * np.pi * 7.3 * t) \
        + 0.05 * np.random.default_rng(0).standard_normal(n)

    t_scipy = _median_seconds(lambda: find_peaks(S, mod="scipy"))
    t_numpy = _median_seconds(lambda: find_peaks(S, mod="numpy"))
    t_numpy_f = _median_seconds(
        lambda: find_peaks(S, mod="numpy", height=0.5, distance=20)
    )
    if _HAS_NUMBA:
        find_peaks(S, mod="numba")  # 预热 (首次含 JIT 编译)
        t_numba = _median_seconds(lambda: find_peaks(S, mod="numba"))
    else:
        t_numba = float("inf")

    idx_sp, _ = find_peaks(S, mod="scipy", height=0.5, distance=20)
    idx_np, _ = find_peaks(S, mod="numpy", height=0.5, distance=20)
    assert np.array_equal(idx_sp, idx_np)  # 无平台正弦: 两后端逐位一致

    print(f"[peaks-perf] n={n}: scipy={t_scipy * 1e3:7.2f} ms | "
          f"numpy={t_numpy * 1e3:7.2f} ms | "
          f"numpy+filter={t_numpy_f * 1e3:7.2f} ms | "
          f"numba={t_numba * 1e3:7.2f} ms | peaks={len(idx_sp)}")
    assert t_scipy < 3.0 and t_numpy < 3.0 and t_numpy_f < 3.0


# --------------------------------------------------------------------------- #
# 6. Cache 注册
# --------------------------------------------------------------------------- #
def test_scipy_signal_registered_after_scipy_backend():
    from src.Modal_Decomposition.Base.Cache import cache

    find_peaks(_synth(), mod="scipy")
    assert cache.check("scipy.signal")


def test_get_peaks_registered_in_cache():
    from src.Modal_Decomposition import Utils as U
    from src.Modal_Decomposition.Base.Cache import cache

    mod = U.get_peaks()
    assert mod is cache.get("Modal_Decomposition.Utils.Peaks")
    assert callable(mod.find_peaks)
