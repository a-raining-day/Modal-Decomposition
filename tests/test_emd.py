"""
tests for ``Modal_Decomposition.EMD`` (native EMD implementation).

覆盖:
1. 重构精确性 —— IMFs 之和 + Res == 输入;
2. 模式恢复 —— 首 IMF 跟踪最高频分量; 与 PyEMD 首 IMF 的一致性;
3. 停止条件 —— 单调/常量/极值不足输入 → 0 个 IMF 且 Res == S; max_imf 截断;
4. 参数校验与退化输入 (短信号 / 多通道 / 空);
5. dtype —— f32 保持, int/f16 提升 f64;
6. spline_kind (CubicSpline/PCHIP/linear) 与 nbsym 变体;
7. 性能烟测 (数字供模块 docstring 引用)。
"""

import time

import numpy as np
import pytest

from src.Modal_Decomposition.EMD import EMDConfig, EMD
from src.Modal_Decomposition.Utils.Monotonicity import is_monotonic

pytestmark = pytest.mark.filterwarnings("ignore:.*:UserWarning")


def _two_tone(n=512, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, n, endpoint=False)
    S = np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 37 * t)
    if noise:
        S = S + noise * rng.standard_normal(n)
    return S, t


# --------------------------------------------------------------------------- #
# 1. 重构精确性
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n", [64, 256, 1024])
def test_reconstruction_is_exact(n):
    S, _ = _two_tone(n)
    r = EMD().decompose(S)
    recon = r.IMFs.sum(axis=0) + r.Res
    assert r.IMFs.ndim == 2 and r.IMFs.shape[1] == n
    assert np.allclose(recon, S, atol=1e-9)
    assert isinstance(r.config, EMDConfig)
    assert isinstance(r.info, dict) and len(r.info["iterations"]) == r.IMFs.shape[0]


def test_result_contract_matches_framework():
    from src.Modal_Decomposition.Base import DecompositionResult, Decomposer

    r = EMD().decompose(_two_tone()[0])
    assert isinstance(r, DecompositionResult)
    assert isinstance(EMD(), Decomposer)
    assert EMD().name == "EMD"
    assert EMD()(np.sin(np.linspace(0, 20, 128))).IMFs.ndim == 2  # __call__


# --------------------------------------------------------------------------- #
# 2. 模式恢复
# --------------------------------------------------------------------------- #
def test_first_imf_tracks_highest_frequency():
    S, t = _two_tone(n=512)
    r = EMD().decompose(S)
    assert r.IMFs.shape[0] >= 2
    high = np.sin(2 * np.pi * 37 * t)
    corr = float(np.corrcoef(r.IMFs[0], high)[0, 1])
    assert corr > 0.95, f"first IMF corr with 37Hz tone = {corr:.4f}"


def test_agreement_with_pyemd_first_imf():
    pytest.importorskip("PyEMD")
    from PyEMD import EMD as PyEMD_EMD

    S, _ = _two_tone(n=1024)
    native = EMD().decompose(S).IMFs[0]
    arr = np.asarray(PyEMD_EMD().emd(S), dtype=np.float64)
    pyemd_first = arr[0]
    if pyemd_first.size != native.size:  # 行数不同取逐点比较可能错位
        pytest.skip("row counts differ")
    corr = float(np.corrcoef(native, pyemd_first)[0, 1])
    assert corr > 0.9, f"native vs PyEMD first-IMF corr = {corr:.4f}"


# --------------------------------------------------------------------------- #
# 3. 停止条件与退化输入
# --------------------------------------------------------------------------- #
def test_constant_input_no_imf():
    S = np.full(128, 3.0)
    r = EMD().decompose(S)
    assert r.IMFs.shape[0] == 0
    assert np.array_equal(r.Res, S)


def test_monotonic_input_no_imf():
    S = np.linspace(0, 1, 128)
    assert is_monotonic(S)
    r = EMD().decompose(S)
    assert r.IMFs.shape[0] == 0
    assert np.array_equal(r.Res, S)


def test_max_imf_truncation():
    S, _ = _two_tone(n=512)
    full = EMD().decompose(S)
    assert full.IMFs.shape[0] >= 3
    cut = EMD(max_imf=2).decompose(S)
    assert cut.IMFs.shape[0] == 2
    assert np.allclose(cut.IMFs.sum(0) + cut.Res, S, atol=1e-9)


def test_short_signal_terminates():
    S = np.sin(np.linspace(0, 10, 16))
    r = EMD().decompose(S)
    assert np.allclose(r.IMFs.sum(0) + r.Res, S, atol=1e-9)


def test_invalid_parameters_raise():
    with pytest.raises(ValueError, match="nbsym"):
        EMD(nbsym=-1)
    with pytest.raises(ValueError, match="spline_kind"):
        EMD(spline_kind="bogus")
    with pytest.raises(ValueError, match="max_iter"):
        EMD(max_iter=0)
    for bad in (0.0, 1.5):
        with pytest.raises(ValueError, match="sd_thr"):
            EMD(sd_thr=bad)
    with pytest.raises(ValueError, match="max_imf"):
        EMD(max_imf=0)


def test_multichannel_and_empty_rejected():
    with pytest.raises(ValueError, match="1"):
        EMD().decompose(np.zeros((2, 64)))
    with pytest.raises(ValueError, match="empty"):
        EMD().decompose(np.array([]))


# --------------------------------------------------------------------------- #
# 4. dtype
# --------------------------------------------------------------------------- #
def test_float32_preserved():
    S, _ = _two_tone(n=256)
    S = S.astype(np.float32)
    r = EMD().decompose(S)
    assert r.IMFs.dtype == np.float32
    assert r.Res.dtype == np.float32
    assert np.allclose(r.IMFs.sum(0) + r.Res, S, atol=1e-5)


def test_int_and_float16_promoted_to_float64():
    S = (100 * np.sin(np.linspace(0, 40, 256))).astype(np.int64)
    r = EMD().decompose(S)
    assert r.IMFs.dtype == np.float64
    S16 = _two_tone(n=256)[0].astype(np.float16)
    r16 = EMD().decompose(S16)
    assert r16.IMFs.dtype == np.float64


# --------------------------------------------------------------------------- #
# 5. 算法变体
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("spline_kind", ["CubicSpline", "PCHIP", "linear"])
def test_spline_kind_variants(spline_kind):
    S, _ = _two_tone(n=256)
    r = EMD(spline_kind=spline_kind).decompose(S)
    assert np.allclose(r.IMFs.sum(0) + r.Res, S, atol=1e-9)


@pytest.mark.parametrize("nbsym", [0, 1, 2, 5])
def test_nbsym_variants(nbsym):
    S, _ = _two_tone(n=256)
    r = EMD(nbsym=nbsym).decompose(S)
    assert np.allclose(r.IMFs.sum(0) + r.Res, S, atol=1e-9)


# --------------------------------------------------------------------------- #
# 6. 性能烟测
# --------------------------------------------------------------------------- #
def test_performance_smoke():
    S, _ = _two_tone(n=4096, noise=0.05, seed=1)

    def _median(fn, repeats=7):
        fn()  # 预热 (scipy 惰性 import 等一次性成本不计入)
        ts = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            fn()
            ts.append(time.perf_counter() - t0)
        return float(np.median(ts))

    r0 = EMD().decompose(S)
    t_native = _median(lambda: EMD().decompose(S))

    pyemd_t = float("nan")
    try:
        from PyEMD import EMD as PyEMD_EMD

        pyemd_t = _median(lambda: PyEMD_EMD().emd(S))
    except ImportError:
        pass

    print(f"[emd-perf] n=4096 native={t_native * 1e3:7.2f} ms | "
          f"PyEMD={pyemd_t * 1e3:7.2f} ms | imfs={r0.IMFs.shape[0]}")
    assert t_native < 5.0
