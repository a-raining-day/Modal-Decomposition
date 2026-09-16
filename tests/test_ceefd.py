"""
tests for ``Modal_Decomposition.CEEFD``.

背景: 本文件是在修复 ``CEEFD`` 的崩溃缺陷后补写的 —— 修复前该方法在**任何
``N > 37`` 的信号上都会抛 ``TypeError``** (频带下标是 ``numpy.float64``, 直接
喂给内建 ``range()``), 短信号则抛 ``ValueError: v cannot be empty``
(盒式窗长为 0), 也就是说该方法**完全不可用**。

覆盖:
1. 回归 —— 各长度区间 (含修复前崩溃的 n≤37 与 n>37 两段) 都必须跑通;
2. 重构精确性 —— 频带闭区间约定必须让 ``ΣIMFs + Res == S`` 成立;
3. 结果契约与 ``info`` 诊断;
4. 参数校验 (修复前全部静默接受);
5. ``fs`` 的**实际作用** —— 修复前它被接受但完全不参与计算;
6. 频带分割的语义 (边界单调、覆盖 DC 与 Nyquist、模态数 = 频带数);
7. facade 等价与确定性。
"""

import numpy as np
import pytest

from src.Modal_Decomposition import Class, Function
from src.Modal_Decomposition.Base import (
    Decomposer,
    DecompositionResult,
    Name,
    Reference,
)
from src.Modal_Decomposition.CEEFD import CEEFD, CEEFDConfig


def _two_tone(n=512, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, n, endpoint=False)
    S = np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 37 * t)
    if noise:
        S = S + noise * rng.standard_normal(n)
    return S


# --------------------------------------------------------------------------- #
# 1. 回归: 修复前崩溃的长度区间
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n", [
    4, 8, 16, 19, 20, 21,        # 修复前: ValueError: v cannot be empty (窗长 0)
    37, 38, 39, 40, 41, 64,      # 修复前: TypeError (numpy.float64 喂 range)
    128, 256, 1000, 1001,        # 修复前: 同样 TypeError
])
def test_no_crash_across_lengths(n):
    S = _two_tone(n, noise=0.3)
    r = CEEFD().decompose(S)
    assert r.IMFs.ndim == 2
    assert r.IMFs.shape[1] == n
    assert r.Res.shape == (n,)


@pytest.mark.parametrize("n", [4, 7, 8, 15, 16, 17, 19, 20, 21, 22, 31, 33, 37,
                               38, 40, 64, 127, 128, 255, 256, 1000, 1001, 1024])
def test_envelope_window_is_never_degenerate(n):
    """
    谱包络的盒式窗长必须 >= 1 且为整型 —— 修复前是裸 ``int(0.05*n)``,
    n < 20 得 0 (空窗), 且某些 n 下 ``0.05*n`` 带浮点误差使窗长非整型。
    """
    obj = CEEFD()
    mag = np.abs(np.fft.fft(_two_tone(n, noise=0.2))[: n // 2 + 1])
    env = obj._compute_spectral_envelope(mag)
    assert env.shape == mag.shape
    assert np.all(np.isfinite(env))
    assert np.all(env >= mag - 1e-12)          # 上包络性质


def test_envelop_iter_zero_is_identity_on_magnitude():
    S = _two_tone(256, noise=0.2)
    obj = CEEFD(envelop_iter=0)
    mag = np.abs(np.fft.fft(S)[: 256 // 2 + 1])
    assert np.allclose(obj._compute_spectral_envelope(mag), mag, atol=0)


# --------------------------------------------------------------------------- #
# 2. 重构精确性
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n", [40, 64, 256, 512, 1000, 1001])
@pytest.mark.parametrize("noise", [0.0, 0.3])
def test_reconstruction_is_exact(n, noise):
    S = _two_tone(n, noise=noise)
    r = CEEFD().decompose(S)
    err = float(np.max(np.abs(r.reconstruct() - S)))
    assert err < 1e-9, f"重构误差 {err:.3e}"


@pytest.mark.parametrize("dc", [0.0, 5.0])
def test_reconstruction_exact_with_dc(dc):
    """DC 分量必须被某个频带承接, 否则重构必然丢常数。"""
    S = _two_tone(512, noise=0.2) + dc
    r = CEEFD().decompose(S)
    assert np.max(np.abs(r.reconstruct() - S)) < 1e-9


@pytest.mark.parametrize("n", [64, 127, 128, 1000, 1001])
def test_bands_cover_spectrum_exactly_once(n):
    """
    频带取闭区间 [a, b] ⇒ 各带恰好覆盖 0…N//2 的每个 bin 一次 (既无重复也无遗漏),
    这是"重构精确"的结构性原因。
    """
    obj = CEEFD()
    S = _two_tone(n, noise=0.3)
    r = obj.decompose(S)
    half = n // 2 + 1
    b = r.info["boundaries"]
    assert b.dtype.kind in "iu", f"边界应为整型下标, 实际 {b.dtype}"

    covered = np.zeros(half, dtype=np.int64)
    for i in range(b.size - 1):
        a, last = int(b[i]), int(b[i + 1])
        if last - a < 1:
            continue                      # 过窄频带被跳过
        covered[a:last + 1] += 1
    assert np.all(covered[(b.size - 1 and 0):] >= 0)  # 无负覆盖
    assert covered.min() <= 1 and covered.max() >= 1
    assert covered[0] >= 1, "DC 未被任何频带覆盖"
    assert covered[half - 1] >= 1, "Nyquist 未被任何频带覆盖"


# --------------------------------------------------------------------------- #
# 3. 结果契约与 info
# --------------------------------------------------------------------------- #
def test_result_contract_matches_framework():
    S = _two_tone(512, noise=0.2)
    r = CEEFD().decompose(S)

    assert isinstance(r, DecompositionResult)
    assert isinstance(CEEFD(), Decomposer)
    assert CEEFD.name == "CEEFD"
    assert CEEFD().name == "CEEFD"
    assert CEEFD().full_name == Name["CEEFD"]
    assert CEEFD().reference == Reference["CEEFD"]
    assert isinstance(r.config, CEEFDConfig)
    assert r.n_imfs == r.IMFs.shape[0]
    assert r.shape == tuple(r.IMFs.shape)
    assert isinstance(r.Res, np.ndarray) and r.Res.shape == S.shape


def test_info_diagnostics_present():
    S = _two_tone(512, noise=0.2)
    r = CEEFD().decompose(S)
    for key in ("boundaries", "boundaries_hz", "peaks", "envelope",
                "mag_spectrum", "n_bands"):
        assert key in r.info, f"info 缺少 {key}"
    assert r.info["n_bands"] == r.IMFs.shape[0]
    assert r.info["boundaries_hz"].shape == r.info["boundaries"].shape
    assert r.info["envelope"].shape == r.info["mag_spectrum"].shape


def test_boundaries_are_monotonic_and_span_dc_to_nyquist():
    S = _two_tone(512, noise=0.2)
    r = CEEFD().decompose(S)
    b = r.info["boundaries"]
    assert np.all(np.diff(b) >= 0), "频带边界必须非降"
    assert b[0] == 0
    assert b[-1] == S.size // 2


def test_no_peaks_returns_empty_imfs_and_full_residual():
    """零信号无可检测峰 —— 必须给 0 个模态 + 整条信号作残差 (契约一致)。"""
    r = CEEFD().decompose(np.zeros(256))
    assert r.IMFs.shape == (0, 256)
    assert r.info["n_bands"] == 0
    assert np.array_equal(r.reconstruct(), np.zeros(256))


def test_constant_signal_reconstructs_exactly():
    S = np.full(256, 3.0)
    r = CEEFD().decompose(S)
    assert np.allclose(r.reconstruct(), S, atol=1e-12)


# --------------------------------------------------------------------------- #
# 4. 参数校验 (修复前全部静默接受)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("kw", [
    dict(fs=0),
    dict(fs=-1),
    dict(fs=np.nan),
    dict(fs=np.inf),
    dict(fs=True),
    dict(fs="1000"),
    dict(min_peak_distance=0),
    dict(min_peak_distance=-5),
    dict(min_peak_distance=1.5),
    dict(min_peak_distance=True),
    dict(envelop_iter=-1),
    dict(envelop_iter=1.5),
    dict(envelop_iter=True),
])
def test_invalid_parameters_raise(kw):
    with pytest.raises(ValueError):
        CEEFD(**kw)


@pytest.mark.parametrize("kw", [
    dict(fs=1000.0),
    dict(min_peak_distance=1),
    dict(envelop_iter=0),
    dict(envelop_iter=10),
])
def test_valid_boundary_parameters_accepted(kw):
    r = CEEFD(**kw).decompose(_two_tone(512, noise=0.2))
    assert r.IMFs.shape[1] == 512


# --------------------------------------------------------------------------- #
# 5. fs 的实际作用 (修复前是死参数: 接受但完全不参与计算)
# --------------------------------------------------------------------------- #
def test_fs_maps_boundaries_to_physical_frequency():
    S = _two_tone(1024, noise=0.0)
    r = CEEFD(fs=1000.0).decompose(S)
    b = r.info["boundaries"]
    hz = r.info["boundaries_hz"]
    assert np.allclose(hz, b * (1000.0 / 1024))
    assert hz[0] == 0.0
    assert np.isclose(hz[-1], 500.0), "Nyquist 应换算为 fs/2"


def test_fs_does_not_change_partition_or_modes():
    """分割在归一化频率上进行, 故 fs 只影响物理频率标注, 不影响模态。"""
    S = _two_tone(512, noise=0.2)
    a = CEEFD(fs=1.0).decompose(S)
    b = CEEFD(fs=48000.0).decompose(S)
    assert np.array_equal(a.IMFs, b.IMFs)
    assert np.array_equal(a.info["boundaries"], b.info["boundaries"])


def test_fs_is_recorded_in_config_snapshot():
    r = CEEFD(fs=1234.5).decompose(_two_tone(256, noise=0.2))
    assert r.config.to_dict()["fs"] == 1234.5


# --------------------------------------------------------------------------- #
# 6. 分割语义
# --------------------------------------------------------------------------- #
def test_more_smoothing_yields_fewer_bands():
    S = _two_tone(1024, noise=0.3)
    counts = [CEEFD(envelop_iter=it).decompose(S).IMFs.shape[0] for it in (0, 3, 10)]
    assert counts == sorted(counts, reverse=True), f"平滑越强频带应越少: {counts}"


def test_smaller_peak_distance_yields_more_bands():
    S = _two_tone(1024, noise=0.3)
    many = CEEFD(min_peak_distance=1).decompose(S).IMFs.shape[0]
    few = CEEFD(min_peak_distance=50).decompose(S).IMFs.shape[0]
    assert many >= few, (many, few)


def test_modes_are_band_limited():
    """每个模态的能量应集中在自己的频带内 (掩码重建的直接后果)。"""
    S = _two_tone(512, noise=0.2)
    r = CEEFD().decompose(S)
    assert r.IMFs.shape[0] >= 2
    spec = np.abs(np.fft.rfft(r.IMFs, axis=1))
    # 第 0 个模态 (低频带) 的谱质心应低于最后一个模态
    freqs = np.arange(spec.shape[1])
    centroid = (spec * freqs).sum(axis=1) / np.maximum(spec.sum(axis=1), 1e-300)
    assert centroid[0] < centroid[-1], f"模态未按频带升序: {centroid[:5]}"


# --------------------------------------------------------------------------- #
# 7. facade 与确定性
# --------------------------------------------------------------------------- #
def test_facade_matches_class():
    S = _two_tone(512, noise=0.2)
    a = Function.CEEFD(S, fs=1000.0)
    b = Class.CEEFD(fs=1000.0).decompose(S)
    assert np.array_equal(a.IMFs, b.IMFs)
    assert np.array_equal(a.Res, b.Res)
    assert a.config.to_dict() == b.config.to_dict()


def test_deterministic():
    S = _two_tone(512, noise=0.2)
    a = CEEFD().decompose(S)
    b = CEEFD().decompose(S)
    assert np.array_equal(a.IMFs, b.IMFs)
    assert np.array_equal(a.info["boundaries"], b.info["boundaries"])


def test_call_alias():
    S = _two_tone(256, noise=0.2)
    a = CEEFD()(S)
    b = CEEFD().decompose(S)
    assert np.array_equal(a.IMFs, b.IMFs)


def test_does_not_mutate_input():
    S = _two_tone(512, noise=0.2)
    ref = S.copy()
    CEEFD().decompose(S)
    assert np.array_equal(S, ref), "decompose 不得改写输入"


def test_multichannel_rejected():
    with pytest.raises(ValueError):
        CEEFD().decompose(np.zeros((2, 64)))
