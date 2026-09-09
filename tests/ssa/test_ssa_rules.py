"""
SSA rank-rule / window / denoising tests (strided patch-SVD study features).

Covers:
* default mode unchanged: every component returned, exact reconstruction;
* energy rule k matches the cumulative squared-singular energy definition;
* SVHT rule k matches the Gavish-Donoho median threshold;
* SVHT-clip bounds: floor 2 and the energy cap;
* denoising smoke: SVHT-clip gain >> 99.5 %-energy gain on a tone + wgn case;
* windowed OLA (hann/hamming) reproduces the signal exactly where the
  synthesis weight is non-zero (interior; endpoints of zero-endpoint
  windows are dropped);
* groups x rank-rule intersection semantics;
* parameter validation of the new arguments.
"""

import numpy as np
import pytest

from Modal_Decomposition import Class, Function
from Modal_Decomposition.SSA import (
    RANK_RULES,
    WINDOW_OPTIONS,
    SSA,
    energy_rank,
    svht_clip_rank,
    svht_omega,
    svht_rank,
    window_vec,
)

N = 4096
FS = 1000.0


@pytest.fixture(scope="module")
def ssa_signal():
    """Two-tone + light noise (fixed seed)."""
    rng = np.random.default_rng(0)
    t = np.arange(N) / FS
    s = (
        np.sin(2 * np.pi * 37.0 * t)
        + 0.6 * np.sin(2 * np.pi * 113.0 * t)
        + 0.05 * rng.standard_normal(N)
    )
    return s


def _snr_db(clean, noisy):
    noise = noisy - clean
    return 10.0 * np.log10(np.sum(clean ** 2) / np.sum(noise ** 2) + 1e-30)


# --------------------------------------------------------------------- #
# defaults stay classical
# --------------------------------------------------------------------- #
def test_ssa_default_mode_returns_all_components(ssa_signal):
    obj = Class.SSA()
    r = obj.decompose(ssa_signal)
    assert r.config.rank_rule is None
    assert r.config.window == "rect"
    assert r.config.groups is None
    assert np.max(np.abs(r.reconstruct() - ssa_signal)) < 1e-9
    assert r.IMFs.shape[0] == obj.sigma_.size


def test_ssa_default_is_unchanged_bitwise_reference(ssa_signal):
    """New pipeline with defaults equals the classical stride-based result."""
    a = Class.SSA().decompose(ssa_signal)
    b = Class.SSA(stride=1, window="rect", rank_rule=None).decompose(ssa_signal)
    assert np.array_equal(a.IMFs, b.IMFs)


# --------------------------------------------------------------------- #
# pure rule helpers
# --------------------------------------------------------------------- #
def test_svht_omega_reference_values():
    # reference values of the Gavish-Donoho median multiplier curve
    assert np.isclose(svht_omega(0.0), 1.43)
    assert np.isclose(svht_omega(1.0), 2.86)  # 0.56 - 0.95 + 1.82 + 1.43
    assert svht_omega(0.5) > svht_omega(0.0)


def test_energy_rank_white_noise_ceil_fraction():
    # flat spectrum: k_r0 == ceil(r0 * L) exactly
    sv = np.ones(128)
    assert energy_rank(sv, 0.995) == int(np.ceil(0.995 * 128))
    assert energy_rank(sv, 1.0) == 128


def test_svht_rank_low_rank_signal():
    # rank-4 signal: leading singular values dominate the median
    sv = np.array([100.0, 50.0, 30.0, 20.0, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3])
    k = svht_rank(sv, beta=0.5)
    assert k == int(np.sum(sv > svht_omega(0.5) * np.median(sv)))
    assert k >= 4


def test_svht_clip_rank_bounds():
    sv = np.array([100.0, 50.0, 30.0, 20.0] + list(np.linspace(1.0, 0.1, 60)))
    k = svht_clip_rank(sv, beta=0.5, floor=2, cap_frac=0.95)
    assert 2 <= k <= energy_rank(sv, 0.95)
    # floor respected on pure noise (k_sv would be ~0 with a flat tail)
    flat = np.full(64, 3.0)
    assert svht_clip_rank(flat, beta=0.5, floor=2, cap_frac=0.95) >= 2


# --------------------------------------------------------------------- #
# decompose-level rule behaviour
# --------------------------------------------------------------------- #
def test_energy_rule_rank_matches_manual(ssa_signal):
    obj = Class.SSA(window_size=256, rank_rule="energy", energy_frac=0.90)
    r = obj.decompose(ssa_signal)
    tot = float(np.sum(obj.sigma_ ** 2))
    k_manual = int(np.argmax(np.cumsum(obj.sigma_ ** 2) >= 0.90 * tot)) + 1
    assert r.IMFs.shape[0] == k_manual
    assert r.config.rank_rule == "energy"
    assert r.config.energy_frac == 0.90
    assert r.info["k"] == k_manual
    assert r.info["rank_rule"] == "energy"
    # energy retained by the truncated patch matrix: at least the requested
    # fraction; overshoot bounded by the k-th singular energy share
    kept = float(np.sum(obj.sigma_[:k_manual] ** 2)) / tot
    assert kept >= 0.90 - 1e-12
    assert kept <= 0.90 + obj.sigma_[k_manual - 1] ** 2 / tot + 1e-12


def test_svht_rule_rank_matches_manual(ssa_signal):
    obj = Class.SSA(window_size=256, rank_rule="svht")
    r = obj.decompose(ssa_signal)
    L = obj.U_.shape[0]          # embedding rows
    K = obj.V_.shape[1]          # trajectory columns
    beta = min(L, K) / max(L, K)
    assert r.IMFs.shape[0] == svht_rank(obj.sigma_, beta)
    assert r.IMFs.shape[0] == int(
        np.sum(obj.sigma_ > svht_omega(beta) * np.median(obj.sigma_))
    )


def test_svht_clip_rule_keeps_two_at_least_and_capped(ssa_signal):
    obj = Class.SSA(window_size=256, rank_rule="svht_clip", svht_cap_frac=0.95)
    r = obj.decompose(ssa_signal)
    L = obj.U_.shape[0]
    K = obj.V_.shape[1]
    beta = min(L, K) / max(L, K)
    assert r.IMFs.shape[0] == svht_clip_rank(obj.sigma_, beta, 2, 0.95)
    assert 2 <= r.IMFs.shape[0] <= energy_rank(obj.sigma_, 0.95)


def test_denoise_gain_rule_comparison():
    """SVHT-clip >> 99.5 % energy rule on a separable tone + white-noise case."""
    rng = np.random.default_rng(7)
    t = np.arange(8192) / FS
    s = np.sin(2 * np.pi * 47.0 * t) + 0.5 * np.sin(2 * np.pi * 173.0 * t)
    noise = rng.standard_normal(8192) * float(np.std(s))  # SNR ~ 0 dB
    x = s + noise

    obj = Class.SSA(window_size=512, rank_rule="svht_clip")
    clip = obj.decompose(x)
    y_clip = clip.IMFs.sum(axis=0)

    obj2 = Class.SSA(window_size=512, rank_rule="energy", energy_frac=0.995)
    y_r995 = obj2.decompose(x).IMFs.sum(axis=0)

    g_clip = _snr_db(s, y_clip) - _snr_db(s, x)
    g_995 = _snr_db(s, y_r995) - _snr_db(s, x)
    # broadband white noise keeps the 99.5 % energy rank near full rank
    assert g_995 < 1.0, f"energy-99.5 rule should gain little on wgn, got {g_995:.2f} dB"
    assert g_clip > 3.0, f"SVHT-clip should gain clearly, got {g_clip:.2f} dB"
    assert g_clip - g_995 > 3.0


def test_facade_accepts_new_kwargs(ssa_signal):
    r = Function.SSA(ssa_signal, window_size=256, stride=64,
                     rank_rule="svht_clip", window="hamming")
    assert r.IMFs.ndim == 2
    assert np.all(np.isfinite(r.IMFs))


# --------------------------------------------------------------------- #
# windowed OLA reconstruction
# --------------------------------------------------------------------- #
@pytest.mark.parametrize("window", ["hann", "hamming"])
def test_windowed_full_reconstruction_interior(ssa_signal, window):
    r = Class.SSA(window_size=256, window=window).decompose(ssa_signal)
    assert r.config.window == window
    recon = r.reconstruct()
    sl = slice(1, -1) if window == "hann" else slice(0, N)
    assert np.max(np.abs(recon[sl] - ssa_signal[sl])) < 1e-9, window


def test_hann_zero_endpoints_only():
    s = np.arange(8.0)
    # windowed full decomposition: samples whose accumulated window weight
    # is zero (the first sample, and the last sample fed by the w[-1] = 0
    # endpoint of the last patch at stride == 1) reconstruct to 0; all
    # interior samples are exact
    obj = Class.SSA(window_size=4, stride=1, window="hann")
    r = obj.decompose(s)
    recon = r.reconstruct()
    assert np.max(np.abs(recon[1:-1] - s[1:-1])) < 1e-9
    assert recon[0] == 0.0 and recon[-1] == 0.0


def test_window_vec_shapes_and_rect():
    for kind in WINDOW_OPTIONS:
        w = window_vec(16, kind)
        assert w.shape == (16,)
        assert np.all(np.isfinite(w))
    assert np.array_equal(window_vec(8, "rect"), np.ones(8))
    assert np.array_equal(window_vec(8, "none"), np.ones(8))
    assert np.allclose(window_vec(8, "hann"), np.hanning(8))


# --------------------------------------------------------------------- #
# groups x rank rule
# --------------------------------------------------------------------- #
def test_groups_intersect_rank_rule(ssa_signal):
    # energy 0.995 on a tone-dominated signal retains only a handful of
    # leading triples; far-index groups (inside the singular-value range but
    # beyond the retained rank) are dropped, leading groups are kept as-is
    obj = Class.SSA(
        window_size=256,
        groups=[[0, 1], [2, 3], [250, 251]],
        rank_rule="energy",
        energy_frac=0.995,
    )
    r = obj.decompose(ssa_signal)
    assert r.info["k"] < 250          # far groups lie beyond the rank
    assert r.config.groups is not None
    for g in r.config.groups:
        assert max(g) < r.info["k"]
    assert len(r.config.groups) == 2  # far group dropped
    # reconstruction of retained groups is the denoised signal
    assert np.all(np.isfinite(r.reconstruct()))


def test_rank_rule_requires_energy_for_groups_out_of_range(ssa_signal):
    obj = Class.SSA(window_size=256, groups=[[0, 99999]])
    with pytest.raises(ValueError):
        obj.decompose(ssa_signal)


# --------------------------------------------------------------------- #
# validation
# --------------------------------------------------------------------- #
def test_validation_new_params(ssa_signal):
    with pytest.raises(ValueError):
        Class.SSA(window="kaiser")
    with pytest.raises(ValueError):
        Class.SSA(rank_rule="r995")
    with pytest.raises(ValueError):
        Class.SSA(rank_rule="energy", energy_frac=0.0)
    with pytest.raises(ValueError):
        Class.SSA(rank_rule="energy", energy_frac=1.5)
    with pytest.raises(ValueError):
        Class.SSA(rank_rule="svht_clip", svht_cap_frac=1.5)


def test_config_snapshot_contains_new_fields(ssa_signal):
    r = Class.SSA(window_size=256, stride=4).decompose(ssa_signal)
    data = r.config.to_dict()
    for key in ("window", "rank_rule", "energy_frac", "svht_cap_frac"):
        assert key in data
