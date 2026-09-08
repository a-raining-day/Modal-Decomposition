"""
SSA stride tests: stride in {1, 2, 4, 8, 16, 32, 64, 128, 256}.

Covers:
* exact reconstruction (sum of ALL components == signal) for every stride,
* stride=1 being bitwise identical to the default (classical Hankel),
* component count shrinking as stride grows,
* tone tracking of the first component for non-aliasing strides,
* parameter validation (non-positive / non-integer stride, L < stride).
"""

import numpy as np
import pytest

from Modal_Decomposition import Class

STRIDES = [1, 2, 4, 8, 16, 32, 64, 128, 256]


@pytest.fixture(scope="module")
def ssa_signal():
    """Short two-tone + noise signal (2048 samples, 37 Hz / 113 Hz)."""
    rng = np.random.default_rng(0)
    t = np.arange(2048) / 1000.0
    s = (
        np.sin(2 * np.pi * 37.0 * t)
        + 0.6 * np.sin(2 * np.pi * 113.0 * t)
        + 0.05 * rng.standard_normal(2048)
    )
    return s, t


@pytest.mark.parametrize("stride", STRIDES)
def test_ssa_reconstruction_exact(ssa_signal, stride):
    s, _ = ssa_signal
    r = Class.SSA(stride=stride).decompose(s)
    assert r.IMFs.ndim == 2
    assert r.IMFs.shape[-1] == s.size
    # full decomposition (groups=None -> all components) must reconstruct S
    assert np.max(np.abs(r.reconstruct() - s)) < 1e-9
    assert r.config.stride == stride


def test_ssa_stride1_matches_default(ssa_signal):
    s, _ = ssa_signal
    a = Class.SSA().decompose(s)
    b = Class.SSA(stride=1).decompose(s)
    assert np.array_equal(a.IMFs, b.IMFs)


def test_ssa_stride_shrinks_component_count(ssa_signal):
    s, _ = ssa_signal
    counts = [Class.SSA(stride=st).decompose(s).IMFs.shape[0] for st in STRIDES]
    assert all(c1 >= c2 for c1, c2 in zip(counts, counts[1:]))
    assert counts[0] > counts[-1]


@pytest.mark.parametrize("stride", [1, 4, 16, 64])
def test_ssa_first_component_tracks_tone(ssa_signal, stride):
    s, t = ssa_signal
    tone = np.sin(2 * np.pi * 37.0 * t)
    r = Class.SSA(stride=stride).decompose(s)
    corr = float(np.corrcoef(r.IMFs[0], tone)[0, 1])
    assert corr > 0.9, f"stride={stride}: first-component corr {corr:.3f}"


@pytest.mark.parametrize("bad", [0, -2, 2.5])
def test_ssa_bad_stride_raises(bad):
    with pytest.raises(ValueError):
        Class.SSA(stride=bad)


def test_ssa_window_smaller_than_stride_raises():
    # N=16 -> default L = 5 < 8: some samples would never be covered
    with pytest.raises(ValueError):
        Class.SSA(stride=8).decompose(np.arange(16.0))


# --- window_size (= patch length) x stride combos -------------------------


def test_ssa_defaults_are_hankel(ssa_signal):
    s, _ = ssa_signal
    r = Class.SSA().decompose(s)
    # defaults: window_size = N // 3 (the patch length), stride = 1
    # -> consecutive patches differ by one sample = classical Hankel matrix
    assert r.config.window_size == s.size // 3
    assert r.config.stride == 1
    assert np.array_equal(r.IMFs, Class.SSA(stride=1).decompose(s).IMFs)


@pytest.mark.parametrize(
    "window_size,stride",
    [(64, 1), (64, 4), (200, 8), (400, 32),
     (37, 1), (37, 4), (37, 16), (32, 16), (16, 16),
     (8, 4), (4, 4), (4, 1), (1, 1)],
)
def test_ssa_window_size_stride_combos_reconstruct(ssa_signal, window_size, stride):
    s, _ = ssa_signal
    r = Class.SSA(window_size=window_size, stride=stride).decompose(s)
    assert r.config.window_size == window_size
    assert r.config.stride == stride
    assert np.max(np.abs(r.reconstruct() - s)) < 1e-9


@pytest.mark.parametrize("window_size,stride", [(8, 16), (4, 16), (37, 64), (16, 32)])
def test_ssa_stride_exceeding_window_size_raises(ssa_signal, window_size, stride):
    s, _ = ssa_signal
    with pytest.raises(ValueError):
        Class.SSA(window_size=window_size, stride=stride).decompose(s)


# --- boundary conditions: 1 <= window_size < N (no N // 2 cap) ---------------


def test_ssa_window_size_n_minus_one_allowed():
    s = np.arange(16, dtype=np.float64)
    r = Class.SSA(window_size=15, stride=1).decompose(s)
    assert r.config.window_size == 15
    assert np.max(np.abs(r.reconstruct() - s)) < 1e-9


@pytest.mark.parametrize("window_size", [16, 17, 100])
def test_ssa_window_size_not_less_than_n_raises(window_size):
    s = np.arange(16, dtype=np.float64)
    with pytest.raises(ValueError):
        Class.SSA(window_size=window_size).decompose(s)


@pytest.mark.parametrize("window_size", [0, -3, 2.5])
def test_ssa_window_size_invalid_values_raise(window_size):
    with pytest.raises(ValueError):
        Class.SSA(window_size=window_size)


def test_ssa_single_sample_signal_rejected():
    # length-1 signals are rejected by the shared input layer (0-d after
    # squeeze), so SSA never sees N == 1
    with pytest.raises(ValueError):
        Class.SSA().decompose(np.array([3.0]))
