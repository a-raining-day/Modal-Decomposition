"""
Utility layer tests: validation, monotonicity, uniformity.
"""

import numpy as np
import pytest

from Modal_Decomposition.Utils import (
    Check_Time_and_Signal,
    is_monotonic,
    is_uniform,
    monotonic,
    require_ndim,
    to_signal,
)


# --- to_signal ---

def test_to_signal_from_list():
    out = to_signal([1, 2, 3])
    assert isinstance(out, np.ndarray)
    assert out.dtype == np.float64
    assert out.shape == (3,)


def test_to_signal_squeezes_singleton_dims():
    assert to_signal([[1.0, 2.0, 3.0]]).shape == (3,)


def test_to_signal_scalar_raises():
    with pytest.raises(ValueError):
        to_signal(3.0)


# --- require_ndim ---

def test_require_ndim_ok_and_fail():
    a = np.zeros(5)
    require_ndim(a, {1})
    with pytest.raises(ValueError, match="expected input dimension"):
        require_ndim(a, {2}, "X")


# --- Check_Time_and_Signal ---

def test_default_time_axis():
    S = np.arange(8, dtype=float)
    S2, T, N = Check_Time_and_Signal(S)
    assert N == 8
    assert np.allclose(T, np.arange(8))


def test_time_length_mismatch_raises():
    with pytest.raises(ValueError, match="Length mismatch"):
        Check_Time_and_Signal(np.zeros(8), T=np.arange(9))


def test_time_duplicates_raise():
    with pytest.raises(ValueError, match="duplicate"):
        Check_Time_and_Signal(np.zeros(4), T=[0.0, 1.0, 1.0, 2.0])


def test_unsorted_time_sorts_with_warning():
    S = np.array([10.0, 20.0, 30.0])
    T = np.array([2.0, 0.0, 1.0])
    with pytest.warns(UserWarning, match="reordered"):
        S2, T2, _ = Check_Time_and_Signal(S, T)
    assert np.allclose(T2, [0.0, 1.0, 2.0])
    assert np.allclose(S2, [20.0, 30.0, 10.0])


def test_ndim_enforcement():
    with pytest.raises(ValueError):
        Check_Time_and_Signal(np.zeros((2, 3)), ndim={1}, method="X")


def test_empty_signal_raises():
    with pytest.raises(ValueError, match="empty"):
        Check_Time_and_Signal(np.array([]))


def test_multichannel_sort_last_axis():
    S = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    T = np.array([2.0, 0.0, 1.0])
    with pytest.warns(UserWarning):
        S2, T2, _ = Check_Time_and_Signal(S, T)
    assert np.allclose(S2[0], [2.0, 3.0, 1.0])
    assert np.allclose(S2[1], [5.0, 6.0, 4.0])


# --- is_uniform ---

def test_is_uniform():
    assert is_uniform(np.arange(5))
    assert is_uniform(np.array([0.5]))
    assert not is_uniform(np.array([0.0, 1.0, 2.5]))


# --- monotonic / is_monotonic ---

def test_monotonic_increasing():
    assert monotonic(np.array([1.0, 2.0, 2.0, 3.0]))


def test_monotonic_decreasing():
    assert monotonic(np.array([3.0, 2.0, 1.0]))


def test_monotonic_non_monotonic():
    assert not monotonic(np.array([1.0, 3.0, 2.0]))


def test_monotonic_strict():
    assert monotonic(np.array([1.0, 2.0, 3.0]), strict=True)
    assert not monotonic(np.array([1.0, 2.0, 2.0]), strict=True)


def test_monotonic_directional_mod():
    assert monotonic(np.array([1.0, 2.0, 3.0]), mod="increasing")
    assert not monotonic(np.array([1.0, 2.0, 3.0]), mod="decreasing")
    assert monotonic(np.array([3.0, 2.0, 1.0]), mod="decreasing")
    assert not monotonic(np.array([3.0, 2.0, 1.0]), mod="increasing")


def test_monotonic_single_element():
    assert monotonic(np.array([5.0]))


def test_monotonic_nan_raises():
    with pytest.raises(ValueError, match="NaN or Inf"):
        monotonic(np.array([1.0, np.nan, 3.0]))


def test_monotonic_wrong_dim_raises():
    with pytest.raises(ValueError):
        monotonic(np.zeros((2, 2)))


def test_monotonic_empty_raises():
    with pytest.raises(ValueError):
        monotonic(np.array([]))


def test_is_monotonic_alias():
    assert is_monotonic(np.array([3.0, 2.0, 1.0]))
    assert not is_monotonic(np.array([1.0, 3.0, 2.0]))


def test_monotonic_chunked_path():
    rng = np.random.default_rng(1)
    arr = np.sort(rng.standard_normal(2_000_000))
    assert monotonic(arr)
    arr2 = arr.copy()
    arr2[1_000_000] = arr2[0] - 1
    assert not monotonic(arr2)


# --- Equal semantics (constant sequences) ---

def test_monotonic_constant_non_strict():
    # A constant sequence satisfies every direction in non-strict mode.
    assert monotonic(np.array([2.0, 2.0, 2.0]))
    assert monotonic(np.array([2.0] * 5), mod="increasing")
    assert monotonic(np.array([2.0] * 5), mod="decreasing")
    assert is_monotonic(np.array([2.0] * 5))


def test_monotonic_constant_strict():
    # A constant sequence is not strictly monotonic.
    assert not monotonic(np.array([2.0, 2.0, 2.0]), strict=True)
    assert not monotonic(np.array([2.0] * 5), strict=True, mod="increasing")
    assert not monotonic(np.array([2.0] * 5), strict=True, mod="decreasing")
    assert not is_monotonic(np.array([2.0] * 5), strict=True)


def test_monotonic_chunked_constant():
    cs = 1024
    arr = np.full(3 * cs + 7, 1.5)
    assert monotonic(arr, chunk_size=cs)
    assert not monotonic(arr, strict=True, chunk_size=cs)
    assert is_monotonic(arr, chunk_size=cs)


# --- Equal boundaries defer the direction decision to the chunks ---

def test_monotonic_chunked_equal_seam_ramp_plateau():
    # Regression: rising first chunk + plateau tail crossing the seam.
    cs = 1024
    arr = np.concatenate([np.arange(cs, dtype=np.float64), np.full(50, cs - 1.0)])
    assert monotonic(arr, chunk_size=cs)
    # Mirror case: falling ramp + plateau tail.
    dec = np.concatenate([np.arange(cs, 0, -1, dtype=np.float64), np.full(50, 1.0)])
    assert monotonic(dec, chunk_size=cs)


def test_monotonic_chunked_equal_seam_ties_inside_chunk():
    # Non-decreasing with ties inside chunks, flat tail, ties at the seams.
    cs = 4
    arr = np.array([0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0, 3.0])
    assert monotonic(arr, chunk_size=cs)
    dec = arr[::-1].copy()
    assert monotonic(dec, chunk_size=cs)


def test_monotonic_chunked_wiggle_under_equal_seams():
    # A wiggle chunk under equal seams must not be silently ignored.
    cs = 4
    arr = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 3.0, 1.0, 1.0, 2.0, 3.0, 4.0])
    assert not monotonic(arr, chunk_size=cs)


def test_monotonic_chunked_direction_flip():
    # Up, then down, with ties at the seams -> not monotone.
    cs = 4
    arr = np.array([0.0, 1.0, 2.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 2.0, 1.0, 0.0])
    assert not monotonic(arr, chunk_size=cs)


def test_monotonic_chunked_singleton_tail():
    # Last chunk of size 1 must not break strict monotonicity.
    cs = 1024
    arr = np.arange(cs + 1, dtype=np.float64)
    assert monotonic(arr, chunk_size=cs)
    assert monotonic(arr, strict=True, chunk_size=cs)


def test_monotonic_chunked_strict_tie_rejected():
    cs = 4
    arr = np.array([0.0, 1.0, 2.0, 3.0, 3.0, 4.0, 5.0, 6.0])
    assert not monotonic(arr, strict=True, chunk_size=cs)


# --- dtype preservation (no float64 upcast) ---

def test_monotonic_preserves_input_dtype():
    assert monotonic(np.array([1, 2, 2, 3]))                 # int64
    assert monotonic(np.array([3, 2, 1]), mod="decreasing")  # int64
    assert not monotonic(np.array([1, 3, 2]))                # int64
    assert monotonic(np.arange(10, dtype=np.float16))
    assert monotonic(np.arange(10, dtype=np.float32))
    assert monotonic(np.arange(10, dtype=np.uint8), mod="increasing")
    assert not monotonic(np.array([True, False, True]))      # bool


def test_monotonic_float16_chunked():
    cs = 4
    arr = np.arange(12, dtype=np.float16)
    assert monotonic(arr, chunk_size=cs)
    assert monotonic(arr, strict=True, chunk_size=cs)
    assert monotonic(arr[::-1].copy(), chunk_size=cs)        # decreasing


def test_monotonic_strict_ties_any_dtype():
    assert not monotonic(np.array([1, 1, 2], dtype=np.float16), strict=True)
    assert not monotonic(np.array([1, 1, 2], dtype=np.int64), strict=True)
    assert monotonic(np.array([1, 1, 2], dtype=np.float16))


def test_monotonic_nan_float16_raises():
    with pytest.raises(ValueError, match="NaN or Inf"):
        monotonic(np.array([1.0, np.nan, 3.0], dtype=np.float16))


def test_monotonic_non_numeric_raises():
    with pytest.raises(ValueError):
        monotonic(np.array(["a", "b"]))
    with pytest.raises(ValueError):
        monotonic(np.array([1 + 2j, 3 + 4j]))
    with pytest.raises(ValueError):
        monotonic(np.array([np.datetime64("2020-01-01"), np.datetime64("2020-01-02")]))
