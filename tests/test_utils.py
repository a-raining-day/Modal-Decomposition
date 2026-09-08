"""
Utility layer tests: validation, monotonicity, uniformity.
"""

import numpy as np
import pytest

from Modal_Decomposition.Utils import (
    Check_Time_and_Signal,
    detect_dtype,
    get_available_memory,
    get_memory_policy,
    is_monotonic,
    is_uniform,
    monotonic,
    require_ndim,
    set_absolute_limit,
    set_memmap_ratio,
    should_use_memmap,
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


# --- safe validation levels ---

def test_monotonic_safe_levels_match_on_valid_data():
    rng = np.random.default_rng(3)
    arr = np.sort(rng.standard_normal(5000))
    for safe in (0, 1, 2):
        assert monotonic(arr, safe=safe)
        assert monotonic(arr[::-1].copy(), safe=safe)
        assert not monotonic(rng.standard_normal(5000), safe=safe)
    arr2 = np.sort(rng.standard_normal(20_000))
    for safe in (0, 1, 2):
        assert monotonic(arr2, chunk_size=4096, safe=safe)


def test_monotonic_safe1_skips_nan_check():
    # safe=1: NaN is not detected; it silently fails the comparison.
    assert not monotonic(np.array([1.0, np.nan, 3.0]), safe=1)
    with pytest.raises(ValueError, match="NaN or Inf"):
        monotonic(np.array([1.0, np.nan, 3.0]), safe=0)
    with pytest.raises(ValueError, match="NaN or Inf"):
        monotonic(np.array([1.0, np.nan, 3.0]), safe=2)


def test_monotonic_safe2_raises_on_first_nan():
    # NaN inside a chunk and at a seam, chunked path.
    cs = 4
    a = np.arange(12, dtype=np.float64)
    a[5] = np.nan                       # inside the 2nd chunk
    with pytest.raises(ValueError, match="NaN or Inf"):
        monotonic(a, chunk_size=cs, safe=2)
    b = np.arange(12, dtype=np.float64)
    b[4] = np.nan                       # seam value (index cs)
    with pytest.raises(ValueError, match="NaN or Inf"):
        monotonic(b, chunk_size=cs, safe=2)
    c = np.arange(13, dtype=np.float64)
    c[-1] = np.inf                      # singleton tail chunk
    with pytest.raises(ValueError, match="NaN or Inf"):
        monotonic(c, chunk_size=cs, safe=2)
    # safe=2 single-shot: whole array is one chunk.
    with pytest.raises(ValueError, match="NaN or Inf"):
        monotonic(np.array([1.0, np.nan]), safe=2)


def test_monotonic_safe_invalid():
    with pytest.raises(ValueError, match="safe"):
        monotonic(np.array([1.0, 2.0]), safe=3)


# --- dtype detection ---

def test_detect_dtype():
    assert detect_dtype(np.array([True, False])) == "bool"
    assert detect_dtype(np.array([1, 2, 3])) == "int"
    assert detect_dtype(np.array([1], dtype=np.uint8)) == "int"
    assert detect_dtype(np.array([1.0])) == "float"
    assert detect_dtype(np.array([1.0], dtype=np.float16)) == "float"
    with pytest.raises(ValueError):
        detect_dtype(np.array([1 + 2j]))
    with pytest.raises(ValueError):
        detect_dtype(np.array(["a"]))


# --- memory policy ---

@pytest.fixture(autouse=True)
def _reset_memory_policy():
    yield
    set_memmap_ratio(0.6)


def test_memory_policy_strategies_are_exclusive():
    set_absolute_limit(1000)
    policy = get_memory_policy()
    assert policy["use_ratio_strategy"] is False
    assert policy["absolute_memmap_limit_bytes"] == 1000
    set_memmap_ratio(0.5)
    policy = get_memory_policy()
    assert policy["use_ratio_strategy"] is True
    assert policy["memmap_ratio_limit"] == 0.5


def test_memory_policy_validation():
    with pytest.raises(ValueError):
        set_memmap_ratio(0.0)
    with pytest.raises(ValueError):
        set_memmap_ratio(1.5)
    with pytest.raises(ValueError):
        set_absolute_limit(0)
    with pytest.raises(ValueError):
        set_absolute_limit(-1)


def test_should_use_memmap():
    set_absolute_limit(1000)
    assert should_use_memmap(999) is False
    assert should_use_memmap(1000) is True
    assert should_use_memmap(500, extra=500) is True


def test_get_available_memory():
    assert get_available_memory(force=True) > 0
    assert get_available_memory() > 0  # cached


@pytest.fixture()
def ws_tmp():
    """Workspace-local temp dir (pytest's tmp_path is blocked by the sandbox)."""
    import shutil
    from pathlib import Path

    d = Path(__file__).resolve().parent / ".pytest_tmp"
    d.mkdir(exist_ok=True)
    yield d
    shutil.rmtree(d, ignore_errors=True)


# --- input-layer size detection and memmap (Check_Time_and_Signal) ---

def test_check_time_and_signal_npy_path(ws_tmp):
    arr = np.sort(np.random.default_rng(2).standard_normal(3000))
    path = ws_tmp / "sig.npy"
    np.save(path, arr)
    S, T, N = Check_Time_and_Signal(str(path))
    assert S.shape == (3000,) and N == 3000 and S.dtype == np.float64
    assert np.allclose(S, arr)
    S2, _, _ = Check_Time_and_Signal(path)
    assert np.allclose(S2, arr)
    with pytest.raises(ValueError, match=".npy"):
        Check_Time_and_Signal(str(ws_tmp / "sig.txt"))


def test_check_time_and_signal_path_memmap_when_policy_triggers(ws_tmp):
    arr = np.sort(np.random.default_rng(4).standard_normal(3000))
    path = ws_tmp / "sig.npy"
    np.save(path, arr)
    set_absolute_limit(1)  # anything triggers memmap
    S, _, _ = Check_Time_and_Signal(str(path))
    assert isinstance(S, np.memmap)
    assert np.allclose(S, arr)


def test_check_time_and_signal_ndarray_memmap_when_policy_triggers(monkeypatch):
    set_absolute_limit(1)
    # 默认保留输入 dtype: int64 输入按 int64 落盘 (不再强制转 float64)。
    monkeypatch.setattr("Modal_Decomposition.Utils.Check._F64_DISK_BYTES", 1)
    S, _, _ = Check_Time_and_Signal(np.arange(10))
    assert isinstance(S, np.memmap)
    assert S.dtype == np.int64
    assert np.array_equal(S, np.arange(10))


def test_check_time_and_signal_long_list_becomes_memmap(monkeypatch):
    monkeypatch.setattr("Modal_Decomposition.Utils.Check._LIST_MEM_THRESHOLD", 5)
    S, _, _ = Check_Time_and_Signal(list(range(10)))
    assert isinstance(S, np.memmap)
    assert S.dtype == np.float64
    assert np.allclose(S, np.arange(10, dtype=np.float64))


def test_check_time_and_signal_non_f64_large_goes_disk(monkeypatch):
    """显式 dtype='float64' 时: f32 输入单趟直写 f64 memmap。"""
    monkeypatch.setattr("Modal_Decomposition.Utils.Check._F64_DISK_BYTES", 1)
    S, _, _ = Check_Time_and_Signal(np.arange(10, dtype=np.float32), dtype="float64")
    assert isinstance(S, np.memmap)
    assert S.dtype == np.float64
    assert np.allclose(S, np.arange(10, dtype=np.float64))


def test_monotonic_chunk_adaptation():
    # Tiny ratio limit -> input alone exceeds it -> chunk_size unchanged,
    # results stay correct.
    arr = np.sort(np.random.default_rng(5).standard_normal(200_000))
    set_memmap_ratio(1e-9)
    assert monotonic(arr, chunk_size=1_000_000)          # no shrink, single-shot
    assert monotonic(arr[::-1].copy(), chunk_size=1_000_000)
    assert not monotonic(np.random.default_rng(6).standard_normal(200_000),
                         chunk_size=1_000_000)
    # Absolute limit just above the input size -> chunk_size is shrunk to the
    # floor and the chunked path still gives correct results.
    set_absolute_limit(arr.nbytes + 1)
    assert monotonic(arr, chunk_size=1_000_000, strict=True)
    assert monotonic(arr[::-1].copy(), chunk_size=1_000_000)
    assert not monotonic(np.random.default_rng(7).standard_normal(200_000),
                         chunk_size=1_000_000)


# ------------------------------------------------------------------ #
# Check_Time_and_Signal 优化: dtype 保留 / 单趟转换 / 分块时间轴校验
# ------------------------------------------------------------------ #
def test_check_dtype_keep_preserves_input_dtype():
    for arr in (
        np.arange(8, dtype=np.float32),
        np.arange(8, dtype=np.int64),
        np.array([True, False, True]),
    ):
        S, _, N = Check_Time_and_Signal(arr, dtype=None)
        assert S.dtype == arr.dtype
        assert N == arr.size
        assert np.array_equal(S, arr)


def test_check_default_keeps_input_dtype():
    """默认 dtype=None: 不转 float64, 保留输入 dtype (省内存)。"""
    S, _, _ = Check_Time_and_Signal(np.arange(8, dtype=np.float32))
    assert S.dtype == np.float32
    S2, _, _ = Check_Time_and_Signal(np.arange(8, dtype=np.int16))
    assert S2.dtype == np.int16


def test_check_explicit_float64_converts():
    S, _, _ = Check_Time_and_Signal(np.arange(8, dtype=np.float32), dtype="float64")
    assert S.dtype == np.float64


def test_check_dtype_invalid_raises():
    with pytest.raises(ValueError, match="dtype"):
        Check_Time_and_Signal(np.arange(8.0), dtype="float32")


def test_check_dtype_keep_squeezes_and_keeps_dtype():
    S, _, N = Check_Time_and_Signal(
        np.arange(8, dtype=np.float32).reshape(1, -1), dtype=None
    )
    assert S.ndim == 1 and S.dtype == np.float32 and N == 8


def test_check_dtype_keep_rejects_non_numeric():
    for bad in (np.array(["a", "b"]), np.arange(4, dtype=np.complex128)):
        with pytest.raises(ValueError):
            Check_Time_and_Signal(bad, dtype=None)


def test_check_time_duplicates_chunked(monkeypatch):
    import Modal_Decomposition.Utils.Check as check_mod

    monkeypatch.setattr(check_mod, "_FILL_CHUNK_ELEMS", 5)
    T = np.arange(10.0)
    T[5] = 4.0  # 块宽 5 时的跨块接缝重复 (chunk0 末=4, chunk1 首=4)
    with pytest.raises(ValueError, match="duplicate"):
        Check_Time_and_Signal(np.arange(10.0), T)


def test_check_time_unsorted_chunked_warns_and_sorts(monkeypatch):
    import Modal_Decomposition.Utils.Check as check_mod

    monkeypatch.setattr(check_mod, "_FILL_CHUNK_ELEMS", 5)
    T = np.arange(10.0)[::-1].copy()
    with pytest.warns(UserWarning, match="reordered"):
        S, T2, _ = Check_Time_and_Signal(np.arange(10.0), T)
    assert np.array_equal(T2, np.arange(10.0))
    assert np.array_equal(S, np.arange(10.0)[::-1])


def test_check_single_pass_convert_f32_to_f64_memmap(monkeypatch):
    """显式 dtype='float64' + 内存策略触发时: f32 单趟直写 f64 memmap。"""
    import Modal_Decomposition.Utils.Check as check_mod

    monkeypatch.setattr(check_mod, "should_use_memmap", lambda nbytes, extra=0: True)
    S, _, _ = Check_Time_and_Signal(
        np.arange(20, dtype=np.float32), dtype="float64"
    )
    assert isinstance(S, np.memmap)
    assert S.dtype == np.float64
    assert np.allclose(S, np.arange(20.0))


def test_check_keep_dtype_with_memmap_policy(monkeypatch):
    import Modal_Decomposition.Utils.Check as check_mod

    monkeypatch.setattr(check_mod, "should_use_memmap", lambda nbytes, extra=0: True)
    S, _, _ = Check_Time_and_Signal(np.arange(20, dtype=np.float32), dtype=None)
    assert isinstance(S, np.memmap)
    assert S.dtype == np.float32
    assert np.allclose(S, np.arange(20.0))


def test_check_long_list_keep_dtype(monkeypatch):
    import Modal_Decomposition.Utils.Check as check_mod

    monkeypatch.setattr(check_mod, "_LIST_MEM_THRESHOLD", 5)
    S, _, _ = Check_Time_and_Signal([0.5, 1.5, 2.5, 3.5, 4.5, 5.5], dtype=None)
    assert isinstance(S, np.memmap)
    # Python list 本身无 dtype, keep 模式下退化为 float64
    assert S.dtype == np.float64
    assert np.allclose(S, np.arange(6) + 0.5)
