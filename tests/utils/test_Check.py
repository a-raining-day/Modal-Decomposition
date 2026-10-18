"""
``Modal_Decomposition.Utils.Check`` 模块测试。

覆盖：导出一致性、`validate_mod` / `validate_safe`、`to_array` / `to_dtype`、
各类 `is_*` / `require_*` 判据、`not_numpy_arr` / `not_numpy_memmap`（本轮修过的
缺陷点）、`Check_Time_and_Signal` 的 `need_T` 分支，以及边界输入。
"""

import tempfile

import numpy as np
import pytest

from Modal_Decomposition.Define import SAFE_MODS
from Modal_Decomposition.Utils import PEAKS_MODS, Check, Check_Time_and_Signal


# --------------------------------------------------------------------------- #
# 导出一致性
# --------------------------------------------------------------------------- #
def test_public_surface():
    """`Check` 与 `Check_Time_and_Signal` 都能从 Utils 顶层取到；Check 的方法齐备。

    注: ``Utils`` 包本身**没有** ``__all__``（各子模块各自维护），所以这里按属性
    存在性断言，而不是查 ``__all__``。
    """
    import Modal_Decomposition.Utils as U
    assert U.Check is Check
    assert callable(U.Check_Time_and_Signal)
    for name in ("validate_mod", "validate_safe", "to_array", "to_dtype",
                 "not_numpy_arr", "not_numpy_memmap", "require_ndim", "is_uniform"):
        assert hasattr(Check, name), f"Check.{name} 缺失"


# --------------------------------------------------------------------------- #
# validate_mod / validate_safe
# --------------------------------------------------------------------------- #
def test_validate_mod_string_and_enum():
    """字符串与 Enum 成员都能通过，且返回 Enum 成员（幂等）。"""
    assert Check.validate_mod("both", PEAKS_MODS) is PEAKS_MODS.Both
    assert Check.validate_mod(PEAKS_MODS.Max, PEAKS_MODS) is PEAKS_MODS.Max


def test_validate_mod_rejects_unknown():
    with pytest.raises(ValueError):
        Check.validate_mod("not-a-mode", PEAKS_MODS)


def test_validate_safe():
    for s in ("safe", "suspicious", "unknown"):
        assert Check.validate_safe(s) is SAFE_MODS(s)
    with pytest.raises(ValueError):
        Check.validate_safe("nope")


# --------------------------------------------------------------------------- #
# to_array / to_dtype
# --------------------------------------------------------------------------- #
def test_to_array_accepts_list():
    out = Check.to_array([1, 2, 3], ndim=1)
    assert isinstance(out, np.ndarray) and out.shape == (3,)


def test_to_array_rejects_wrong_ndim():
    with pytest.raises((ValueError, TypeError)):
        Check.to_array(np.zeros((2, 8)), ndim=1)


def test_to_array_rejects_non_numeric():
    with pytest.raises((TypeError, ValueError)):
        Check.to_array(np.array(["a", "b"]), ndim=1)


def test_to_dtype():
    out = Check.to_dtype(np.array([1.0, 2.0]), np.float32)
    assert out.dtype == np.float32
    assert np.allclose(out, [1.0, 2.0])


# --------------------------------------------------------------------------- #
# not_numpy_arr / not_numpy_memmap —— 本轮修复的缺陷点
# --------------------------------------------------------------------------- #
def test_not_numpy_arr():
    """普通 ndarray 通过，非 ndarray 抛错。"""
    assert Check.not_numpy_arr(np.zeros(4)) is True
    with pytest.raises(TypeError):
        Check.not_numpy_arr([0.0, 1.0])


def test_not_numpy_memmap_accepts_plain_ndarray():
    """回归：普通 ndarray **必须**通过（修复前这里会抛 TypeError）。"""
    assert Check.not_numpy_memmap(np.zeros(4)) is True


def test_not_numpy_memmap_rejects_memmap():
    """真正的 memmap 才被拒绝。"""
    with tempfile.TemporaryDirectory() as d:
        mm = np.memmap(f"{d}/m.dat", dtype=np.float64, mode="w+", shape=(8,))
        mm[:] = 1.0
        try:
            with pytest.raises(TypeError):
                Check.not_numpy_memmap(mm)
        finally:
            del mm


# --------------------------------------------------------------------------- #
# 判据族
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("arr,expected", [
    (np.arange(5.0), True),
    (np.array([...]), False),          # Ellipsis 不是数值数组
])
def test_is_numeric(arr, expected):
    assert bool(Check.is_numeric(arr)) is expected


def test_is_uniform():
    assert Check.is_uniform(np.arange(10.0)) is True
    assert Check.is_uniform(np.array([0.0, 1.0, 3.0])) is False


def test_require_ndim_and_nonempty():
    Check.require_ndim(np.zeros(4), {1})
    with pytest.raises((ValueError, TypeError)):
        Check.require_ndim(np.zeros((2, 4)), {1})
    with pytest.raises((ValueError, TypeError)):
        Check.require_nonempty(np.array([]))


# --------------------------------------------------------------------------- #
# Check_Time_and_Signal
# --------------------------------------------------------------------------- #
def test_ctas_need_t_false_returns_pair():
    S, N = Check_Time_and_Signal(np.arange(16.0), T=None, ndim={1}, method="T", need_T=False)
    assert S.shape == (16,) and N == 16


def test_ctas_need_t_true_returns_triple():
    S, T, N = Check_Time_and_Signal(np.arange(16.0), T=None, ndim={1}, method="T", need_T=True)
    assert S.shape == (16,) and T.shape == (16,) and N == 16


def test_ctas_copy_isolation():
    """copy=True 时返回的是独立内存，改它不影响调用方数组。"""
    src = np.arange(8.0)
    out, _ = Check_Time_and_Signal(src, T=None, ndim={1}, method="T", need_T=False, copy=True)
    out += 100.0
    assert not np.array_equal(src, out)
    assert np.array_equal(src, np.arange(8.0))


def test_ctas_rejects_2d():
    with pytest.raises((ValueError, TypeError)):
        Check_Time_and_Signal(np.zeros((2, 8)), T=None, ndim={1}, method="T", need_T=False)


def test_ctas_dtype_argument():
    out, _ = Check_Time_and_Signal(np.arange(8.0), T=None, ndim={1}, method="T",
                                   need_T=False, dtype=np.float32)
    assert out.dtype == np.float32


def test_ctas_dtype_policy():
    """整型输入**保持整型**（不隐式提升）；list 输入转成 float64。

    浮点化是各分解方法自己做的一步（``Check.to_dtype``），不在统一入参层。
    """
    out_i, _ = Check_Time_and_Signal(np.arange(8, dtype=np.int32), T=None, ndim={1},
                                     method="T", need_T=False)
    assert out_i.dtype == np.int32
    out_l, _ = Check_Time_and_Signal([1, 2, 3], T=None, ndim={1}, method="T", need_T=False)
    assert out_l.dtype == np.float64
