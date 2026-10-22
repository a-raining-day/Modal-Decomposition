"""
``Modal_Decomposition.Utils.Monotonicity`` 的一模块一测试。

被测模块: ``src/Modal_Decomposition/Utils/Monotonicity.py``
配套报告: ``docs/test/utils/Monotonicity_Engine_Report.md``

覆盖的契约(与 ``gen_tool_doc.py`` 里 ``Utils`` 契约一一对应):
  1. ``__all__`` 覆盖 ``monotonic`` / ``give_monotony`` / ``MONOTONY_MODS`` /
     ``MONOTONY_MODS_ARGS`` / ``Monotony``;
  2. ``mod`` 在 5 组代表数组上, 对 ``increasing`` / ``decreasing`` /
     ``monotonic`` 三种取值的真值表(含 ``strict=False/True``);
  3. ``strict=True`` 与 ``strict=False`` 在带相等项数组上结果不同;
  4. ``mod`` 同时接受字符串与 ``MONOTONY_MODS`` 成员, 结果一致;
  5. 非法 ``mod`` 抛 ``ValueError``;
  6. ``give_monotony`` 返回 ``Monotony`` 成员, 且与 ``monotonic`` 的判断自洽;
  7. 边界输入不崩(空数组 / 长度 1 / 全 NaN / 二维 / 非法 ``chunk_size``);
  8. 不改写调用方数组且 dtype 保持不变; 分块路径与小数组路径结果一致。

所有断言均按**实测行为**书写(见 ``_probe`` 记录的返回值), 不含猜测。
"""

import numpy as np
import pytest

from Modal_Decomposition.Utils import (
    MONOTONY_MODS,
    MONOTONY_MODS_ARGS,
    Monotony,
    give_monotony,
    monotonic,
)
from Modal_Decomposition.Utils import Monotonicity as MonotonicityModule


# --------------------------------------------------------------------------- #
# 夹具: 5 组代表数组 + 真值表
# --------------------------------------------------------------------------- #
STRICT_INC = np.array([1, 2, 3, 4, 5])
STRICT_DEC = np.array([5, 4, 3, 2, 1])
INC_WITH_EQUALS = np.array([1, 2, 2, 3, 5])
RISE_THEN_FALL = np.array([0, 1, 0])
CONSTANT = np.array([7, 7, 7, 7])

# key -> (数组, give_monotony 实测返回, 三种 mod 的 (strict=False, strict=True) 真值)
CASES = {
    "strict_increasing": (
        STRICT_INC,
        Monotony.StrictIncreasing,
        {"increasing": (True, True), "decreasing": (False, False), "monotonic": (True, True)},
    ),
    "strict_decreasing": (
        STRICT_DEC,
        Monotony.StrictDecreasing,
        {"increasing": (False, False), "decreasing": (True, True), "monotonic": (True, True)},
    ),
    "increasing_with_equals": (
        INC_WITH_EQUALS,
        Monotony.MixedIncreasing,
        {"increasing": (True, False), "decreasing": (False, False), "monotonic": (True, False)},
    ),
    "rise_then_fall": (
        RISE_THEN_FALL,
        Monotony.No,
        {"increasing": (False, False), "decreasing": (False, False), "monotonic": (False, False)},
    ),
    "constant": (
        CONSTANT,
        Monotony.Equal,
        {"increasing": (True, False), "decreasing": (True, False), "monotonic": (True, False)},
    ),
}

CASE_IDS = list(CASES)
MODS = ["increasing", "decreasing", "monotonic"]

# non-strict 下"可视为单调"的 give_monotony 结果集合(自洽性用)
_NON_STRICT_OK = {
    Monotony.StrictIncreasing,
    Monotony.StrictDecreasing,
    Monotony.MixedIncreasing,
    Monotony.MixedDecreasing,
    Monotony.Equal,
}
# 与某个方向明确相容的 give_monotony 结果集合
_INCREASING_OK = {Monotony.StrictIncreasing, Monotony.MixedIncreasing, Monotony.Equal}
_DECREASING_OK = {Monotony.StrictDecreasing, Monotony.MixedDecreasing, Monotony.Equal}


# --------------------------------------------------------------------------- #
# 1. 公开导出
# --------------------------------------------------------------------------- #
def test_all_exports_are_complete():
    """``__all__`` 恰好覆盖 5 个公开名字。"""
    expected = {"monotonic", "give_monotony", "MONOTONY_MODS", "MONOTONY_MODS_ARGS", "Monotony"}
    assert set(MonotonicityModule.__all__) == expected
    assert len(MonotonicityModule.__all__) == 5


@pytest.mark.parametrize(
    "name",
    ["monotonic", "give_monotony", "MONOTONY_MODS", "MONOTONY_MODS_ARGS", "Monotony"],
)
def test_all_exports_are_resolvable(name):
    """``__all__`` 里的每个名字都能在模块与包命名空间里取到。"""
    assert hasattr(MonotonicityModule, name)
    assert getattr(MonotonicityModule, name) is not None


def test_public_imports_from_package():
    """包级导入可用, 且与模块内是同一个对象。"""
    from Modal_Decomposition.Utils import Monotonicity as mod

    assert monotonic is mod.monotonic
    assert give_monotony is mod.give_monotony
    assert Monotony is mod.Monotony
    assert MONOTONY_MODS is mod.MONOTONY_MODS
    assert MONOTONY_MODS_ARGS is mod.MONOTONY_MODS_ARGS


def test_monotony_enum_members_and_encoding():
    """``Monotony`` 的 6 个成员及其 3 位二进制编码。"""
    assert [m.name for m in Monotony] == [
        "Equal",
        "MixedIncreasing",
        "MixedDecreasing",
        "StrictIncreasing",
        "StrictDecreasing",
        "No",
    ]
    assert Monotony.Equal.value == 0b000
    assert Monotony.MixedIncreasing.value == 0b001
    assert Monotony.MixedDecreasing.value == 0b010
    assert Monotony.StrictIncreasing.value == 0b101
    assert Monotony.StrictDecreasing.value == 0b110
    assert Monotony.No.value == 0b111


def test_monotony_mods_members():
    """``MONOTONY_MODS`` 的三个方向成员。"""
    assert [(m.name, m.value) for m in MONOTONY_MODS] == [
        ("Increasing", "increasing"),
        ("Decreasing", "decreasing"),
        ("Monotonic", "monotonic"),
    ]


# --------------------------------------------------------------------------- #
# 2. 三种 mod × 5 组数组 × 两种 strict 的真值表
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("case", CASE_IDS, ids=CASE_IDS)
@pytest.mark.parametrize("mod", MODS)
def test_mod_truth_table(case, mod):
    arr, _expected_give, table = CASES[case]
    non_strict, strict = table[mod]
    assert monotonic(arr, strict=False, mod=mod) is non_strict
    assert monotonic(arr, strict=True, mod=mod) is strict


@pytest.mark.parametrize("case", CASE_IDS, ids=CASE_IDS)
def test_default_mod_is_monotonic(case):
    """``mod`` 默认值是 ``"monotonic"``, 与显式传入等价。"""
    arr = CASES[case][0]
    for strict in (False, True):
        assert monotonic(arr, strict=strict) == monotonic(arr, strict=strict, mod="monotonic")


def test_increasing_and_decreasing_are_mutually_exclusive_on_strict_arrays():
    """严格单调的数组只会满足一个方向。"""
    assert monotonic(STRICT_INC, mod="increasing") and not monotonic(STRICT_INC, mod="decreasing")
    assert monotonic(STRICT_DEC, mod="decreasing") and not monotonic(STRICT_DEC, mod="increasing")


def test_monotonic_mod_accepts_either_direction():
    """``mod="monotonic"`` 对严格递增/严格递减都返回 True。"""
    assert monotonic(STRICT_INC, mod="monotonic") is True
    assert monotonic(STRICT_DEC, mod="monotonic") is True
    assert monotonic(RISE_THEN_FALL, mod="monotonic") is False


# --------------------------------------------------------------------------- #
# 3. strict=True / strict=False 在带相等项数组上结果不同
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("case", CASE_IDS, ids=CASE_IDS)
@pytest.mark.parametrize("mod", MODS)
def test_strict_implies_non_strict(case, mod):
    """``strict=True`` 是更强的约束: 通过严格判定则必然通过非严格判定。"""
    arr = CASES[case][0]
    if monotonic(arr, strict=True, mod=mod):
        assert monotonic(arr, strict=False, mod=mod) is True


@pytest.mark.parametrize(
    "case,mod",
    [
        ("increasing_with_equals", "increasing"),
        ("increasing_with_equals", "monotonic"),
        ("constant", "increasing"),
        ("constant", "decreasing"),
        ("constant", "monotonic"),
    ],
    ids=[
        "inc_equals-increasing",
        "inc_equals-monotonic",
        "constant-increasing",
        "constant-decreasing",
        "constant-monotonic",
    ],
)
def test_strict_differs_from_non_strict_on_equal_containing_array(case, mod):
    """含相等项的数组: 非严格通过, 严格不通过(两种结果**不同**)。"""
    arr = CASES[case][0]
    assert monotonic(arr, strict=False, mod=mod) is True
    assert monotonic(arr, strict=True, mod=mod) is False


def test_strict_is_noop_on_fully_strict_arrays():
    """数组本身严格单调时, strict 开关不改变结果。"""
    for arr in (STRICT_INC, STRICT_DEC):
        for mod in MODS:
            assert monotonic(arr, strict=False, mod=mod) == monotonic(arr, strict=True, mod=mod)


def test_strict_switch_on_minimal_equal_pair():
    """最小规模的相等项: ``[4, 4]``。"""
    pair = np.array([4.0, 4.0])
    assert monotonic(pair, strict=False, mod="increasing") is True
    assert monotonic(pair, strict=False, mod="decreasing") is True
    assert monotonic(pair, strict=True, mod="increasing") is False
    assert monotonic(pair, strict=True, mod="decreasing") is False


# --------------------------------------------------------------------------- #
# 4. 字符串与 Enum 成员两种写法等价
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("case", CASE_IDS, ids=CASE_IDS)
@pytest.mark.parametrize("strict", [False, True], ids=["non_strict", "strict"])
def test_string_and_enum_mod_agree(case, strict):
    arr = CASES[case][0]
    for mod_str, mod_enum in (
        ("increasing", MONOTONY_MODS.Increasing),
        ("decreasing", MONOTONY_MODS.Decreasing),
        ("monotonic", MONOTONY_MODS.Monotonic),
    ):
        assert monotonic(arr, strict=strict, mod=mod_str) == monotonic(
            arr, strict=strict, mod=mod_enum
        )


@pytest.mark.parametrize("case", CASE_IDS, ids=CASE_IDS)
def test_enum_mod_matches_expected_table(case):
    """``MONOTONY_MODS`` 成员给出的结果与真值表一致。"""
    arr, _give, table = CASES[case]
    for member in MONOTONY_MODS:
        non_strict, strict = table[member.value]
        assert monotonic(arr, strict=False, mod=member) is non_strict
        assert monotonic(arr, strict=True, mod=member) is strict


# --------------------------------------------------------------------------- #
# 5. 非法 mod 抛 ValueError
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "bad_mod",
    ["up", "Increasing", "", "strictly_increasing", "monotonic ", 1],
    ids=["up", "title_case", "empty", "strictly_", "trailing_space", "int"],
)
def test_invalid_mod_raises_value_error(bad_mod):
    with pytest.raises(ValueError):
        monotonic(STRICT_INC, mod=bad_mod)


def test_invalid_mod_message_mentions_allowed_values():
    """错误信息里列出了合法取值, 便于调用方自查。"""
    with pytest.raises(ValueError) as excinfo:
        monotonic(STRICT_INC, mod="nope")
    message = str(excinfo.value)
    for token in ("Increasing", "Decreasing", "Monotonic"):
        assert token in message


def test_invalid_mod_rejected_before_array_handling():
    """非法 mod 在二维输入上也先报 mod 错误(参数校验在最前)。"""
    with pytest.raises(ValueError) as excinfo:
        monotonic(np.zeros((2, 3)), mod="nope")
    assert "mod" in str(excinfo.value)


# --------------------------------------------------------------------------- #
# 6. give_monotony 的返回值与自洽性
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("case", CASE_IDS, ids=CASE_IDS)
def test_give_monotony_returns_expected_member(case):
    arr, expected, _table = CASES[case]
    got = give_monotony(arr)
    assert isinstance(got, Monotony)
    assert got is expected


@pytest.mark.parametrize("case", CASE_IDS, ids=CASE_IDS)
def test_give_monotony_is_consistent_with_monotonic(case):
    """give_monotony 的分类与 monotonic 的判断互相自洽。"""
    arr = CASES[case][0]
    m = give_monotony(arr)

    # (a) mod="monotonic" 非严格: 只要不是 No 就应通过
    assert monotonic(arr, strict=False, mod="monotonic") is (m is not Monotony.No)
    assert (m is not Monotony.No) == (m in _NON_STRICT_OK)

    # (b) mod="monotonic" 严格: 只有两个 Strict* 成员才应通过
    strictly = m in (Monotony.StrictIncreasing, Monotony.StrictDecreasing)
    assert monotonic(arr, strict=True, mod="monotonic") is strictly

    # (c) 方向自洽
    assert monotonic(arr, strict=False, mod="increasing") is (m in _INCREASING_OK)
    assert monotonic(arr, strict=False, mod="decreasing") is (m in _DECREASING_OK)

    # (d) 严格方向自洽
    assert monotonic(arr, strict=True, mod="increasing") is (m is Monotony.StrictIncreasing)
    assert monotonic(arr, strict=True, mod="decreasing") is (m is Monotony.StrictDecreasing)


def test_give_monotony_mixed_decreasing():
    """含相等项的递减数组 -> ``MixedDecreasing``(补充的第 6 组)。"""
    arr = np.array([5, 4, 4, 3, 3, 1])
    assert give_monotony(arr) is Monotony.MixedDecreasing
    assert monotonic(arr, strict=False, mod="decreasing") is True
    assert monotonic(arr, strict=False, mod="increasing") is False
    assert monotonic(arr, strict=True, mod="decreasing") is False


def test_give_monotony_never_returns_no_for_monotone_input():
    """``arange`` 与 ``arange[::-1]`` 分别落在两个 Strict* 成员上。"""
    assert give_monotony(np.arange(5)) is Monotony.StrictIncreasing
    assert give_monotony(np.arange(5)[::-1]) is Monotony.StrictDecreasing


def test_give_monotony_accepts_lists_and_tuples():
    """非 ndarray 输入也能被转换后分类。"""
    assert give_monotony([1, 2, 3]) is Monotony.StrictIncreasing
    assert give_monotony((3, 2, 1)) is Monotony.StrictDecreasing
    assert monotonic([1, 2, 3], mod="increasing") is True


# --------------------------------------------------------------------------- #
# 7. 边界输入
# --------------------------------------------------------------------------- #
def test_empty_array():
    """空数组: 分类为 Equal; 明确方向恒为 True, 严格 ``monotonic`` 为 False。"""
    empty = np.array([])
    assert give_monotony(empty) is Monotony.Equal
    assert monotonic(empty) is True
    assert monotonic(empty, mod="increasing") is True
    assert monotonic(empty, mod="decreasing") is True
    assert monotonic(empty, strict=True, mod="increasing") is True
    assert monotonic(empty, strict=True, mod="decreasing") is True
    # 空序列不是"严格单调"的(没有相邻对可以严格递增)
    assert monotonic(empty, strict=True, mod="monotonic") is False


def test_single_element_array():
    """长度 1: 行为与空数组一致(都为 Equal / 空真)。"""
    single = np.array([3.0])
    assert give_monotony(single) is Monotony.Equal
    assert monotonic(single) is True
    assert monotonic(single, mod="increasing") is True
    assert monotonic(single, mod="decreasing") is True
    assert monotonic(single, strict=True, mod="increasing") is True
    assert monotonic(single, strict=True, mod="monotonic") is False


def test_all_nan_raises_at_default_safe_level():
    """默认 ``safe="unknown"`` 会做有限性检查, 全 NaN 直接抛 ValueError。"""
    nan_arr = np.full(4, np.nan)
    with pytest.raises(ValueError):
        give_monotony(nan_arr)
    with pytest.raises(ValueError):
        monotonic(nan_arr)
    with pytest.raises(ValueError):
        monotonic(nan_arr, mod="increasing")


@pytest.mark.parametrize("safe", ["safe", "suspicious"], ids=["safe", "suspicious"])
def test_all_nan_does_not_crash_when_validation_skipped(safe):
    """跳过有限性检查时全 NaN 不崩: NaN 比较全为 False -> Equal。"""
    nan_arr = np.full(4, np.nan)
    assert give_monotony(nan_arr, safe=safe) is Monotony.Equal
    assert monotonic(nan_arr, safe=safe) is True
    assert monotonic(nan_arr, safe=safe, mod="increasing") is False
    assert monotonic(nan_arr, safe=safe, mod="decreasing") is False


def test_nan_within_values_does_not_crash():
    """部分 NaN(safe="safe"): 分类为 Equal, 方向判定为 False, 不抛异常。"""
    arr = np.array([1.0, np.nan, 3.0])
    assert give_monotony(arr, safe="safe") is Monotony.Equal
    assert monotonic(arr, safe="safe", mod="increasing") is False


def test_two_dimensional_input_rejected():
    """二维输入在两处都抛 ValueError(ndim 校验)。"""
    with pytest.raises(ValueError):
        give_monotony(np.zeros((2, 3)))
    with pytest.raises(ValueError):
        monotonic(np.zeros((2, 3)))


@pytest.mark.parametrize("bad_safe", ["bogus", 0, 3, None], ids=["bogus", "int0", "int3", "none"])
def test_invalid_safe_raises_value_error(bad_safe):
    """``safe`` 只接受 3 个字符串档位。"""
    with pytest.raises(ValueError):
        monotonic(STRICT_INC, safe=bad_safe)


@pytest.mark.parametrize("bad_chunk", [0, -1], ids=["zero", "negative"])
def test_invalid_chunk_size_raises_value_error(bad_chunk):
    with pytest.raises(ValueError):
        give_monotony(STRICT_INC, chunk_size=bad_chunk)
    with pytest.raises(ValueError):
        monotonic(STRICT_INC, chunk_size=bad_chunk)


def test_huge_chunk_size_equivalent_to_default():
    """``chunk_size`` 大于数组长度时走非分块路径, 结果不变。"""
    assert give_monotony(STRICT_INC, chunk_size=10**9) is Monotony.StrictIncreasing
    assert monotonic(STRICT_INC, mod="increasing", chunk_size=10**9) is True


# --------------------------------------------------------------------------- #
# 8. 无副作用: 不改写输入 / dtype 保持 / 确定性 / 分块与整体一致
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "dtype",
    [np.int32, np.int64, np.float32, np.float64],
    ids=["int32", "int64", "float32", "float64"],
)
def test_does_not_mutate_input_array(dtype):
    """调用方数组不被改写, 且 dtype 不被就地改变。"""
    arr = np.array([1, 2, 2, 3, 1], dtype=dtype)
    before = arr.copy()
    assert give_monotony(arr) is Monotony.No
    assert monotonic(arr, mod="monotonic") is False
    assert monotonic(arr, mod="increasing") is False
    assert np.array_equal(arr, before)
    assert arr.dtype == np.dtype(dtype)


def test_does_not_mutate_input_list():
    """列表输入不被改写。"""
    values = [1, 2, 2, 3]
    before = list(values)
    assert give_monotony(values) is Monotony.MixedIncreasing
    assert monotonic(values) is True
    assert values == before


def test_results_are_deterministic():
    """同一输入重复调用结果逐次相同。"""
    for arr in (STRICT_INC, STRICT_DEC, INC_WITH_EQUALS, RISE_THEN_FALL, CONSTANT):
        first = (give_monotony(arr), monotonic(arr), monotonic(arr, strict=True))
        for _ in range(3):
            assert (give_monotony(arr), monotonic(arr), monotonic(arr, strict=True)) == first


@pytest.mark.parametrize("chunk_size", [2, 3, 4], ids=["chunk2", "chunk3", "chunk4"])
@pytest.mark.parametrize("case", CASE_IDS, ids=CASE_IDS)
def test_chunked_direction_paths_match_single_shot(case, chunk_size):
    """显式方向的分块路径与整体判定完全一致(两种 strict 都一致)。"""
    arr, _expected, table = CASES[case]
    for mod in ("increasing", "decreasing"):
        non_strict, strict = table[mod]
        assert monotonic(arr, strict=False, mod=mod, chunk_size=chunk_size) is non_strict
        assert monotonic(arr, strict=True, mod=mod, chunk_size=chunk_size) is strict


@pytest.mark.parametrize("chunk_size", [2, 3, 4], ids=["chunk2", "chunk3", "chunk4"])
@pytest.mark.parametrize("case", CASE_IDS, ids=CASE_IDS)
def test_chunked_non_strict_monotonic_matches_single_shot(case, chunk_size):
    """``mod="monotonic", strict=False`` 的分块结果与整体一致。"""
    arr, _expected, table = CASES[case]
    expected = table["monotonic"][0]
    assert monotonic(arr, strict=False, mod="monotonic", chunk_size=chunk_size) is expected


@pytest.mark.parametrize("chunk_size", [2, 3, 4], ids=["chunk2", "chunk3", "chunk4"])
@pytest.mark.parametrize("case", CASE_IDS, ids=CASE_IDS)
def test_chunked_give_monotony_matches_whole(case, chunk_size):
    """分块 ``give_monotony`` 必须与整体分类**完全相同**（含严格位）。

    历史缺陷（本轮已修）：末块只剩 1 个元素时被分类为 ``Equal``（严格位 0），
    合并后严格位被抹掉，``Strict*`` 退化成同方向的 ``Mixed*``。
    修法：单元素块不含相邻对，其 ``Equal`` 属于空信息，不能参与合并。
    """
    arr, expected, _table = CASES[case]
    got = give_monotony(arr, chunk_size=chunk_size)
    assert isinstance(got, Monotony)
    assert got is expected, (case, chunk_size, got, expected)


@pytest.mark.parametrize("chunk_size", [2, 4], ids=["chunk2", "chunk4"])
def test_chunked_strict_monotonic_mod_is_exact(chunk_size):
    """历史缺陷（本轮已修）：末块为单元素时 ``monotonic(strict=True, mod="monotonic")``
    会对真正严格递增的数组返回 ``False``；现在分块与整体一致。"""
    arr = STRICT_INC                      # 长度 5, 5 % 2 == 5 % 4 == 1
    assert give_monotony(arr) is Monotony.StrictIncreasing
    assert give_monotony(arr, chunk_size=chunk_size) is Monotony.StrictIncreasing
    assert monotonic(arr, strict=True, mod="monotonic") is True
    assert monotonic(arr, strict=True, mod="monotonic", chunk_size=chunk_size) is True
    assert monotonic(arr, strict=True, mod="increasing", chunk_size=chunk_size) is True


@pytest.mark.parametrize("n", [8, 12], ids=["n8", "n12"])
@pytest.mark.parametrize("chunk_size", [2, 3, 4], ids=["chunk2", "chunk3", "chunk4"])
def test_chunked_path_exact_when_chunks_are_not_singleton_tails(n, chunk_size):
    """末块元素数 >= 2 时(n % chunk_size != 1), 分块分类与整体完全一致。"""
    arr = np.arange(n)
    assert give_monotony(arr, chunk_size=chunk_size) is Monotony.StrictIncreasing
    assert monotonic(arr, strict=True, mod="monotonic", chunk_size=chunk_size) is True


def test_chunked_path_detects_direction_break_across_chunk_boundary():
    """方向反转恰好落在分块边界上时也能被检出。"""
    arr = np.array([1, 2, 3, 2, 1])
    assert give_monotony(arr, chunk_size=2) is Monotony.No
    assert monotonic(arr, chunk_size=2, mod="monotonic") is False
    assert monotonic(arr, chunk_size=2, mod="increasing") is False
