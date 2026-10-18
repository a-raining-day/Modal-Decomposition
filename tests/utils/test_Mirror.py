"""
``Modal_Decomposition.Utils.Mirror`` 的统一契约测试。

覆盖: 导出契约 / 五种 ``mod`` 的返回类型与长度规律 / 字符串与 Enum 等价 /
非法 ``mod`` / 边界输入 (短信号 / 常量 / NaN / 空 / 2-D) / 确定性 /
不改写调用方输入 / ``T`` 入参。

实测结论 (与断言一一对应)
--------------------------
- ``mod="mirror"``      -> 1-D ``ndarray``, 长度 ``N + 2*nbsym``;
- ``mod="extrema"``     -> 3 元组 ``(T, A, is_max)``, 长度 ``L + 2*nbsym``;
- ``mod="polynomial"``  -> 3 元组, 长度 ``L + 2*nbsym`` (要求 ``order <= nbsym``);
- ``mod="parallel"``    -> 3 元组, 长度 ``L + 2*nbsym``;
- ``mod="BLCSE"``       -> 3 元组, 长度 ``L + 4`` (每端固定补一对, 与 ``nbsym`` 无关),
  其中 ``L = find_peaks(S, mod="both")[0].size``。
"""

import typing

import numpy as np
import pytest

from Modal_Decomposition.Utils import MIRROR_MODS, MIRROR_MODS_ARGS, find_peaks, mirror

ALL_MODS = [
    MIRROR_MODS.Mirror,
    MIRROR_MODS.Extrema,
    MIRROR_MODS.Polynomial,
    MIRROR_MODS.Parallel,
    MIRROR_MODS.BLCSE,
]
MOD_IDS = [m.name for m in ALL_MODS]

# 极值域模式 (返回极值表三元组) —— 与只返回 1-D 信号的 "mirror" 区分
EXTREMA_MODS = [
    MIRROR_MODS.Extrema,
    MIRROR_MODS.Polynomial,
    MIRROR_MODS.Parallel,
    MIRROR_MODS.BLCSE,
]
EXTREMA_IDS = [m.name for m in EXTREMA_MODS]


def _sig(n: int = 256, seed: int = 0) -> np.ndarray:
    """双音 + 轻噪声, 极值足够多 (nbsym=1..3 都能外延)。"""
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 1.0, n, endpoint=False)
    return (
        np.sin(2 * np.pi * 5 * t)
        + 0.5 * np.sin(2 * np.pi * 20 * t)
        + 0.05 * rng.standard_normal(n)
    )


def _n_extrema(S: np.ndarray) -> int:
    """极值总数 L (极大 + 极小), 与实现内部用的是同一个入口。"""
    return int(find_peaks(S, mod="both")[0].size)


def _assert_triple(out, n_expected: int | None = None) -> None:
    """极值域模式的统一返回值契约: 3 元组 ``(T, A, is_max)``, 等长 1-D、时间升序。"""
    assert isinstance(out, tuple), f"expected tuple, got {type(out).__name__}"
    assert len(out) == 3, f"expected 3 elements, got {len(out)}"
    extrema_T, extrema_arr, is_max = out

    for arr, name in ((extrema_T, "T"), (extrema_arr, "A")):
        assert isinstance(arr, np.ndarray), f"{name} must be ndarray"
        assert arr.ndim == 1, f"{name} must be 1-D, got ndim={arr.ndim}"
        assert np.issubdtype(arr.dtype, np.floating), f"{name} dtype={arr.dtype}"
        assert np.all(np.isfinite(arr)), f"{name} contains NaN/Inf"

    assert isinstance(is_max, np.ndarray)
    assert is_max.ndim == 1
    assert is_max.dtype == np.bool_, f"is_max dtype={is_max.dtype}"

    assert extrema_T.shape == extrema_arr.shape == is_max.shape
    assert extrema_T.size >= 1
    # 极值表按时间严格升序, 否则调用方无法直接拿去插值
    assert np.all(np.diff(extrema_T) > 0), "extrema time axis must be strictly increasing"

    if n_expected is not None:
        assert extrema_T.size == n_expected


# --------------------------------------------------------------------------- #
# 1. 导出契约
# --------------------------------------------------------------------------- #
def test_dunder_all_and_public_exports():
    from Modal_Decomposition.Utils import Mirror as mirror_module

    assert set(mirror_module.__all__) == {"mirror", "MIRROR_MODS", "MIRROR_MODS_ARGS"}
    for name in mirror_module.__all__:
        assert hasattr(mirror_module, name), f"missing export: {name}"
    # 包级再导出指向同一个对象
    assert mirror.__module__ == "Modal_Decomposition.Utils.Mirror"
    assert callable(mirror)


def test_enum_members_and_literal_alias_agree():
    assert [m.name for m in MIRROR_MODS] == [
        "Mirror", "Extrema", "Polynomial", "Parallel", "BLCSE",
    ]
    # Literal 别名与 Enum 取值同源 (顺序也一致), 供 Check.validate_mod 使用
    assert typing.get_args(MIRROR_MODS_ARGS) == tuple(m.value for m in MIRROR_MODS)


# --------------------------------------------------------------------------- #
# 2. mod="mirror": 返回 ndarray, 长度 >= len(S), 且随 nbsym 不减
# --------------------------------------------------------------------------- #
def test_mirror_mode_returns_1d_array():
    S = _sig()
    out = mirror(S, mod="mirror", nbsym=2)
    assert isinstance(out, np.ndarray)
    assert out.ndim == 1
    assert out.dtype == S.dtype
    assert out.size >= S.size
    assert out.size == S.size + 2 * 2


@pytest.mark.parametrize("nbsym", [0, 1, 2, 3, 4])
def test_mirror_mode_length_is_n_plus_2nbsym(nbsym):
    S = _sig()
    assert mirror(S, mod="mirror", nbsym=nbsym).size == S.size + 2 * nbsym


def test_mirror_mode_length_non_decreasing_in_nbsym():
    S = _sig()
    lengths = [mirror(S, mod="mirror", nbsym=n).size for n in (1, 2, 3)]
    assert lengths[0] <= lengths[1] <= lengths[2]
    assert lengths == [S.size + 2, S.size + 4, S.size + 6]


def test_mirror_mode_reflection_is_exact():
    """反射轴是端点内侧第一个样本 (与 np.pad(mode="reflect") 同义)。"""
    out = mirror(np.arange(6.0), mod="mirror", nbsym=2)
    assert out.tolist() == [2.0, 1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0]


def test_mirror_mode_keeps_middle_segment():
    S = _sig()
    out = mirror(S, mod="mirror", nbsym=3)
    assert np.array_equal(out[3:3 + S.size], S)


def test_mirror_mode_nbsym_zero_returns_independent_copy():
    S = _sig()
    out = mirror(S, mod="mirror", nbsym=0)
    assert np.array_equal(out, S)
    assert out is not S
    assert not np.shares_memory(out, S)


def test_mirror_mode_short_and_single_point_signals():
    short = np.arange(3.0)
    assert mirror(short, mod="mirror", nbsym=5).size == short.size + 2 * 5

    one = mirror(np.array([2.5]), mod="mirror", nbsym=4)
    assert one.size == 1 + 2 * 4
    assert np.all(one == 2.5)          # 单点走 edge padding


def test_mirror_mode_preserves_float32_dtype():
    out = mirror(_sig().astype(np.float32), mod="mirror", nbsym=2)
    assert out.dtype == np.float32


# --------------------------------------------------------------------------- #
# 3. mod="extrema": 返回 3 元组, 形状自洽
# --------------------------------------------------------------------------- #
def test_extrema_mode_returns_triple():
    S = _sig()
    L = _n_extrema(S)
    out = mirror(S, mod="extrema", nbsym=2)
    _assert_triple(out, L + 2 * 2)


def test_extrema_mode_middle_segment_equals_find_peaks():
    S = _sig()
    idx, vals, mask = find_peaks(S, mod="both")
    extrema_T, extrema_arr, is_max = mirror(S, mod="extrema", nbsym=2)
    mid = slice(2, 2 + idx.size)
    assert np.array_equal(extrema_T[mid], idx.astype(np.float64))
    assert np.array_equal(extrema_arr[mid], vals)
    assert np.array_equal(is_max[mid], mask)


@pytest.mark.parametrize("nbsym", [1, 2, 3])
def test_extrema_mode_length_grows_with_nbsym(nbsym):
    S = _sig()
    L = _n_extrema(S)
    _assert_triple(mirror(S, mod="extrema", nbsym=nbsym), L + 2 * nbsym)


# --------------------------------------------------------------------------- #
# 4. 其余三个极值域成员逐个跑
# --------------------------------------------------------------------------- #
def test_polynomial_mode_returns_triple():
    S = _sig()
    L = _n_extrema(S)
    out = mirror(S, mod="polynomial", nbsym=2, poly_order=2, time_poly_order=2)
    _assert_triple(out, L + 2 * 2)


def test_polynomial_mode_requires_orders_within_nbsym():
    S = _sig()
    with pytest.raises(ValueError):
        mirror(S, mod="polynomial", nbsym=2, poly_order=3)
    with pytest.raises(ValueError):
        mirror(S, mod="polynomial", nbsym=2, time_poly_order=3)
    # 默认 poly_order=2 > nbsym=1, 直接拒绝
    with pytest.raises(ValueError):
        mirror(S, mod="polynomial", nbsym=1)


def test_parallel_mode_returns_triple_and_extends_outside():
    S = _sig()
    L = _n_extrema(S)
    idx = find_peaks(S, mod="both")[0]
    out = mirror(S, mod="parallel", nbsym=2)
    _assert_triple(out, L + 2 * 2)
    extrema_T = out[0]
    assert extrema_T[0] < idx[0]
    assert extrema_T[-1] > idx[-1]


@pytest.mark.parametrize("nbsym", [1, 3])
def test_blcse_mode_returns_triple_with_fixed_extra_pair(nbsym):
    """BLCSE 每端固定补一对 (极小, 极大): 长度恒为 L + 4, 与 nbsym 无关。"""
    S = _sig()
    L = _n_extrema(S)
    _assert_triple(mirror(S, mod="BLCSE", nbsym=nbsym), L + 4)


def test_blcse_mode_requires_both_maxima_and_minima():
    with pytest.raises(ValueError):
        mirror(np.ones(64), mod="BLCSE", nbsym=2)


# --------------------------------------------------------------------------- #
# 5. 字符串与 Enum 成员结果逐位相同 (五个成员全覆盖)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("mod", ALL_MODS, ids=MOD_IDS)
def test_string_and_enum_are_bitwise_identical(mod):
    S = _sig()
    by_str = mirror(S, mod=mod.value, nbsym=2)
    by_enum = mirror(S, mod=mod, nbsym=2)
    if isinstance(by_str, tuple):
        assert isinstance(by_enum, tuple)
        assert len(by_str) == len(by_enum) == 3
        for a, b in zip(by_str, by_enum):
            assert a.dtype == b.dtype
            assert np.array_equal(a, b)
    else:
        assert np.array_equal(by_str, by_enum)


# --------------------------------------------------------------------------- #
# 6. 非法 mod 抛 ValueError
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("bad", ["nope", "", "Mirror", "MIRROR", 0, 1, 3.14, None])
def test_invalid_mod_raises_value_error(bad):
    with pytest.raises(ValueError):
        mirror(_sig(64), mod=bad, nbsym=2)


# --------------------------------------------------------------------------- #
# 7. 边界输入
# --------------------------------------------------------------------------- #
def test_len8_short_signal_every_mod():
    S8 = np.sin(np.linspace(0.0, 6.0, 8))       # 一峰一谷
    assert mirror(S8, mod="mirror", nbsym=1).size == S8.size + 2

    for mod in EXTREMA_MODS:
        kwargs = {"poly_order": 1, "time_poly_order": 1} if mod is MIRROR_MODS.Polynomial else {}
        out = mirror(S8, mod=mod, nbsym=1, **kwargs)
        _assert_triple(out)
        assert out[0].size >= _n_extrema(S8)


def test_constant_signal_mirror_ok_extrema_rejected():
    C = np.ones(64)
    out = mirror(C, mod="mirror", nbsym=2)
    assert out.size == C.size + 4
    assert np.all(out == 1.0)

    for mod in EXTREMA_MODS:
        with pytest.raises(ValueError):
            mirror(C, mod=mod, nbsym=2)


@pytest.mark.parametrize("mod", ALL_MODS, ids=MOD_IDS)
def test_nan_signal_rejected_without_mutation(mod):
    S = _sig(64)
    S[10] = np.nan
    with pytest.raises(ValueError):
        mirror(S, mod=mod, nbsym=2)
    assert np.isnan(S[10])            # 输入未被中途改写


def test_empty_and_2d_inputs_are_rejected():
    with pytest.raises(ValueError):
        mirror(np.array([]), nbsym=1)
    with pytest.raises(ValueError):
        mirror(np.zeros((2, 8)), nbsym=1)


def test_invalid_parameters_raise():
    S = _sig()
    with pytest.raises(ValueError):
        mirror(S, nbsym=-1)
    with pytest.raises(TypeError):
        mirror(S, nbsym=1.5)
    with pytest.raises(TypeError):
        mirror(S, nbsym=True)
    with pytest.raises(TypeError):
        mirror(S, verbose=1)
    with pytest.raises(ValueError):
        mirror(S, peaks_backend="nope")
    with pytest.raises(ValueError):
        mirror(S, safe="nope")


# --------------------------------------------------------------------------- #
# 8. 确定性 + 不改写调用方数组
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("mod", ALL_MODS, ids=MOD_IDS)
def test_deterministic_and_input_untouched(mod):
    S = _sig()
    before = S.copy()
    first = mirror(S, mod=mod, nbsym=2)
    second = mirror(S, mod=mod, nbsym=2)

    if isinstance(first, tuple):
        assert len(first) == len(second) == 3
        for a, b in zip(first, second):
            assert np.array_equal(a, b)
            assert not np.shares_memory(a, S)
    else:
        assert np.array_equal(first, second)
        assert not np.shares_memory(first, S)

    assert np.array_equal(S, before)


# --------------------------------------------------------------------------- #
# 9. T 入参
# --------------------------------------------------------------------------- #
def test_mirror_mode_ignores_T():
    S = _sig()
    T = np.linspace(0.0, 1.0, S.size)
    assert np.array_equal(
        mirror(S, T=T, mod="mirror", nbsym=2),
        mirror(S, mod="mirror", nbsym=2),
    )


@pytest.mark.parametrize("mod", EXTREMA_MODS, ids=EXTREMA_IDS)
def test_T_is_used_and_extension_stays_outside(mod):
    S = _sig()
    T = np.linspace(0.0, 1.0, S.size)          # 与样本序号成线性关系
    extrema_T = mirror(S, T=T, mod=mod, nbsym=2)[0]
    assert np.all(np.diff(extrema_T) > 0)
    # 端部外延点必须落在中段 (真实极值) 之外 —— 注意本信号的末极值就在最后一个样本,
    # 因此不能用 T[-1] 作参照 (外延点会恰好等于边界时刻)。
    central = slice(2, extrema_T.size - 2)
    assert extrema_T[0] < extrema_T[central.start]
    assert extrema_T[-1] > extrema_T[central.stop - 1]


def test_T_size_mismatch_raises():
    S = _sig()
    with pytest.raises(ValueError):
        mirror(S, T=np.arange(10.0), mod="extrema", nbsym=2)


def test_non_monotonic_T_is_rearranged_with_warning():
    S = _sig()
    T = np.linspace(0.0, 1.0, S.size)[::-1]
    with pytest.warns(UserWarning, match="rearranged"):
        extrema_T = mirror(S, T=T, mod="extrema", nbsym=2, verbose=True)[0]
    assert np.all(np.diff(extrema_T) > 0)


def test_verbose_controls_diagnostics():
    import warnings

    S = _sig()
    T = np.linspace(0.0, 1.0, S.size) ** 2        # 非均匀 (但递增)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        mirror(S, T=T, mod="extrema", nbsym=2, verbose=False)
        assert caught == []

    with pytest.warns(UserWarning, match="not uniform"):
        mirror(S, T=T, mod="extrema", nbsym=2, verbose=True)
