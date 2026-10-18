"""
``Modal_Decomposition.Utils.Peaks`` 模块测试。

覆盖 Utils 层统一契约（导出一致性 / 字符串与 Enum 双入口 / 非法值抛
``ValueError`` / 返回契约（元组长度、``int64`` 索引、``vals == S[idx]``）/ 与 scipy
参考实现逐位一致 / 边界输入 / 确定性且不改写调用方数组），以及本模块专有项：

* ``mod="both"`` 返回 **3 元组** ``(idx, vals, is_max)``，``is_max.sum()`` 等于
  ``mod="max"`` 的个数，且 ``idx`` 是极大 + 极小按样本序的严格升序归并；
* ``PEAKS_SUPPORTED_FILTERS``（``height`` / ``threshold`` / ``distance``）三个键
  在 numpy 后端**按峰型方向**解释（``mod="min"`` 等价于在 ``-S`` 上过滤），
  且与 scipy 后端同语义；
* ``numpy`` / ``scipy`` / ``numba`` 三后端在同一输入上逐位一致。

关于 NaN / Inf（实测行为，与模块 docstring 中"NaN 不会成为峰"的描述互补）：
默认 ``safe="unknown"`` 时 ``Check.to_array`` 做有限性检查，含 NaN / Inf 的浮点
输入**抛 ValueError("NaN or Inf found")**，不是静默跳过；``safe="safe"`` 跳过该
检查，此时 NaN 因比较恒为 False 而不会被当作峰（也不崩）。
"""

import inspect
import typing

import numpy as np
import pytest

import Modal_Decomposition.Utils as utils
import Modal_Decomposition.Utils.Peaks as peaks_module
from Modal_Decomposition.Utils import (
    PEAKS_BACKEND,
    PEAKS_BACKEND_ARGS,
    PEAKS_MODS,
    PEAKS_MODS_ARGS,
    PEAKS_SUPPORTED_FILTERS,
    PEAKS_SUPPORTED_FILTERS_ARGS,
    find_peaks,
)

BACKENDS = ("numpy", "scipy")          # numba 另有用例（可选依赖）
MODS = ("max", "min", "both")


# --------------------------------------------------------------------------- #
# 构造器
# --------------------------------------------------------------------------- #
def wave() -> np.ndarray:
    """11 点波形：5 个极大 (1/2/3/1/2) + 4 个极小 (0)，无平台。"""
    return np.array([0.0, 1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 1.0, 0.0, 2.0, 0.0],
                    dtype=np.float64)


def separated() -> np.ndarray:
    """21 点波形：极大位于 2/10/18（高 1/3/2），两两相距 8 > 5。

    背景是 0→0.1 的缓坡而不是常数，避免零点平台（numpy / numba 用基础规则、
    scipy 用平台中点规则，平台上的结果**本就允许不同**，见模块 docstring）。
    """
    S = np.linspace(0.0, 0.1, 21)
    S[[2, 10, 18]] = [1.0, 3.0, 2.0]
    return S


def noisy(n: int = 256, seed: int = 0) -> np.ndarray:
    """固定种子的随机信号（无平台，避免 numpy / scipy 的平台规则差异）。"""
    return np.random.default_rng(seed).standard_normal(n)


# --------------------------------------------------------------------------- #
# 1. 公开面 / 导出一致性
# --------------------------------------------------------------------------- #
def test_all_is_complete_and_names_are_real():
    """``__all__`` 覆盖 7 个公开名，且每个名字都能从模块与 Utils 顶层取到。"""
    expected = {
        "find_peaks",
        "PEAKS_MODS", "PEAKS_MODS_ARGS",
        "PEAKS_BACKEND", "PEAKS_BACKEND_ARGS",
        "PEAKS_SUPPORTED_FILTERS", "PEAKS_SUPPORTED_FILTERS_ARGS",
    }
    assert isinstance(peaks_module.__all__, list)
    assert set(peaks_module.__all__) == expected
    assert len(peaks_module.__all__) == len(set(peaks_module.__all__))
    for name in peaks_module.__all__:
        assert hasattr(peaks_module, name), f"{name} 不在模块命名空间里"

    # Utils 顶层 re-export 的是同一对象，不是副本
    assert utils.find_peaks is peaks_module.find_peaks
    assert utils.PEAKS_MODS is peaks_module.PEAKS_MODS
    assert utils.PEAKS_SUPPORTED_FILTERS is peaks_module.PEAKS_SUPPORTED_FILTERS


def test_signature_defaults():
    """统一签名：``S`` 必填；``mod`` / ``backend`` / ``safe`` 有默认值；过滤走 ``**kwargs``。"""
    sig = inspect.signature(find_peaks)
    params = sig.parameters
    first = next(iter(params))
    assert first == "S"
    assert params["S"].default is inspect.Parameter.empty
    assert params["mod"].default == "max"
    assert params["backend"].default == "numpy"
    assert params["safe"].default == "unknown"
    assert params["kwargs"].kind is inspect.Parameter.VAR_KEYWORD
    # mod / backend 可以位置传参
    assert np.array_equal(find_peaks(wave(), "max", "scipy")[0], [1, 3, 5, 7, 9])


@pytest.mark.parametrize("enum, alias, expected", [
    (PEAKS_MODS, PEAKS_MODS_ARGS, ("max", "min", "both")),
    (PEAKS_BACKEND, PEAKS_BACKEND_ARGS, ("scipy", "numpy", "numba")),
    (PEAKS_SUPPORTED_FILTERS, PEAKS_SUPPORTED_FILTERS_ARGS,
     ("height", "threshold", "distance")),
], ids=["mods", "backend", "filters"])
def test_enum_and_literal_alias_are_paired(enum, alias, expected):
    """每组 Enum 的成员值与对应 Literal 别名的参数表一致，且 ``enum(x)`` 幂等。"""
    assert typing.get_args(alias) == expected
    assert tuple(member.value for member in enum) == expected
    for member in enum:
        assert enum(member.value) is member
        assert enum(member) is member


# --------------------------------------------------------------------------- #
# 2. 返回契约（2 元组 / 3 元组 + is_max 掩码）
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("backend", BACKENDS)
def test_return_contract(backend):
    """``max`` / ``min`` 返回 2 元组，``both`` 返回 3 元组；
    索引恒为 ``int64``，取值恒是原信号幅值，``both`` 已按样本序严格升序归并。"""
    S = wave()
    out_max = find_peaks(S, mod="max", backend=backend)
    out_min = find_peaks(S, mod="min", backend=backend)
    out_both = find_peaks(S, mod="both", backend=backend)
    assert len(out_max) == 2 and len(out_min) == 2 and len(out_both) == 3

    idx_max, vals_max = out_max
    idx_min, vals_min = out_min
    idx_both, vals_both, is_max = out_both

    for idx, vals in ((idx_max, vals_max), (idx_min, vals_min),
                      (idx_both, vals_both)):
        assert isinstance(idx, np.ndarray) and idx.ndim == 1
        assert idx.dtype == np.int64
        assert vals.shape == idx.shape
        assert np.array_equal(vals, S[idx])
        assert not np.shares_memory(vals, S)        # 取值是拷贝, 不是视图

    # mod="both" 的语义: 极大 / 极小两组按样本序归并
    assert idx_both.size == idx_max.size + idx_min.size
    assert idx_max.size and idx_min.size
    assert np.all(np.diff(idx_both) > 0)
    assert is_max.dtype == np.bool_
    assert is_max.shape == idx_both.shape
    assert is_max.sum() == idx_max.size          # 关键契约
    assert np.array_equal(idx_both[is_max], idx_max)
    assert np.array_equal(idx_both[~is_max], idx_min)


# --------------------------------------------------------------------------- #
# 3. 字符串 / Enum 双入口
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("mod", MODS)
def test_mod_string_and_enum_agree(mod):
    """``mod="both"`` 与 ``mod=PEAKS_MODS.Both`` 等写法逐位相同。"""
    S = wave()
    by_str = find_peaks(S, mod=mod)
    by_enum = find_peaks(S, mod=PEAKS_MODS(mod))
    assert len(by_str) == len(by_enum)
    for a, b in zip(by_str, by_enum):
        assert np.array_equal(a, b)


@pytest.mark.parametrize("backend", BACKENDS)
def test_backend_string_and_enum_agree(backend):
    """``backend="scipy"`` 与 ``backend=PEAKS_BACKEND.Scipy`` 结果逐位相同。"""
    S = wave()
    by_str = find_peaks(S, mod="both", backend=backend)
    by_enum = find_peaks(S, mod="both", backend=PEAKS_BACKEND(backend))
    for a, b in zip(by_str, by_enum):
        assert np.array_equal(a, b)


# --------------------------------------------------------------------------- #
# 4. numpy / scipy 后端逐位一致
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("mod", MODS)
def test_numpy_and_scipy_backends_are_bitwise_equal(mod):
    """同一输入上 numpy 与 scipy 后端的索引 / 取值 / 掩码逐位相同。"""
    for S in (wave(), separated(), noisy()):
        ref = find_peaks(S, mod=mod, backend="numpy")
        got = find_peaks(S, mod=mod, backend="scipy")
        assert len(ref) == len(got) == (3 if mod == "both" else 2)
        for a, b in zip(ref, got):
            assert np.array_equal(a, b)


# --------------------------------------------------------------------------- #
# 5. 非法 mod / backend
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("bad", ["nope", "MAX", "", "maxx", None, 1, 2.0])
def test_invalid_mod_raises_value_error(bad):
    with pytest.raises(ValueError):
        find_peaks(wave(), mod=bad)


@pytest.mark.parametrize("bad", ["nope", "Numpy", "", "gpu", None, 0, 1.0])
def test_invalid_backend_raises_value_error(bad):
    with pytest.raises(ValueError):
        find_peaks(wave(), backend=bad)


def test_invalid_mod_checked_before_backend():
    """两个参数都非法时仍抛 ``ValueError``（先校验 ``mod``）。"""
    with pytest.raises(ValueError):
        find_peaks(wave(), mod="nope", backend="nope")


# --------------------------------------------------------------------------- #
# 6. PEAKS_SUPPORTED_FILTERS: height / threshold / distance
# --------------------------------------------------------------------------- #
def test_supported_filter_enum_values_are_real_keywords():
    """枚举成员名就是 ``find_peaks`` 实际接受的过滤键名，过滤只会删峰不会新增。"""
    S = wave()
    base = find_peaks(S, mod="max")[0]
    kwargs_for = {
        PEAKS_SUPPORTED_FILTERS.Height: {"height": 1.5},
        PEAKS_SUPPORTED_FILTERS.Threshold: {"threshold": 1.0},
        PEAKS_SUPPORTED_FILTERS.Distance: {"distance": 3},
    }
    assert set(kwargs_for) == set(PEAKS_SUPPORTED_FILTERS)
    for member, kw in kwargs_for.items():
        assert list(kw) == [member.value]
        got = find_peaks(S, mod="max", **kw)[0]
        assert got.size <= base.size
        assert np.isin(got, base).all()


def test_distance_enforces_min_spacing():
    """``distance=5`` 时保留峰的间距 >= 5，三个相距 8 的峰全部保留。"""
    S = separated()
    assert np.array_equal(find_peaks(S, mod="max")[0], [2, 10, 18])
    for backend in BACKENDS:
        idx, vals = find_peaks(S, mod="max", backend=backend, distance=5)
        assert np.array_equal(idx, [2, 10, 18])
        assert np.all(np.diff(idx) >= 5)
        assert np.array_equal(vals, S[idx])


def test_distance_removes_crowded_peaks():
    """``distance`` 贪心保留更高的峰：wave 上只有最高峰 (idx=5, 值 3) 存活。"""
    S = wave()
    idx = find_peaks(S, mod="max", distance=5)[0]
    assert np.array_equal(idx, [5])
    assert np.all(np.diff(idx) >= 5) if idx.size > 1 else True

    # "both": 距离按峰型分别施加（极大只和极大比, 极小只和极小比）
    idx_both, _, is_max = find_peaks(S, mod="both", distance=5)
    for sel in (idx_both[is_max], idx_both[~is_max]):
        if sel.size > 1:
            assert np.all(np.diff(sel) >= 5)


@pytest.mark.parametrize("backend", BACKENDS)
def test_distance_rejects_non_positive(backend):
    """``distance`` 必须 >= 1（numpy 自校验；scipy 由 scipy.signal 抛 ValueError）。"""
    for bad in (0, -2):
        with pytest.raises(ValueError, match="distance"):
            find_peaks(wave(), backend=backend, distance=bad)


def test_height_filters_low_peaks():
    """``height=2.0`` 去掉幅值 1 的低峰 (idx=1/7)，保留 idx=3/5/9。"""
    S = wave()
    base = find_peaks(S, mod="max")[0]
    idx = find_peaks(S, mod="max", height=2.0)[0]
    assert np.array_equal(base, [1, 3, 5, 7, 9])
    assert np.array_equal(idx, [3, 5, 9])
    assert np.all(S[idx] >= 2.0)
    assert idx.size < base.size


def test_height_interval():
    """``height=(min, max)`` 是区间过滤：排除 3.0（超上界）与 1.0（低于下界）。"""
    S = wave()
    assert np.array_equal(find_peaks(S, mod="max", height=(1.5, 2.5))[0], [3, 9])


def test_height_on_min_is_negated():
    """``mod="min"`` 的 height 语义等价于在 ``-S`` 上过滤。"""
    S = wave()
    # 极小值都是 0 -> -S == 0: >= -0.5 全保留, >= 0.5 全丢弃
    assert np.array_equal(find_peaks(S, mod="min", height=-0.5)[0], [2, 4, 6, 8])
    assert find_peaks(S, mod="min", height=0.5)[0].size == 0


@pytest.mark.parametrize("backend", BACKENDS)
def test_height_matches_scipy_semantics(backend):
    S = wave()
    assert np.array_equal(
        find_peaks(S, mod="max", backend=backend, height=1.5)[0], [3, 5, 9])
    assert np.array_equal(
        find_peaks(S, mod="max", backend=backend, height=(1.5, 2.5))[0], [3, 9])
    assert np.array_equal(
        find_peaks(S, mod="min", backend=backend, height=-0.5)[0], [2, 4, 6, 8])


def test_threshold_filters_shallow_peaks():
    """``threshold`` 要求峰相对两侧邻点的落差都大于阈值。"""
    S = wave()
    assert np.array_equal(find_peaks(S, mod="max", threshold=1.0)[0], [1, 3, 5, 7, 9])
    assert np.array_equal(find_peaks(S, mod="max", threshold=1.5)[0], [3, 5, 9])
    for backend in BACKENDS:
        assert np.array_equal(
            find_peaks(S, mod="max", backend=backend, threshold=1.5)[0], [3, 5, 9])


def test_both_applies_filters_per_peak_type():
    """``mod="both"`` 时过滤器按类型分别施加，再归并。"""
    S = wave()
    idx, vals, is_max = find_peaks(S, mod="both", height=1.5)
    assert np.array_equal(idx, [3, 5, 9])            # 极小 (值 0) 全被过滤
    assert is_max.all()
    assert np.array_equal(vals, [2.0, 3.0, 2.0])

    idx2, _, is_max2 = find_peaks(S, mod="both", height=-0.5)
    assert idx2.size == 9                            # 两类都保留
    assert is_max2.sum() == 5

    idx3, _, is_max3 = find_peaks(S, mod="both", height=(1.5, 2.5))
    assert np.array_equal(idx3, [3, 9]) and is_max3.all()


def test_unknown_kwargs_ignored_by_numpy_backend():
    """numpy 后端只识别三个过滤键，其余键忽略（兼容 Envelope 等调用方的透传）。"""
    S = wave()
    base = find_peaks(S, mod="max")[0]
    assert np.array_equal(find_peaks(S, mod="max", prominence=1.0, foo=3)[0], base)


def test_unknown_kwargs_forwarded_to_scipy_backend():
    """scipy 后端全参数透传，未识别的键由 scipy 抛 ``TypeError``。"""
    with pytest.raises(TypeError):
        find_peaks(wave(), mod="max", backend="scipy", foo=3)


# --------------------------------------------------------------------------- #
# 7. 边界输入
# --------------------------------------------------------------------------- #
def test_constant_and_zero_signals_have_no_peaks():
    """常量 / 全零信号上没有峰（严格不等式判定），不会退化成"处处是峰"。"""
    for S in (np.ones(7), np.zeros(7)):
        for mod in MODS:
            idx = find_peaks(S, mod=mod)[0]
            assert idx.size == 0
        assert find_peaks(S, mod="max", backend="scipy")[0].size == 0
        assert find_peaks(S, mod="min", backend="scipy")[0].size == 0


@pytest.mark.parametrize("n", [1, 2])
def test_short_signal(n):
    """长度 1/2：单峰型沿用"每个下标都是峰"的旧行为；both 返回空集 + 空掩码。"""
    S = np.arange(n, dtype=np.float64)
    for mod in ("max", "min"):
        idx, vals = find_peaks(S, mod=mod)
        assert np.array_equal(idx, np.arange(n))
        assert np.array_equal(vals, S)

    idx_b, vals_b, is_max = find_peaks(S, mod="both")
    assert idx_b.size == 0 and vals_b.size == 0 and is_max.size == 0
    assert idx_b.dtype == np.int64 and is_max.dtype == np.bool_


def test_empty_signal_returns_empty():
    S = np.array([], dtype=np.float64)
    for mod in MODS:
        out = find_peaks(S, mod=mod)
        assert out[0].size == 0 and out[1].size == 0
    assert find_peaks(S, mod="both")[2].size == 0


def test_nan_and_inf_raise_clean_value_error():
    """默认 ``safe="unknown"`` 下 NaN / Inf 被前置校验拒绝（可控报错, 不崩）。"""
    for bad in (np.array([0.0, 1.0, np.nan, 2.0, 0.0]),
                np.array([0.0, np.inf, 1.0, 0.0])):
        for mod in MODS:
            with pytest.raises(ValueError, match="NaN or Inf"):
                find_peaks(bad, mod=mod)
        with pytest.raises(ValueError, match="NaN or Inf"):
            find_peaks(bad, mod="max", backend="scipy")


def test_nan_safe_mode_does_not_crash_and_never_returns_nan():
    """``safe="safe"`` 跳过有限性检查：NaN 因比较恒 False 不会被当成峰。"""
    S = np.array([0.0, 1.0, np.nan, 2.0, 0.0, 3.0, 0.0])
    nan_at = 2
    for mod in MODS:
        idx, vals = find_peaks(S, mod=mod, safe="safe")[:2]
        assert nan_at not in idx
        assert np.all(np.isfinite(vals))
        assert np.array_equal(vals, S[idx])


def test_safe_modes_agree_on_finite_input():
    S = wave()
    for mod in MODS:
        relaxed = find_peaks(S, mod=mod, safe="safe")
        strict = find_peaks(S, mod=mod, safe="unknown")
        for a, b in zip(relaxed, strict):
            assert np.array_equal(a, b)


@pytest.mark.parametrize("bad", [
    np.zeros((2, 8)),
    np.array([0 + 0j, 1 + 0j, 0 + 0j]),
    np.array(["a", "b", "c"]),
])
def test_rejects_invalid_inputs(bad):
    """2-D / 复数 / 非数值 dtype 都被拒绝（ValueError 或 TypeError）。"""
    with pytest.raises((ValueError, TypeError)):
        find_peaks(bad)


def test_accepts_array_like_and_preserves_dtype():
    """array-like 输入可用；整数信号的原 dtype 在 ``values`` 上保留。"""
    assert np.array_equal(find_peaks([0.0, 1.0, 0.0, 2.0, 0.0])[0], [1, 3])
    Si = np.array([0, 1, 0, 2, 0], dtype=np.int32)
    idx, vals = find_peaks(Si)
    assert np.array_equal(idx, [1, 3])
    assert vals.dtype == np.int32
    assert np.array_equal(vals, Si[idx])


# --------------------------------------------------------------------------- #
# 8. 确定性 / 无副作用
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("mod", MODS)
@pytest.mark.parametrize("backend", BACKENDS)
def test_deterministic_and_does_not_mutate_input(mod, backend):
    """同输入两次逐位相同；调用方数组不被改写，返回的取值也不与之共享内存。"""
    S = wave()
    before = S.copy()
    kwargs = dict(mod=mod, backend=backend, distance=3)
    first = find_peaks(S, **kwargs)
    second = find_peaks(S, **kwargs)
    assert len(first) == len(second)
    for a, b in zip(first, second):
        assert np.array_equal(a, b)
    assert np.array_equal(S, before)
    assert not np.shares_memory(first[0], S)
    assert not np.shares_memory(first[1], S)
    # 距离过滤确实生效（wave 上极大两两相距 2 < 3）
    if mod == "max":
        assert np.all(np.diff(first[0]) >= 3)


def test_read_only_input_is_supported():
    """只读数组可安全传入（实现不写入调用方数据）。"""
    S = wave()
    S.flags.writeable = False
    idx, vals, is_max = find_peaks(S, mod="both")
    assert idx.size == 9 and is_max.sum() == 5
    assert np.array_equal(vals, S[idx])


# --------------------------------------------------------------------------- #
# 9. 可选后端: numba（缺依赖时跳过）
# --------------------------------------------------------------------------- #
def test_numba_backend_matches_numpy():
    """numba 后端与 numpy 共用同一份过滤逻辑，结果逐位一致。"""
    pytest.importorskip("numba")
    for mod in MODS:
        for S in (wave(), separated(), noisy(120)):
            ref = find_peaks(S, mod=mod, backend="numpy")
            got = find_peaks(S, mod=mod, backend="numba")
            assert len(ref) == len(got) == (3 if mod == "both" else 2)
            for a, b in zip(ref, got):
                assert np.array_equal(a, b)
    assert np.array_equal(
        find_peaks(separated(), mod="max", backend="numba", distance=5)[0],
        [2, 10, 18],
    )
    assert find_peaks(wave(), mod="min", backend="numba", height=0.5)[0].size == 0
