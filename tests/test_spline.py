"""
tests for ``Modal_Decomposition.Utils.Spline``.

覆盖:
1. 正确性 —— 节点精确性 / 平滑函数逼近 / 导数与积分代理 / 多后端一致性;
2. 极端输入 —— 空、单点、长度不一致、重复节点、乱序、NaN/Inf、常量、
   二维、超大样本、未知后端、两节点退化解;
3. 性能/耗时 —— 各后端拟合与求值的实测耗时 (宽松上限防回归, 明细打印);
4. Cache 集成 —— 首次使用后模块以 ``CACHE_KEY`` 注册进全局 import 缓存。

运行: python -m pytest tests/test_spline.py -v   (仓库根目录, src 布局)
"""

import time
import warnings

import numpy as np
import pytest

from src.Modal_Decomposition.Base.Cache import cache
from src.Modal_Decomposition.Base.ConstDefine import SPLINE_KIND
from src.Modal_Decomposition.Utils.Spline import (
    Spline,
    spline,
)

# --------------------------------------------------------------------------- #
# fixtures / helpers
# --------------------------------------------------------------------------- #
X = np.linspace(0.0, 10.0, 11)
Y = np.sin(X)
XI = np.linspace(0.0, 10.0, 1001)
GT = np.sin(XI)


def _median_seconds(fn, repeats: int = 3) -> float:
    """运行 fn 若干次, 返回中位耗时 (秒)。"""
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    return float(np.median(samples))


# --------------------------------------------------------------------------- #
# 1. 正确性
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("spline_kind", SPLINE_KIND)
def test_all_backends_interpolate_nodes_exactly(spline_kind):
    """每个后端都必须精确穿过全部节点 (节点精确性)。"""
    sp = spline(X, Y, spline_kind)
    assert sp.backend == spline_kind
    assert np.allclose(sp(X), Y, atol=1e-7), f"{spline_kind} node exactness failed"


@pytest.mark.parametrize("spline_kind", SPLINE_KIND)
def test_all_backends_approximate_smooth_function(spline_kind):
    """在平滑函数上, 所有后端逼近误差都应在合理范围内。"""
    sp = spline(X, Y, spline_kind)
    err = float(np.abs(sp(XI) - GT).max())
    assert err < 0.2, f"{spline_kind} max err {err:.3e} too large on sin(11 nodes)"


def test_default_backend_is_univariate_spline_exact_interp():
    """默认对外表现为 UnivariateSpline, 且默认 s=0 (精确插值)。"""
    sp = spline(X, Y)
    assert sp.backend == "UnivariateSpline"
    assert sp.params.get("s") == 0
    # 默认配置与显式 UnivariateSpline(s=0) 一致
    sp2 = spline(X, Y, spline_kind="UnivariateSpline", s=0)
    assert np.allclose(sp(XI), sp2(XI), atol=1e-12)


def test_cubic_polynomial_reproduced_exactly_by_cubic_backends():
    """三次样条 (k=3, not-a-knot) 应精确还原三次多项式。"""
    rng = np.random.default_rng(7)
    x = np.sort(rng.uniform(-2, 2, 16))
    p = lambda t: t ** 3 - 2.0 * t + 0.5
    for backend in ("UnivariateSpline", "CubicSpline"):
        sp = spline(x, p(x), backend)
        xs = np.linspace(x[0], x[-1], 2001)
        assert np.allclose(sp(xs), p(xs), atol=1e-6), f"{backend} cubic exactness"


def test_derivative_and_integral_proxy():
    """derivative / integral 代理可用且数值正确 (UnivariateSpline)。"""
    rng = np.random.default_rng(3)
    x = np.linspace(-1.0, 2.0, 13)
    p = lambda t: t ** 3 - 2.0 * t + 0.5
    sp = spline(x, p(x))  # UnivariateSpline, s=0, k=3
    xs = np.linspace(x[0], x[-1], 501)
    d = sp.derivative(1)
    assert np.allclose(d(xs), 3 * xs ** 2 - 2.0, atol=1e-6)
    # integral: ∫ p = x^4/4 - x^2 + 0.5x
    a, b = -0.5, 1.3
    exact = (b ** 4 / 4 - b ** 2 + 0.5 * b) - (a ** 4 / 4 - a ** 2 + 0.5 * a)
    assert abs(sp.integral(a, b) - exact) < 1e-8
    # 不支持的属性应给出清晰的 AttributeError
    with pytest.raises(AttributeError):
        sp.definitely_not_an_attribute  # noqa: B018


def test_scalar_and_array_evaluation():
    """标量求值返回 float, 数组求值返回同形数组。"""
    sp = spline(X, Y)
    v = sp(3.0)
    assert isinstance(v, float)
    xi = np.linspace(0, 10, 7)
    out = sp(xi)
    assert isinstance(out, np.ndarray) and out.shape == xi.shape
    assert isinstance(sp([1.0, 2.0]), np.ndarray)


def test_repr_contains_backend():
    sp = spline(X, Y)
    assert "UnivariateSpline" in repr(sp)


# --------------------------------------------------------------------------- #
# 2. 极端输入
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "bad",
    [
        ([], []),                       # 空
        ([1.0], [2.0]),                 # 单点
        (X, Y[:5]),                     # 长度不一致
        (np.column_stack([X, X]), Y),   # 2-D x
        (X, np.column_stack([Y, Y])),   # 2-D y
        (X, ["a"] * len(X)),            # 非数值
        (np.r_[X, [np.nan]], np.r_[Y, [1.0]]),   # NaN
        (np.r_[X, [1.0]], np.r_[Y, [np.inf]]),   # Inf
        (np.array([0.0, 1.0, 2.0, 2.0, 3.0]), np.arange(5.0)),  # 重复 x (已排序)
    ],
)
def test_invalid_inputs_raise_valueerror(bad):
    x, y = bad
    with pytest.raises(ValueError):
        spline(x, y)


def test_duplicate_nodes_raise():
    """重复 x 节点直接报错 (浮点舍入碰撞同理, 不做去抖)。"""
    with pytest.raises(ValueError, match="strictly increasing"):
        spline(np.array([1.0, 1.0]), np.array([1.0, 2.0]))


def test_unsorted_x_sorts_with_warning_and_matches_sorted():
    """乱序 (但唯一) 输入: UserWarning + 与升序输入结果一致。"""
    rng = np.random.default_rng(11)
    perm = rng.permutation(len(X))
    with pytest.warns(UserWarning, match="reordered"):
        sp_shuffled = spline(X[perm], Y[perm])
    sp_sorted = spline(X, Y)
    assert np.allclose(sp_shuffled(XI), sp_sorted(XI), atol=1e-12)


def test_constant_y_gives_constant_spline():
    """常量 y 返回常量插值 (UnivariateSpline 界外也保持常量)。"""
    c = 3.14159
    sp = spline(X, np.full_like(X, c))
    assert np.allclose(sp(np.linspace(-5, 15, 201)), c, atol=1e-9)


@pytest.mark.parametrize("spline_kind", ["CubicSpline", "PCHIP", "Akima"])
def test_two_point_degenerate(spline_kind):
    """两节点退化解: 默认 k=3 的 UnivariateSpline 明确报错, 其余后端可用。"""
    with pytest.raises(ValueError, match="at least"):
        spline(X[:2], Y[:2])  # UnivariateSpline 默认 k=3 需要 >= 4 节点
    sp = spline(X[:2], Y[:2], spline_kind)
    assert np.allclose(sp(np.array([X[0], X[1]])), Y[:2], atol=1e-12)


def test_two_point_univariate_with_k1():
    """UnivariateSpline(k=1) 允许两节点线性插值。"""
    sp = spline(X[:2], Y[:2], k=1)
    assert np.allclose(sp(np.array([X[0], X[1]])), Y[:2], atol=1e-12)


@pytest.mark.parametrize(
    "spline_kind",
    ["UNIVARIATESPLINE", "cubic_spline", "cubic", "pchipinterpolator", "Akima1DInterpolator"],
)
def test_non_canonical_kind_raises(spline_kind):
    """非 canonical 后端名 (大小写/别名/分隔符变体) 一律拒绝。"""
    with pytest.raises(ValueError, match="Unknown spline_kind"):
        spline(X, Y, spline_kind=spline_kind)


def test_unknown_backend_raises_with_options():
    with pytest.raises(ValueError, match="Unknown spline_kind") as ei:
        spline(X, Y, spline_kind="no-such")
    assert "UnivariateSpline" in str(ei.value)


def test_large_sample_smoke():
    """大样本冒烟: 10 万节点仍可正确构造与求值。"""
    n = 100_000
    x = np.linspace(0, 1, n)
    y = np.sin(50 * x)
    sp = spline(x, y)  # UnivariateSpline s=0
    xi = x[::137]
    err = np.abs(sp(xi) - np.sin(50 * xi)).max()
    assert err < 1e-6


def test_extrapolation_semantics():
    """外推语义随后端而异 (透明代理, 不做截断/掩膜; scipy 1.15 实测)。"""
    # UnivariateSpline: 恒外推, 界外有限值
    sp = spline(X, Y)
    assert np.all(np.isfinite(sp(np.array([-5.0, 15.0]))))
    # CubicSpline / PCHIP: 默认外推; extrapolate=False -> 界外 NaN (不抛错)
    for backend in ("CubicSpline", "PCHIP"):
        cs = spline(X, Y, backend)
        assert np.all(np.isfinite(cs(np.array([-5.0, 15.0]))))
        cs_nan = spline(X, Y, backend, extrapolate=False)
        assert np.isnan(cs_nan(np.array([15.0])))
    # Akima: 默认界外 NaN; extrapolate=True 打开外推
    ak = spline(X, Y, "Akima")
    assert np.isnan(ak(np.array([15.0])))
    ak_x = spline(X, Y, "Akima", extrapolate=True)
    assert np.all(np.isfinite(ak_x(np.array([-5.0, 15.0]))))


# --------------------------------------------------------------------------- #
# 3. 性能 / 耗时 (宽松上限防回归; 明细打印)
# --------------------------------------------------------------------------- #
def test_performance_fit_and_eval():
    """
    各后端在中等/大样本上的拟合与求值耗时。

    仅做宽松上界断言 (防止灾难性回归), 精确中位数在 -s 输出中可见。
    一次本机测量 (Windows, py3.10, numpy 2.2.6, scipy 1.15.3):
      n=20_000 拟合: UnivariateSpline ~2.2ms, CubicSpline ~1.1ms,
                     PCHIP ~0.8ms, Akima ~1.0ms
      n=20_000, 1e6 点求值: UnivariateSpline ~44.9ms, 其余 ~6.3-6.5ms
    """
    x = np.linspace(0, 1, 20_000)
    y = np.sin(40 * x)
    xi = np.linspace(0, 1, 1_000_000)

    report = {}
    for backend in SPLINE_KIND:
        t_fit = _median_seconds(lambda: spline(x, y, backend), repeats=3)

        def _eval():
            sp = spline(x, y, backend)
            sp(xi)

        t_eval = _median_seconds(_eval, repeats=3)
        report[backend] = (t_fit, t_eval)
        print(f"[spline-perf] {backend:16s} fit={t_fit * 1e3:8.2f} ms "
              f"eval(1e6)={t_eval * 1e3:8.2f} ms")
        assert t_fit < 2.0, f"{backend} fit too slow: {t_fit:.2f}s"
        assert t_eval < 2.0, f"{backend} eval too slow: {t_eval:.2f}s"


# --------------------------------------------------------------------------- #
# 4. Cache 集成 (Utils getter 统一路径: 惰性注册 / 全局取用)
# --------------------------------------------------------------------------- #
def test_utils_getters_register_and_return_cached_instances():
    """Utils.get_spline/get_hilbert/get_envelope: 首次访问惰性注册同一实例。"""
    from src.Modal_Decomposition import Utils as U

    sp_mod = U.get_spline()
    hb_mod = U.get_hilbert()
    env_mod = U.get_envelope()
    assert sp_mod is U.get_spline() is cache.get("Modal_Decomposition.Utils.Spline")
    assert hb_mod is U.get_hilbert() is cache.get("Modal_Decomposition.Utils.Hilbert")
    assert env_mod is U.get_envelope() is cache.get("Modal_Decomposition.Utils.Envelope")
    assert callable(sp_mod.spline)
    assert callable(hb_mod.hilbert)
    assert callable(env_mod.envelope)


def test_other_utils_tools_share_the_same_cache_path():
    """其余 Utils 工具 (含 monotonic 单调性函数) 走同一缓存注册/取值路径。"""
    from src.Modal_Decomposition import Utils as U

    mono = U.get_monotonicity()
    assert mono is cache.get("Modal_Decomposition.Utils.Monotonicity")
    assert callable(mono.monotonic) and callable(mono.is_monotonic)
    # monotonic 经缓存实例与直接导出函数是同一函数对象 (同一模块实例)
    assert mono.is_monotonic is U.is_monotonic

    chk = U.get_check()
    assert chk is cache.get("Modal_Decomposition.Utils.Check")
    assert callable(chk.Check_Time_and_Signal)

    mem = U.get_memory()
    assert mem is cache.get("Modal_Decomposition.Utils.Memory")
    assert callable(mem.should_use_memmap)

    seed = U.get_seed_module()
    assert seed is cache.get("Modal_Decomposition.Utils.Seed")
    assert callable(seed.set_seed)

    # 幂等: 重复 getter 返回同一实例; 七个工具键全部注册
    assert U.get_check() is chk and U.get_memory() is mem
    expected = {
        f"Modal_Decomposition.Utils.{m}"
        for m in ("Check", "Memory", "Monotonicity", "Seed",
                  "Hilbert", "Spline", "Envelope")
    }
    assert expected <= set(cache.names())
