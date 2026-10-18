"""
``Modal_Decomposition.Utils.Poly`` 模块测试。

覆盖：导出一致性、多项式精确性、与 ``numpy.polyfit`` 等价、退化输入、
不改写调用方数组、确定性。
"""

import numpy as np
import pytest

from Modal_Decomposition.Utils import fast_poly
from Modal_Decomposition.Utils.Poly import __all__ as POLY_ALL


def test_all_matches_public_surface():
    """`__all__` 里的名字都能取到，且 `fast_poly` 在其中。"""
    import Modal_Decomposition.Utils.Poly as P
    assert "fast_poly" in POLY_ALL
    for n in POLY_ALL:
        assert hasattr(P, n), f"{n} 在 __all__ 里但取不到"


def test_exact_on_polynomial():
    """二次多项式上必须精确（不是近似）。"""
    x = np.linspace(-1.0, 1.0, 20)
    y = 1.0 + 2.0 * x + 3.0 * x ** 2
    out = fast_poly(x, y, 2, x)
    assert out.shape == x.shape
    assert np.allclose(out, y, atol=1e-8)


def test_matches_numpy_polyfit():
    """与 `numpy.polyfit` 在同一组点上等价。"""
    rng = np.random.default_rng(0)
    x = np.linspace(-2.0, 2.0, 50)
    y = np.sin(x) + 0.05 * rng.standard_normal(x.size)
    x_new = np.linspace(-1.5, 1.5, 37)
    got = fast_poly(x, y, 3, x_new)
    ref = np.polyval(np.polyfit(x, y, 3), x_new)
    assert np.allclose(got, ref, atol=1e-8)


@pytest.mark.parametrize("order", [1, 2, 4])
def test_various_orders_finite(order):
    """不同阶数都返回有限值（阶数高于真实结构时也不应发散成 inf/nan）。"""
    x = np.linspace(0.0, 1.0, 30)
    y = np.cos(3.0 * x)
    out = fast_poly(x, y, order, x)
    assert out.shape == x.shape
    assert np.all(np.isfinite(out))


def test_degenerate_inputs():
    """两点/常值/单点输入：要么给出有限结果，要么抛明确异常，不能静默返回垃圾。"""
    x = np.array([0.0, 1.0])
    y = np.array([1.0, 3.0])
    out = fast_poly(x, y, 1, x)
    assert np.allclose(out, y, atol=1e-8)

    xc = np.linspace(0.0, 1.0, 10)
    yc = np.full(10, 2.5)
    outc = fast_poly(xc, yc, 1, xc)
    assert np.allclose(outc, 2.5, atol=1e-8)


def test_input_not_mutated():
    """不改写调用方数组。"""
    x = np.linspace(0.0, 1.0, 20)
    y = x ** 2
    x0, y0 = x.copy(), y.copy()
    fast_poly(x, y, 2, x)
    assert np.array_equal(x, x0)
    assert np.array_equal(y, y0)


def test_deterministic():
    """同输入两次调用逐位相同。"""
    x = np.linspace(0.0, 1.0, 25)
    y = np.exp(x)
    a = fast_poly(x, y, 2, x)
    b = fast_poly(x, y, 2, x)
    assert np.array_equal(a, b)


def test_invalid_safe_rejected():
    """非法 `safe` 取值必须被拒绝。"""
    x = np.linspace(0.0, 1.0, 10)
    with pytest.raises((ValueError, TypeError)):
        fast_poly(x, x, 1, x, safe="not-a-mode")
