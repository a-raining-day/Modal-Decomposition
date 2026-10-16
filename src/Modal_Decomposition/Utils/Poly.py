import numpy as np
from .Check import Check
from ..Define import SAFE_ARGS

__all__ = ["fast_poly"]

def fast_poly(x_ref: np.ndarray, y_ref: np.ndarray, poly_order: int, x_new: np.ndarray, safe: SAFE_ARGS = "unknown") -> np.ndarray:
    """
    在 ``(x_ref, y_ref)`` 上拟合 ``poly_order`` 次多项式, 求 ``x_new`` 处的值。

    自变量先减去均值、再除以跨度, 然后解一次最小二乘: 外推场景下原始自变量
    (时间戳量级) 会让 Vandermonde 病态 —— ``np.polyfit`` 在这种输入上会抛
    RankWarning 并给出被噪声放大的系数; 而归一化不改变最小二乘解 (自变量的
    可逆仿射变换下解不变)。同时不构造 ``np.poly1d`` 对象, 省掉其构造开销。
    """
    safe = Check.validate_safe(safe)
    x_ref = Check.to_array(x_ref, safe=safe)
    y_ref = Check.to_array(y_ref, safe=safe)
    x_new = Check.to_array(x_new, safe=safe)

    center = x_ref.mean()
    span = x_ref.max() - x_ref.min()
    scale = span if span > 0 else 1.0

    basis: np.ndarray = np.vander((x_ref - center) / scale, poly_order + 1, increasing=True)
    coefficients: np.ndarray = np.linalg.lstsq(basis, y_ref, rcond=None)[0]
    target: np.ndarray = np.vander((x_new - center) / scale, poly_order + 1, increasing=True)

    return target @ coefficients