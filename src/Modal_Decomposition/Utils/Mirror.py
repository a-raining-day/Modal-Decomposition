"""
Extrema endpoint mirror extension (端点镜像外拓, 全库共享)。

模态分解 (EMD / LMD / 极值样条包络等) 都需要在信号端点外**镜像外延极值**,
再对镜像后的极值做样条插值, 以避免端点塌陷 / 边界伪振荡。本模块统一提供
``mirror_extrema``, 覆盖两种镜像语义:

- ``nbsym`` (EMD 经典语义): 每端把 ``nbsym`` 个极值按首/末极值对称外延,
  镜像位置落在 ``[0, n-1]`` 区间之外 (经典 nbsym 端点处理, PyEMD 同款);
- ``edges=(首值, 末值, 样本数)`` (LMD 边界语义): 最外极值不落在边界时, 在
  边界 ``0`` / ``n-1`` 处按 ``2*边界信号值 - 最外极值`` 奇反射补点, 使极值
  覆盖整个信号区间。

两种语义可组合: 先按 ``nbsym`` 外延, 再检查边界覆盖 (组合使用时镜像值的
首个/末个定义以 nbsym 镜像后的序列为准)。

Layering (与 Utils 其余工具一致): 本模块自身不接触缓存;
``Utils.get_mirror()`` 首次访问时惰性 import 并以键
``"Modal_Decomposition.Utils.Mirror"`` 注册进进程级 import 缓存。
"""

import numpy as np
from ..Base.ConstDefine import DEFAULT_NUMPY_TYPE

__all__ = ["mirror_extrema"]


def mirror_extrema(
    idx: np.ndarray,
    vals: np.ndarray,
    nbsym: int = 3,
    edges = None,
    _dtype: np.dtype = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    端点镜像外延 (EMD nbsym / LMD 边界两用, 全库共享)。

    Parameters
    ----------
    idx : np.ndarray (1-D)
        极值位置 (须严格递增)。
    vals : np.ndarray (1-D)
        与 ``idx`` 对应的极值取值 (``signal[idx]``)。
    nbsym : int
        EMD 语义: 每端镜像的极值个数 (默认 0 = 不镜像; 极值少于 2 个时跳过)。
    edges : tuple[float, float, int] | None
        LMD 语义: ``(信号首值, 信号末值, 样本数)``; None 时不补边界点。

    Returns
    -------
    (positions, values)
        ``positions`` : np.ndarray (float64) — 镜像后位置, 严格递增;
        ``values``    : np.ndarray — 与输入 ``vals`` 同 dtype 的对应取值。

    Examples
    --------
    >>> import numpy as np
    >>> from Modal_Decomposition.Utils.Mirror import mirror_extrema
    >>> idx = np.array([5, 9]); vals = np.array([2.0, 3.0])
    >>> pos, val = mirror_extrema(idx, vals, nbsym=1)
    >>> pos.tolist()   # 1 个左镜像 + 原极值 + 1 个右镜像
    [1.0, 5.0, 9.0, 13.0]
    """
    if not isinstance(idx, np.ndarray):
        if _dtype is not None:
            idx = np.asarray(idx, dtype=_dtype)
        else:
            idx = np.asarray(idx, dtype=DEFAULT_NUMPY_TYPE)
    if not isinstance(vals, np.ndarray):
        if _dtype is not None:
            vals = np.asarray(vals, dtype=_dtype)
        else:
            vals = np.asarray(vals, dtype=DEFAULT_NUMPY_TYPE)

    if isinstance(idx, np.ndarray) and _dtype is not None and _dtype != idx.dtype:
        idx = np.asarray(idx, dtype=_dtype)
    if isinstance(vals, np.ndarray) and _dtype is not None and _dtype != vals.dtype:
        vals = np.asarray(vals, dtype=_dtype)

    _dtype = vals.dtype if _dtype is None else _dtype

    if idx.ndim != 1 or vals.ndim != 1 or idx.size != vals.size:
        raise ValueError(
            "mirror_extrema requires 1-D idx/vals of equal length"
        )

    positions = idx.astype(np.float64)
    values = vals.copy()
    if positions.size == 0:
        return positions, values

    # --- EMD nbsym 语义: 每端镜像 nbsym 个极值 ---------------------------- #
    if nbsym > 0 and len(positions) >= 2:
        n = min(nbsym, len(positions) - 1)
        p = len(positions)
        L = p + 2* n

        new_positions = np.empty(L, dtype=positions.dtype)
        new_values = np.empty(L, dtype=values.dtype)

        # 左侧扩展
        new_positions[:n] = 2 * positions[0] - positions[1:n + 1][::-1]
        new_values[:n] = values[1:n + 1][::-1]

        # 中间原始数据
        new_positions[n:n + p] = positions
        new_values[n:n + p] = values

        # 右侧扩展
        new_positions[n + p:] = 2 * positions[-1] - positions[-2:-(n + 2):-1]
        new_values[n + p:] = values[-1 - n:-1][::-1]

        positions = new_positions
        values = new_values

    # --- LMD 边界语义: 未覆盖边界处奇反射补点 ----------------------------- #
    if edges is not None:
        lo, hi, n_samples = float(edges[0]), float(edges[1]), int(edges[2])

        left_pad = positions[0] > 0
        right_pad = positions[-1] < n_samples - 1

        total_len = len(positions) + int(left_pad) + int(right_pad)
        new_positions = np.empty(total_len, dtype=positions.dtype)
        new_values = np.empty(total_len, dtype=values.dtype)

        idx = 0
        if left_pad:
            new_positions[0] = 0.0
            new_values[0] = 2.0 * lo - float(values[0])
            idx = 1

        new_positions[idx:idx + len(positions)] = positions
        new_values[idx:idx + len(values)] = values
        idx += len(positions)

        if right_pad:
            new_positions[idx] = float(n_samples - 1)
            new_values[idx] = 2.0 * hi - float(values[-1])

        positions = new_positions
        values = new_values

    return positions, values
