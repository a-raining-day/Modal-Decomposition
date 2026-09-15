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

``mirror_signal`` —— **整条信号**的端点镜像 (VMD / vmdpy 语义)
-------------------------------------------------------------
与上面两个"极值点"语义不同, VMD 系方法要对**每个样本**做端点外延:
``[flip(S[:L]), S, flip(S[N-R:])]``。本模块统一提供 :func:`mirror_signal`,
``mode`` 控制反射约定 (见其 docstring), ``left``/``right`` 控制两端长度,
``out`` + ``chunk_size`` 支持分块写入 memmap (外存路径不额外占内存)。

Layering (与 Utils 其余工具一致): 本模块自身不接触缓存;
``Utils.get_mirror()`` 首次访问时惰性 import 并以键
``"Modal_Decomposition.Utils.Mirror"`` 注册进进程级 import 缓存。
"""

import numpy as np
from ..Base.ConstDefine import DEFAULT_NUMPY_TYPE

__all__ = ["mirror_extrema", "mirror_signal"]

#: ``mirror_signal`` 的反射约定。
_SIGNAL_MODES = ("symmetric", "reflect")


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


def mirror_signal(
    S: np.ndarray,
    left: int = None,
    right: int = None,
    mode: str = "symmetric",
    out=None,
    chunk_size: int = None,
):
    """
    整条信号的端点镜像延拓: ``[flip(S[:left]), S, flip(S[N-right:])]``。

    Parameters
    ----------
    S : np.ndarray
        1-D 信号 (也接受 memmap)。
    left, right : int | None
        两端镜像的样本数。默认 ``left = N//2``、``right = N - N//2`` ⇒ 总长恰为
        ``2N`` (VMD/vmdpy 的 ``fMirr`` 约定, 且对奇数 N 也不丢样本)。
    mode : {"symmetric", "reflect"}
        反射约定:
        ``"symmetric"`` —— 重复边界样本 (``x[-1] = S[0]``), 即 VMD/vmdpy 与
        ``np.pad(mode="symmetric")`` 的语义 (默认);
        ``"reflect"``   —— 关于边界样本反射 (``x[-1] = S[1]``), 与
        :func:`mirror_extrema` 在密集样本上的反射一致 (两者**不可互换**)。
    out : np.ndarray | None
        可选目标缓冲 (可为 ``np.memmap``)。给出时结果写入其中并返回它,
        长度须等于 ``left + N + right``。
    chunk_size : int | None
        分块搬运的粒度 (与 ``out`` 搭配用于外存路径, 限制工作集);
        ``None`` 时一次性写入。

    Returns
    -------
    np.ndarray
        长度 ``left + N + right`` 的镜像信号 (``out`` 给定时即 ``out``)。

    Raises
    ------
    ValueError
        未知 ``mode``; ``S`` 非 1-D; ``out`` 长度不符。
    """
    if mode not in _SIGNAL_MODES:
        raise ValueError(f"mode must be one of {_SIGNAL_MODES}, got {mode!r}")

    S = np.asarray(S)
    if S.ndim != 1:
        raise ValueError(f"mirror_signal expects a 1-D signal, got ndim={S.ndim}")
    n = S.size
    left = n // 2 if left is None else int(left)
    right = n - n // 2 if right is None else int(right)
    if left < 0 or right < 0 or left + right + n > n + 2 * n:
        raise ValueError(f"invalid left/right ({left}/{right}) for N={n}")

    total = left + n + right
    if out is None:
        out = np.empty(total, dtype=S.dtype)
    elif out.shape[0] != total:
        raise ValueError(f"out length {out.shape[0]} != left+N+right = {total}")

    # 分块搬运: 按**目标区间**对齐取源 (写成 src[a:b][::-1] 只在单块时正确)。
    def _spans(m):
        step = max(1, int(chunk_size) if chunk_size else m)
        for a in range(0, m, step):
            yield a, min(a + step, m)

    if mode == "symmetric":          # x[-k] = S[k-1]  (out[k] = S[left-1-k])
        for a, b in _spans(left):
            out[a:b] = S[left - b:left - a][::-1]
        for a, b in _spans(right):   # x[N-1+j] = S[N-1-j]
            out[left + n + a:left + n + b] = S[n - b:n - a][::-1]
    else:                            # reflect: x[-k] = S[k]  (out[k] = S[left-k])
        for a, b in _spans(left):
            out[a:b] = S[left - b + 1:left - a + 1][::-1]
        for a, b in _spans(right):   # x[N+j] = S[N-2-j]
            out[left + n + a:left + n + b] = S[n - 1 - b:n - 1 - a][::-1]

    out[left:left + n] = S
    if hasattr(out, "flush"):
        out.flush()
    return out
