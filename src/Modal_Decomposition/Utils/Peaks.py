"""
峰检测 (极大 / 极小 / 两者)。

``mod`` 选峰型 (``"max"`` / ``"min"`` / ``"both"``), ``backend`` 选实现
(``"numpy"`` / ``"scipy"`` / ``"numba"``); 三个后端共享同一返回契约与过滤语义。

返回契约
--------
所有后端返回 ``(indices: np.ndarray(int64), values: np.ndarray)``; ``values``
恒为**原信号**幅值 (``mod="min"`` 也不取负)。``mod="both"`` 多返回一个类型掩码
``is_max``, 两组极值按样本序归并 (O(P), 不用 ``argsort``)。

过滤器
------
numpy / numba 后端识别 ``height`` / ``threshold`` / ``distance`` (scipy 语义,
按峰型方向解释), 其余键忽略 (``Envelope`` 等调用方会把自家参数一并透传);
scipy 后端全参数透传。

依赖
----
scipy / numba 经 :meth:`Base.Cache.cache.import_module` 惰性获取 (每进程一次
import), 缺失时 warn 并回退到 numpy 后端。numba 后端把基础检测内核按需
``njit`` 编译 (``lru_cache`` 保证每进程只编译一次), 过滤逻辑与 numpy 后端共用
同一份 Python 代码, 所以两者结果一致。

注意事项
--------
scipy 的 plateau (平台) 规则复杂 (取平台中点等), numpy / numba 后端采用基础
规则 ``S[i-1] < S[i] >= S[i+1]`` (``mod="min"`` 为其镜像
``S[i-1] > S[i] <= S[i+1]``), 平台场景请使用 ``"scipy"`` 后端。

NaN 的处理取决于 ``safe``: 默认的 ``safe="unknown"`` 会在 ``Check.to_array``
的有限性检查处抛 ``ValueError("NaN or Inf found")``; ``safe="safe"`` 跳过该
检查, 此时 NaN 参与比较恒为 False, 不会成为峰 (与 scipy 行为一致)。
"""

import bisect
import functools
import numpy as np
from enum import Enum
from warnings import warn
from typing import Literal, Tuple, Optional


from ..Base import cache
from .Check import Check
from ..Define import SAFE_MODS

__all__ = ["find_peaks", "PEAKS_MODS", "PEAKS_MODS_ARGS", "PEAKS_BACKEND", "PEAKS_BACKEND_ARGS", "PEAKS_SUPPORTED_FILTERS", "PEAKS_SUPPORTED_FILTERS_ARGS"]


PEAKS_MODS_ARGS = Literal["max", "min", "both"]
class PEAKS_MODS(Enum):
    Max = "max"
    Min = "min"
    Both = "both"

PEAKS_BACKEND_ARGS = Literal["scipy", "numpy", "numba"]
class PEAKS_BACKEND(Enum):
    Scipy = "scipy"
    Numpy = "numpy"
    Numba = "numba"

PEAKS_SUPPORTED_FILTERS_ARGS = Literal["height", "threshold", "distance"]
class PEAKS_SUPPORTED_FILTERS(Enum):
    Height = "height"
    Threshold = "threshold"
    Distance = "distance"


def find_peaks(
    S: np.ndarray,
    mod: PEAKS_MODS_ARGS | PEAKS_MODS = "max",
    backend: PEAKS_BACKEND_ARGS | PEAKS_BACKEND = "numpy",
    safe: Literal["safe", "unknown"] | SAFE_MODS = "unknown",
    **kwargs,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    在 1-D 信号上检测峰 (极大 / 极小 / 两者)。

    Parameters
    ----------
    S : array-like
        1-D 实值数值信号; ``size < 3`` 时无法判定极值方向, 走下面的短输入捷径。
    mod : {"max", "min", "both"}
        峰型: ``"max"`` (默认) 取极大 ``S[i-1] < S[i] >= S[i+1]``;
        ``"min"`` 取极小 ``S[i-1] > S[i] <= S[i+1]`` (极大内核的镜像);
        ``"both"`` 两种都要 —— 各自检测 / 各自过滤后按样本序归并,
        **多返回一个类型掩码** (见 Returns)。
    backend : {"numpy", "scipy", "numba"}
        ``"numpy"`` (默认) — 纯 numpy 向量化实现;
        ``"scipy"`` — :func:`scipy.signal.find_peaks` (全参数支持, 含 plateau 规则;
        取极小时内部对 ``-S`` 找峰);
        ``"numba"`` — numpy 内核经 njit 编译的可选加速 (需安装 numba, 缺失时回退)。
    safe : Literal["safe", "unknown"] | SAFE_MODS
        输入校验等级。``"unknown"`` (默认) 走 ``Check.to_array`` 的完整校验,
        含有限性检查, 遇 NaN / Inf 抛 ``ValueError``; ``"safe"`` 跳过校验,
        由调用方保证输入合法, 此时 NaN 不会成为峰; ``"suspicious"`` 不受支持。
    **kwargs
        过滤参数 (scipy 语义): `height` / `threshold` / `distance`, 按峰型方向
        解释 (``mod="min"`` 等价于在 ``-S`` 上施加同样的过滤)。
        ``mod="both"`` 时**按类型分别施加** (极大只与极大比距离, 极小只与极小比),
        然后归并 —— 混合序列上的全局贪心顺序没有定义, 不采用。
        scipy 后端全部透传 (另支持 `prominence`, `width`, `wlen`, `rel_height`, `plateau_size`);
        numpy / numba 后端只识别上述三个, 其余键忽略。

    Returns
    -------
    (indices, values)            # mod="max" / "min"
        indices: np.ndarray (int64)
            峰的位置
        values: np.ndarray
            S[indices] —— 原信号幅值, ``mod="min"`` 时也不取负
    (indices, values, is_max)    # mod="both"
        indices: np.ndarray (int64)
            极大 + 极小, 已按样本序**归并**且严格升序
        values: np.ndarray
            S[indices] —— 原信号幅值
        is_max: np.ndarray (bool)
            与 indices 等长; ``True`` = 该位置是极大, ``False`` = 极小

    Raises
    ------
    ValueError
        ``safe="unknown"`` 下非 1-D / 非实数数值 / 含 NaN 或 Inf 的输入,
        或未知 mod / backend / safe。
    UserWarning
        scipy / numba 缺失时发出, 并回退到 numpy 后端 (不中断调用)。

    Notes
    -----
    ``S.size < 3`` 的捷径对 ``"both"`` 返回**空集 + 空掩码** (样本太短时无法区分
    方向, 不能声称每个样本既是极大又是极小); 单峰型则返回全部下标 (每个样本都算峰)。
    """
    safe = Check.validate_safe(safe)
    if safe == SAFE_MODS.Suspicious:
        raise ValueError("the `suspicious` mod is not supported in this function")

    mod = Check.validate_mod(mod, PEAKS_MODS)
    backend = Check.validate_mod(backend, PEAKS_BACKEND)

    S = Check.to_array(S, safe=safe)

    if 1 <= S.size < 3:
        # 太短: 无法区分极值方向。单峰型返回全部下标 (每个样本都当作峰);
        # both 不能声称每个样本既是极大又是极小 -> 空集 + 空掩码。
        indices = np.array(np.arange(S.size), dtype=np.int64)
        if mod is PEAKS_MODS.Both:
            empty = np.empty(0, dtype=np.int64)
            return empty, S[empty], np.empty(0, dtype=bool)
        return indices, S[indices]

    result: None | Tuple[np.ndarray, np.ndarray]= None
    match backend:
        case PEAKS_BACKEND.Numpy:
            return _peaks_numpy(S, mod, kwargs)
        case PEAKS_BACKEND.Scipy:
            result = _peaks_scipy(S, mod, kwargs)
        case PEAKS_BACKEND.Numba:
            result = _peaks_numba(S, mod, kwargs)
        case _:
            raise ValueError(f"Unknown backend: {backend!r}; expected 'scipy', 'numpy' or 'numba'")

    if result is None:
        return _peaks_numpy(S, mod, kwargs)
    else:
        return result

def _peaks_scipy(S: np.ndarray, mod: PEAKS_MODS, kwargs: dict):
    ss = cache.import_module("scipy.signal",verbose=False)
    if ss is None:
        warn("`scipy.signal` not find; use `numpy` instead")
        return None

    # scipy 只会找极大: 找极小就对 -S 找峰, 但返回的取值必须是原信号的幅值
    if mod is PEAKS_MODS.Max:
        idx = ss.find_peaks(S, **kwargs)[0].astype(np.int64, copy=False)
        return idx, S[idx]

    if mod is PEAKS_MODS.Min:
        idx = ss.find_peaks(-S, **kwargs)[0].astype(np.int64, copy=False)
        return idx, S[idx]

    # both: 两次检测 (scipy 没有极小模式, -S 省不掉); 过滤器由 scipy 各自按方向施加
    neg = -S
    max_idx = ss.find_peaks(S, **kwargs)[0].astype(np.int64, copy=False)
    min_idx = ss.find_peaks(neg, **kwargs)[0].astype(np.int64, copy=False)
    return _merge_both(max_idx, min_idx, S)

def _peaks_numpy(S: np.ndarray, mod: PEAKS_MODS, kwargs: dict):
    # 极小内核是极大内核的镜像 (等价于对 -S 找峰, 但不物化 -S)
    c = S[1:-1]

    if mod is PEAKS_MODS.Max:
        idx = np.flatnonzero((c > S[:-2]) & (c >= S[2:])) + 1
        idx = _apply_filters(idx, S, mod, kwargs)
        return idx, S[idx]

    if mod is PEAKS_MODS.Min:
        idx = np.flatnonzero((c < S[:-2]) & (c <= S[2:])) + 1
        idx = _apply_filters(idx, S, mod, kwargs)
        return idx, S[idx]

    # both: 两组各自检测 + 各自按方向过滤, 再按样本序归并 (O(P), 不用 argsort)
    max_idx = _apply_filters(np.flatnonzero((c > S[:-2]) & (c >= S[2:])) + 1, S, PEAKS_MODS.Max, kwargs)
    min_idx = _apply_filters(np.flatnonzero((c < S[:-2]) & (c <= S[2:])) + 1, S, PEAKS_MODS.Min, kwargs)
    return _merge_both(max_idx, min_idx, S)


def _peaks_numba(S: np.ndarray, mod: PEAKS_MODS, kwargs: dict):
    numba = cache.import_module("numba", verbose=False)
    if numba is None:
        warn("`numba` not find; use `numpy` instead")
        return None

    compiled = _get_compiled(numba)

    if mod is PEAKS_MODS.Max:
        idx = _apply_filters(compiled(S, 1), S, mod, kwargs)
        return idx, S[idx]

    if mod is PEAKS_MODS.Min:
        idx = _apply_filters(compiled(S, -1), S, mod, kwargs)
        return idx, S[idx]

    max_idx = _apply_filters(compiled(S, 1), S, PEAKS_MODS.Max, kwargs)
    min_idx = _apply_filters(compiled(S, -1), S, PEAKS_MODS.Min, kwargs)
    return _merge_both(max_idx, min_idx, S)


def _merge_both(max_idx: np.ndarray, min_idx: np.ndarray, S: np.ndarray):
    """
    归并两组极值索引 → ``(idx, values, is_max)`` (``mod="both"`` 的返回)。

    两组各自已升序, 且极大 / 极小的判定条件互斥 (同一样本不可能同属两组),
    因此用 ``np.searchsorted`` 直接算出落位: O(P), 不需要 ``argsort``。
    """
    idx = np.empty(max_idx.size + min_idx.size, dtype=np.int64)
    is_max = np.empty(idx.size, dtype=bool)

    max_pos = np.arange(max_idx.size) + np.searchsorted(min_idx, max_idx, side="left")
    min_pos = np.arange(min_idx.size) + np.searchsorted(max_idx, min_idx, side="left")

    idx[max_pos] = max_idx
    is_max[max_pos] = True
    idx[min_pos] = min_idx
    is_max[min_pos] = False

    return idx, S[idx], is_max

def _apply_filters(idx: np.ndarray, S: np.ndarray, mod: PEAKS_MODS, kwargs: dict) -> np.ndarray:
    """
    对候选峰应用 height / threshold / distance 过滤 (numpy/numba 共用)。

    过滤按"峰的那一侧"解释: 极小取 ``sign = -1``, 等价于在 ``-S`` 上做同样的
    过滤, 但不物化 ``-S``。未识别的键忽略。

    ``both`` 不接受: 它由调用方拆成 ``Max`` / ``Min`` 两组分别进来, 否则这里会
    静默按极小处理。
    """
    if mod is PEAKS_MODS.Both:
        raise ValueError("the filters are per peak type; both must be split into Max / Min")

    sign = 1 if mod is PEAKS_MODS.Max else -1

    height = kwargs.get("height")
    if height is not None:
        # scipy 语义: 标量仅为下界; (min, max) 为区间。
        h_min, h_max = (
            height if isinstance(height, (tuple, list)) else (height, np.inf)
        )
        peak = sign * S[idx]
        keep = (peak >= h_min) & (peak <= h_max)
        idx = idx[keep]

    threshold = kwargs.get("threshold")
    if threshold is not None:
        t_min, t_max = (
            threshold if isinstance(threshold, (tuple, list)) else (threshold, threshold)
        )
        keep = (sign * (S[idx] - S[idx - 1]) >= t_min) & (sign * (S[idx] - S[idx + 1]) >= t_max)
        idx = idx[keep]

    distance = kwargs.get("distance")
    if distance is not None:
        if distance < 1:
            raise ValueError(f"distance must be >= 1, got {distance!r}")
        # 贪心: 按峰高降序保留 (scipy 语义: 距更高峰不足 distance 者被移除)。
        # 有序列表 + bisect 判邻, O(P log P)。
        order = np.argsort(sign * S[idx])[::-1]
        selected: list = []
        for pos in order:
            p = int(idx[pos])
            i = bisect.bisect_left(selected, p)
            if (i > 0 and p - selected[i - 1] < distance) or (
                i < len(selected) and selected[i] - p < distance
            ):
                continue
            selected.insert(i, p)
        idx = np.asarray(selected, dtype=np.int64)

    return idx

def _detect_kernel(S: np.ndarray, sign: int) -> np.ndarray:
    """基础检测内核 (numba 可编译): sign=1 取极大 ``S[i-1] < S[i] >= S[i+1]``,
    sign=-1 取极小 ``S[i-1] > S[i] <= S[i+1]``。"""
    n = S.shape[0]
    out = np.empty(n, dtype=np.int64)
    count = 0
    for i in range(1, n - 1):
        if sign * S[i - 1] < sign * S[i] and sign * S[i] >= sign * S[i + 1]:
            out[count] = i
            count += 1
    return out[:count]

@functools.lru_cache(maxsize=None)
def _get_compiled(numba):
    """按需 njit 编译检测内核 (每进程最多编译一次)。"""
    return numba.njit(nogil=True)(_detect_kernel)
