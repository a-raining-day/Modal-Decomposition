"""
Peak detection (统一峰检测)。

优化内容 (相对旧版)
--------------------
- **numpy 后端修复**: 旧实现 ``logical_xor(S[:-1]-S[1:], S[1:]-S[:-1])`` 对
  任意符号都恒为 False, 且返回的是长度 n-1 的掩码而非峰索引。现改为与
  :func:`scipy.signal.find_peaks` 同语义的向量化检测, 返回 ``int64`` 索引;
- **统一返回契约**: 所有后端都返回 ``(indices: np.ndarray(int64),
  properties: dict)``, 与 scipy 一致;
- **统一过滤参数**: numpy / numba 后端支持 ``height`` / ``threshold`` /
  ``distance`` (scipy 语义), 其它参数显式 ``NotImplementedError`` (不静默
  忽略); scipy 后端全参数透传;
- **依赖获取**: scipy / numba 经 ``cache.import_module`` 惰性获取 (一次
  import, 无 try/except 样板), 移除了旧式模块级 ``Cache`` 用法;
- **numba 分支修复**: 旧代码引用未定义变量 ``module``。现改为把向量化
  检测内核按需 ``njit`` 编译 (``lru_cache`` 保证每进程只编译一次), 过滤
  逻辑与 numpy 后端共用同一份 Python 代码, 保证两后端结果一致;
- **输入校验**: 1-D、长度 >= 3、实数数值 dtype。

注意: scipy 的 plateau (平台) 规则复杂 (取平台中点等), numpy / numba
后端采用基础规则 ``S[i-1] < S[i] >= S[i+1]``, 平台场景请使用 ``"scipy"``
后端。NaN 在比较中为 False, 不会成为峰 (与 scipy 行为一致)。
"""

import bisect
import functools
from typing import Literal, Optional, Union

import numpy as np

from ..Base.Cache import cache

__all__ = ["find_peaks"]

#: numpy / numba 后端支持的过滤参数 (scipy 语义)。
_SUPPORTED_FILTERS = ("height", "threshold", "distance")


# --------------------------------------------------------------------------- #
# 校验与过滤 (各后端共用)
# --------------------------------------------------------------------------- #
def _validate(S) -> np.ndarray:
    """规范化输入: 1-D、实数数值、长度 >= 3 (不足则返回空结果)。"""
    arr = S if isinstance(S, np.ndarray) else np.asarray(S)
    if arr.ndim != 1:
        raise ValueError(f"Peaks requires a 1-D signal, got {arr.ndim}-D")
    if arr.dtype.kind not in "iuf":
        raise ValueError(
            f"Peaks requires a real numeric dtype, got {arr.dtype}"
        )
    if arr.size < 3:
        return np.array([], dtype=np.int64), True
    return arr, False


def _apply_filters(idx: np.ndarray, S: np.ndarray, kwargs: dict) -> np.ndarray:
    """
    对候选峰应用 height / threshold / distance 过滤 (numpy/numba 共用)。

    Raises
    ------
    NotImplementedError
        出现不支持的过滤参数 (列出支持集)。
    """
    unknown = set(kwargs) - set(_SUPPORTED_FILTERS)
    if unknown:
        raise NotImplementedError(
            f"backend supports only {_SUPPORTED_FILTERS}, "
            f"got unsupported: {sorted(unknown)}"
        )

    height = kwargs.get("height")
    if height is not None:
        # scipy 语义: 标量仅为下界; (min, max) 为区间。
        h_min, h_max = (
            height if isinstance(height, (tuple, list)) else (height, np.inf)
        )
        keep = (S[idx] >= h_min) & (S[idx] <= h_max)
        idx = idx[keep]

    threshold = kwargs.get("threshold")
    if threshold is not None:
        t_min, t_max = (
            threshold if isinstance(threshold, (tuple, list)) else (threshold, threshold)
        )
        keep = (S[idx] - S[idx - 1] >= t_min) & (S[idx] - S[idx + 1] >= t_max)
        idx = idx[keep]

    distance = kwargs.get("distance")
    if distance is not None:
        if distance < 1:
            raise ValueError(f"distance must be >= 1, got {distance!r}")
        # 贪心: 按峰高降序保留 (scipy 语义: 距更高峰不足 distance 者被移除)。
        # 有序列表 + bisect 判邻, O(P log P)。
        order = np.argsort(S[idx])[::-1]
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


def _detect_numpy(S: np.ndarray) -> np.ndarray:
    """向量化基础检测 (numpy 后端): S[i-1] < S[i] >= S[i+1]。"""
    mask = (S[1:-1] > S[:-2]) & (S[1:-1] >= S[2:])
    return np.flatnonzero(mask) + 1


def _detect_kernel(S: np.ndarray) -> np.ndarray:
    """基础检测内核 (numba 可编译): S[i-1] < S[i] >= S[i+1]。"""
    n = S.shape[0]
    out = np.empty(n, dtype=np.int64)
    count = 0
    for i in range(1, n - 1):
        if S[i - 1] < S[i] and S[i] >= S[i + 1]:
            out[count] = i
            count += 1
    return out[:count]


@functools.lru_cache(maxsize=None)
def _get_compiled(numba):
    """按需 njit 编译检测内核 (每进程最多编译一次)。"""
    return numba.njit(nogil=True)(_detect_kernel)


def _peaks_scipy(S: np.ndarray, kwargs: dict):
    ss = cache.import_module(
        "scipy.signal",
        description="scipy.signal: find_peaks (供 Utils.Peaks 的 scipy 后端使用)",
    )
    return ss.find_peaks(S, **kwargs)


def _peaks_numpy(S: np.ndarray, kwargs: dict):
    idx = _detect_numpy(S)
    idx = _apply_filters(idx, S, kwargs)
    return idx, {"peak_heights": S[idx]}


def _peaks_numba(S: np.ndarray, kwargs: dict):
    numba = cache.import_module(
        "numba",
        description="numba: JIT 编译 (供 Utils.Peaks 的 numba 后端使用)",
    )
    compiled = _get_compiled(numba)
    idx = compiled(S)
    idx = _apply_filters(idx, S, kwargs)
    return idx, {"peak_heights": S[idx]}


def find_peaks(
    S,
    mod: Literal["scipy", "numpy", "numba"] = "scipy",
    **kwargs,
) -> tuple[np.ndarray, dict]:
    """
    Find peaks in a 1-D signal.

    Parameters
    ----------
    S : array-like
        1-D real numeric signal (length >= 3; shorter inputs yield no peaks).
    mod : Literal["scipy", "numpy", "numba"]
        Backend:
        ``"scipy"`` (default) — :func:`scipy.signal.find_peaks` (全参数支持,
        含 plateau 规则);
        ``"numpy"`` — 纯 numpy 向量化实现;
        ``"numba"`` — numpy 内核经 ``njit`` 编译的可选加速 (需安装 numba)。
    **kwargs
        过滤参数 (scipy 语义): ``height`` / ``threshold`` / ``distance``。
        scipy 后端全部透传 (另支持 ``prominence`` / ``width`` / ``wlen`` /
        ``rel_height`` / ``plateau_size``); numpy / numba 后端仅支持上述
        三个, 其余抛 ``NotImplementedError``。

    Returns
    -------
    (indices, properties)
        ``indices`` : np.ndarray (int64)
            Peak positions.
        ``properties`` : dict
            ``{"peak_heights": S[indices]}`` (numpy / numba) 或 scipy 的
            完整属性字典 (scipy)。

    Raises
    ------
    ValueError
        非 1-D / 非实数数值输入, 或未知 ``mod``。
    ImportError
        scipy / numba 后端在对应库缺失时。
    NotImplementedError
        numpy / numba 后端收到不支持的过滤参数。
    """
    S, empty = _validate(S)
    if empty:
        return np.array([], dtype=np.int64), {}

    match mod:
        case "scipy":
            return _peaks_scipy(S, kwargs)
        case "numpy":
            return _peaks_numpy(S, kwargs)
        case "numba":
            return _peaks_numba(S, kwargs)
        case _:
            raise ValueError(f"Unknown mod: {mod!r}; expected 'scipy', 'numpy' or 'numba'")


if __name__ == "__main__":
    S = np.array([0, 1, 0, 2, 0, 3, 0], dtype=np.float64)
    idx, props = find_peaks(S, mod="numpy", height=1.5)
    print(idx, props)
