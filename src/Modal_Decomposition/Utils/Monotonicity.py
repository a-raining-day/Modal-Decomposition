"""
单调性判定。

公开 API
--------
- ``give_monotony(arr, ...) -> Monotony``: 把输入归类到 ``Monotony`` 的某个成员。
- ``monotonic(arr, ...) -> bool``: 判断输入是否满足指定的单调性约束。

约定
----
- 校验**只**发生在两个公开函数的开头。其下的辅助函数一律假定入参是合法的一维
  实数值 ``ndarray``, 自身不做任何检查。
- ``Monotony.Equal`` 是常数 (全等) 块或边界的分类: 非严格模式下常数序列对两个
  方向都单调; 严格模式下它不严格单调。
- 保留输入的 dtype: 相邻元素直接比较, 不物化 ``np.diff``, 也不引入 float64 上转。
- ``safe`` 控制校验强度 (见公开函数 docstring): ``0``/``"safe"`` 信任调用方;
  ``1``/``"suspicious"`` 另做形状与维度检查; ``2``/``"unknown"`` 再加数值类型与
  有限性检查。
"""

import numpy as np
from enum import Enum
from typing import Literal

from .Check import Check
from ..Define import SAFE_ARGS, SAFE_MODS, DEFAULT_NUMPY_TYPE
from ..Base import Chunk, CHUNK_MOD_ARGS


__all__ = ["Monotony", "give_monotony", "monotonic", "MONOTONY_MODS", "MONOTONY_MODS_ARGS"]


MONOTONY_MODS_ARGS = Literal["increasing", "decreasing", "monotonic"]
class MONOTONY_MODS(Enum):
    Increasing = "increasing"
    Decreasing = "decreasing"
    Monotonic = "monotonic"


class Monotony(Enum):
    """
    单调性枚举，使用 3 位二进制编码。

    编码格式：
        bit2      : 严格性，1 = 严格，0 = 非严格
        bit1-bit0 : 方向，00 = 常数，01 = 递增，10 = 递减，11 = 无效/No

    各成员编码：
        Equal            = 0b000
        MixedIncreasing  = 0b001
        MixedDecreasing  = 0b010
        StrictIncreasing = 0b101
        StrictDecreasing = 0b110
        No               = 0b111

    合并规则（source 与 to 合并）：
        1. 方向按位或：d = (source & 0b011) | (to & 0b011)
             01 | 10 = 11 → 方向冲突，结果为 No
             其余情况 | 自动得到正确方向（Equal 为 00，起单位元作用）
        2. 严格性按位与：s = (source >> 2) & (to >> 2)
             因 Equal 的严格位为 0，任一方为 Equal 时 s 自动为 0
             双方均严格时 s = 1
        3. 结果 = (s << 2) | d
        4. No 的低两位为 11，与任何值或运算后仍为 11，自然落入 No 分支
    """
    Equal = 0b000
    MixedIncreasing = 0b001
    MixedDecreasing = 0b010
    StrictIncreasing = 0b101
    StrictDecreasing = 0b110
    No = 0b111


# ---------------------------------------------------------------------------
# 公开 API
# ---------------------------------------------------------------------------

def give_monotony(
        arr,
        safe: SAFE_ARGS | SAFE_MODS = "unknown",
        chunk_size: int | None = None,
        chunk_mod: CHUNK_MOD_ARGS | None = None,
) -> Monotony:
    """
    把一维数组归类到某个 ``Monotony`` 成员。

    Parameters
    ----------
    arr : np.ndarray
        一维输入数组。
    safe : Literal[0, 1, 2] | Literal["safe", "suspicious", "unknown"]
        校验强度 (见 ``monotonic``)。
    chunk_size : int, optional
        超过该长度走分块路径; 默认取当前内存策略算出的块长。
    chunk_mod : optional
        分块策略, 透传给 ``Chunk.chunk``。

    Returns
    -------
    Monotony
        ``Equal``、``MixedIncreasing``、``MixedDecreasing``、
        ``StrictIncreasing``、``StrictDecreasing`` 或 ``No``。

    Raises
    ------
    ValueError
        维度不对、``safe`` 非法, 或 (仅 ``safe=2``) 非有限 / 非数值输入。

    Notes
    -----
    常数 (全等) 序列无论方向都报 ``Equal``; 它是否满足调用方的单调性定义由调用方
    自行决定。
    """
    safe = Check.validate_safe(safe)
    arr = Check.to_array(arr, safe=safe)
    return _give_monotony(arr, chunk_size, chunk_mod)


def monotonic(
        arr,
        strict: bool = False,
        mod: MONOTONY_MODS_ARGS | MONOTONY_MODS = "monotonic",
        safe: SAFE_ARGS | SAFE_MODS = "unknown",
        chunk_size: int | None = None,
        chunk_mod: CHUNK_MOD_ARGS | None = None,
) -> bool:
    """
    判断一维数组是否单调。

    Parameters
    ----------
    arr : np.ndarray
        一维输入数组。
    strict : bool
        ``True`` 用严格单调。
    mod : {"increasing", "decreasing", "monotonic"}
        方向约束。
        - ``"increasing"``: 非降 (``strict=True`` 时为严格递增)。
        - ``"decreasing"``: 非增 (``strict=True`` 时为严格递减)。
        - ``"monotonic"``: 至少一个方向上单调。``strict=True`` 时要求严格递增或
          严格递减 (常数数组不算严格单调)。
    safe : Literal[0, 1, 2]
        校验强度。
        ``0``: 信任调用方 —— ``arr`` 必须已是合法的一维实数值 ``ndarray``。
        ``1``: 转成数组并只查维度。
        ``2``: 另查数值类型与有限性。
    chunk_size, chunk_mod:
        分块参数, 同 ``give_monotony``。

    Returns
    -------
    bool
        满足约束为 ``True``, 否则 ``False``。

    Raises
    ------
    ValueError
        ``safe`` / ``mod`` / ``chunk_size`` 非法、维度不对, 或 (仅 ``safe=2``)
        非有限 / 非数值输入。
    """
    safe = Check.validate_safe(safe)
    mod = Check.validate_mod(mod, MONOTONY_MODS)
    arr = Check.to_array(arr, safe=safe)
    return _monotonic_bool(arr, strict, mod, chunk_size, chunk_mod)


# ---------------------------------------------------------------------------
# 内部辅助: 不做校验, 假定入参是合法的一维数组
# ---------------------------------------------------------------------------

def _commutative_semigroup(a: Monotony, b: Monotony) -> Monotony:
    """合并两个 ``Monotony`` 分类。"""
    d = (a.value & 0b011) | (b.value & 0b011)
    if d == 0b11:
        return Monotony.No
    s = (a.value >> 2) & (b.value >> 2)
    return Monotony((s << 2) | d)


def _give_monotony(arr, chunk_size, chunk_mod) -> Monotony:
    n = arr.size
    if n < 2:
        return Monotony.Equal

    if chunk_size is None:
        chunk_size = Chunk.chunk_size(arr.nbytes)

    if n <= chunk_size:
        return _classify(arr)

    acc: Monotony | None = None
    last = None
    for chunk in Chunk.chunk(
            arr,
            chunk_size=chunk_size,
            mod=chunk_mod if chunk_mod is not None else "memory",
    ):
        m = _classify(chunk)
        if m is Monotony.No:
            return Monotony.No

        if acc is not None:
            m = _commutative_semigroup(acc, m)
            if m is Monotony.No:
                return Monotony.No

        if last is not None:
            delta = _pair_monotony(last, chunk[0])
            if _commutative_semigroup(delta, m) is Monotony.No:
                return Monotony.No

        acc = m
        last = chunk[-1]

    return acc if acc is not None else Monotony.Equal


def _monotonic_bool(arr, strict: bool, mod: str, chunk_size, chunk_mod) -> bool:
    if mod == MONOTONY_MODS.Monotonic:
        m = _give_monotony(arr, chunk_size, chunk_mod)
        if m is Monotony.No:
            return False
        if strict:
            return m is Monotony.StrictIncreasing or m is Monotony.StrictDecreasing
        return True

    # 方向已给定: 只做方向判定, 省掉完整分类。
    n = arr.size
    if n < 2:
        return True

    if chunk_size is None:
        chunk_size = Chunk.chunk_size(arr.nbytes)

    if n <= chunk_size:
        return _direction_ok(arr, strict, mod)

    last = None
    for chunk in Chunk.chunk(
            arr,
            chunk_size=chunk_size,
            mod=chunk_mod if chunk_mod is not None else "memory",
    ):
        if chunk.size >= 2 and not _direction_ok(chunk, strict, mod):
            return False
        if last is not None and not _pair_ok(last, chunk[0], strict, mod):
            return False
        last = chunk[-1]

    return True


def _classify(chunk: np.ndarray) -> Monotony:
    """
    用相邻比较给一维序列分类 (假定 ``chunk.size >= 2``; 不足 2 时返回 ``Equal``)。

    两趟比较得到两个布尔掩码, 再由计数决定分类: 避免多次 ``np.all`` 全量扫描,
    也不为每个边界单独分配 2 元素数组。
    """
    n = chunk.size
    if n < 2:
        return Monotony.Equal

    gt = chunk[1:] > chunk[:-1]
    lt = chunk[1:] < chunk[:-1]
    m = n - 1
    n_gt = int(np.count_nonzero(gt))
    n_lt = int(np.count_nonzero(lt))

    if n_gt == m:
        return Monotony.StrictIncreasing
    if n_lt == m:
        return Monotony.StrictDecreasing
    if n_gt == 0:
        return Monotony.MixedDecreasing if n_lt > 0 else Monotony.Equal
    if n_lt == 0:
        return Monotony.MixedIncreasing
    return Monotony.No


def _pair_monotony(a, b) -> Monotony:
    """边界对 (a, b) 给出的 ``Monotony`` 贡献。"""
    if b > a:
        return Monotony.StrictIncreasing
    if b < a:
        return Monotony.StrictDecreasing
    return Monotony.Equal


def _direction_ok(chunk: np.ndarray, strict: bool, mod: str) -> bool:
    """单块 (``chunk.size >= 2``) 是否满足给定方向。"""
    if mod == MONOTONY_MODS.Increasing:
        if strict:
            return bool(np.all(chunk[1:] > chunk[:-1]))
        return bool(np.all(chunk[1:] >= chunk[:-1]))
    # 递减
    if strict:
        return bool(np.all(chunk[1:] < chunk[:-1]))
    return bool(np.all(chunk[1:] <= chunk[:-1]))


def _pair_ok(a, b, strict: bool, mod: str) -> bool:
    """边界对 (a, b) 是否保持给定方向。"""
    if mod == MONOTONY_MODS.Increasing:
        return a < b if strict else a <= b
    return a > b if strict else a >= b