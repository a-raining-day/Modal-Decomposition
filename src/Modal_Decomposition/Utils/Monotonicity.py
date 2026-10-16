"""
Monotonicity checks.

Public API
----------
- ``give_monotony(arr, ...) -> Monotony`` : classify the input into one of
  the ``Monotony`` enum members.
- ``monotonic(arr, ...) -> bool`` : return whether the input satisfies a
  given monotonicity constraint.

Design notes
------------
- Validation happens **only** at the top of the two public functions. Every
  helper below assumes a valid 1-d real numeric ndarray and performs no
  checks of its own.
- ``Monotony.Equal`` is the internal classification for a constant
  (all-equal) chunk or boundary: in non-strict mode a constant sequence is
  monotonic in either direction; in strict mode it is not strictly monotonic.
- The input dtype is preserved: adjacent values are compared directly, so no
  ``np.diff`` is materialized and no float64 upcast is introduced.
- ``safe`` controls how much validation runs (see the public docstrings).
  ``0``/``"safe"`` trusts the caller; ``1``/``"suspicious"`` reshapes and
  checks dimensionality; ``2``/``"unknown"`` additionally checks numeric
  kind and finiteness.
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
# Public API
# ---------------------------------------------------------------------------

def give_monotony(
        arr,
        safe: SAFE_ARGS | SAFE_MODS = "unknown",
        chunk_size: int | None = None,
        chunk_mod: CHUNK_MOD_ARGS | None = None,
) -> Monotony:
    """
    Classify a 1-d array into a ``Monotony`` member.

    Parameters
    ----------
    arr : np.ndarray
        1-d input array.
    safe : Literal[0, 1, 2]
        Validation level (see ``monotonic``).
    chunk_size : int, optional
        Threshold above which the chunked path is used. Defaults to the
        current memory policy's chunk size.
    chunk_mod : optional
        Chunking strategy passed through to ``Chunk.chunk``.

    Returns
    -------
    Monotony
        One of ``Equal``, ``MixedIncreasing``, ``MixedDecreasing``,
        ``StrictIncreasing``, ``StrictDecreasing`` or ``No``.

    Raises
    ------
    ValueError
        On wrong dimensionality, invalid ``safe``, or (``safe=2`` only)
        non-finite / non-numeric input.

    Notes
    -----
    A constant (all-equal) sequence is reported as ``Equal`` regardless of
    direction; the caller decides whether that satisfies their own
    monotonicity definition.
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
    Check whether a 1-d array is monotonic.

    Parameters
    ----------
    arr : np.ndarray
        1-d input array.
    strict : bool
        ``True`` for strict monotonicity.
    mod : {"increasing", "decreasing", "monotonic"}
        Direction constraint.
        - ``"increasing"`` : non-decreasing (or strictly increasing).
        - ``"decreasing"`` : non-increasing (or strictly decreasing).
        - ``"monotonic"``  : monotonic in at least one direction.
          For ``strict=True`` the array must be strictly increasing or
          strictly decreasing (a constant array is *not* strictly monotonic).
    safe : Literal[0, 1, 2]
        Validation level.
        ``0`` : trust the caller -- ``arr`` must already be a valid 1-d real
        numeric ndarray.
        ``1`` : convert and check dimensionality only.
        ``2`` : additionally check numeric kind and finiteness.
    chunk_size, chunk_mod :
        Chunking knobs, as in ``give_monotony``.

    Returns
    -------
    bool
        ``True`` if the condition holds, ``False`` otherwise.

    Raises
    ------
    ValueError
        On invalid ``safe`` / ``mod`` / ``chunk_size``, wrong dimensionality,
        or (``safe=2`` only) non-finite / non-numeric input.
    """
    safe = Check.validate_safe(safe)
    mod = Check.validate_mod(mod, MONOTONY_MODS)
    arr = Check.to_array(arr, safe=safe)
    return _monotonic_bool(arr, strict, mod, chunk_size, chunk_mod)


# ---------------------------------------------------------------------------
# Internal helpers -- no validation; assume a valid 1-d array
# ---------------------------------------------------------------------------

def _commutative_semigroup(a: Monotony, b: Monotony) -> Monotony:
    """Merge two ``Monotony`` classifications."""
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
    if mod == "monotonic":
        m = _give_monotony(arr, chunk_size, chunk_mod)
        if m is Monotony.No:
            return False
        if strict:
            return m is Monotony.StrictIncreasing or m is Monotony.StrictDecreasing
        return True

    # Direction is known in advance: skip the full classification.
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
    Classify a 1-d sequence via adjacent comparisons (chunk.size >= 2 is
    assumed, but size < 2 is handled gracefully by returning ``Equal``).

    Two comparison passes produce two boolean masks; counts then decide the
    classification. This avoids the original five ``np.all`` scans and any
    per-boundary 2-element array allocation.
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
    """Monotony contribution of the boundary pair (a, b)."""
    if b > a:
        return Monotony.StrictIncreasing
    if b < a:
        return Monotony.StrictDecreasing
    return Monotony.Equal


def _direction_ok(chunk: np.ndarray, strict: bool, mod: str) -> bool:
    """Whether a single chunk (chunk.size >= 2) satisfies the direction."""
    if mod == "increasing":
        if strict:
            return bool(np.all(chunk[1:] > chunk[:-1]))
        return bool(np.all(chunk[1:] >= chunk[:-1]))
    # mod == "decreasing"
    if strict:
        return bool(np.all(chunk[1:] < chunk[:-1]))
    return bool(np.all(chunk[1:] <= chunk[:-1]))


def _pair_ok(a, b, strict: bool, mod: str) -> bool:
    """Whether the boundary pair (a, b) preserves the direction."""
    if mod == "increasing":
        return a < b if strict else a <= b
    return a > b if strict else a >= b