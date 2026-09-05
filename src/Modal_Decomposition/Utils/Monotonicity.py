"""
Monotonicity checks.

First: calculate the length of the array.
Second: choose the method: chunked or single-shot.

Notes
-----
- ``Monotony.Equal`` is an internal classification for a constant (all-equal)
  chunk or boundary: in non-strict mode a constant sequence is monotonic (it
  is both non-decreasing and non-increasing, so it satisfies every direction),
  while in strict mode it is not strictly monotonic. The public functions
  still return ``bool``.
- The input dtype is preserved (no float64 upcast): adjacent values are
  compared directly instead of materializing ``np.diff``, which keeps the
  temporary memory at one byte per element and avoids the slow float16 /
  float32 conversion copy.
"""

from enum import Enum

import numpy as np
from typing import Iterable, Literal

from .Check import require_ndim

__all__ = ["Monotony", "monotonic", "is_monotonic"]


class Monotony(Enum):
    Monotonic = 0

    Increasing = 1
    Decreasing = 2

    StrictIncreasing = 3
    StrictDecreasing = 4

    Equal = 5


def _classify_chunk(chunk: np.ndarray, strict: bool) -> Monotony | None:
    """
    Classify a sequence (at least two elements) via adjacent comparisons.

    Parameters
    ----------
    chunk : np.ndarray
        1-d sequence with ``chunk.size >= 2``.
    strict : bool
        Whether strict monotonicity is required.

    Returns
    -------
    Monotony or None
        ``Equal`` for a constant sequence, ``Increasing`` / ``Decreasing``
        (or their strict counterparts) for a one-directional sequence, and
        ``None`` when the signs are mixed. Under ``strict`` any tie returns
        ``None``.

    Notes
    -----
    Comparisons are performed on adjacent values in the native dtype, so no
    difference array is materialized and no integer/float overflow can occur.
    After a one-directional check passes, ``chunk[-1] == chunk[0]`` detects a
    constant chunk in O(1).
    """
    if strict:
        if np.all(chunk[1:] > chunk[:-1]):
            return Monotony.StrictIncreasing

        if np.all(chunk[1:] < chunk[:-1]):
            return Monotony.StrictDecreasing

        return None

    if np.all(chunk[1:] >= chunk[:-1]):
        # Non-decreasing with equal ends is constant.
        return Monotony.Equal if chunk[-1] == chunk[0] else Monotony.Increasing

    if np.all(chunk[1:] <= chunk[:-1]):
        return Monotony.Equal if chunk[-1] == chunk[0] else Monotony.Decreasing

    return None


def _classify_seams(arr: np.ndarray, seam_idx: np.ndarray, strict: bool) -> Monotony | None:
    """
    Classify the sign pattern of the pairs crossing chunk seams.

    Parameters
    ----------
    arr : np.ndarray
        1-d array.
    seam_idx : np.ndarray
        Integer positions ``i`` of the pairs ``(arr[i-1], arr[i])``.
    strict : bool
        Whether strict monotonicity is required.

    Returns
    -------
    Monotony or None
        The common sign pattern of the seam pairs, ``Equal`` when every pair
        ties, or ``None`` when the signs are mixed.
    """
    right = arr[seam_idx]
    left = arr[seam_idx - 1]

    if strict:
        if np.all(right > left):
            return Monotony.StrictIncreasing

        if np.all(right < left):
            return Monotony.StrictDecreasing

        return None

    if np.all(right == left):
        return Monotony.Equal

    if np.all(right >= left):
        return Monotony.Increasing

    if np.all(right <= left):
        return Monotony.Decreasing

    return None


def _single_shot_monotonic(arr: np.ndarray, strict: bool) -> bool:
    """Whole-array check for ``mod == "monotonic"`` when size is manageable."""
    monotony = _classify_chunk(arr, strict)

    if monotony is None:
        return False

    if strict:
        return monotony in (Monotony.StrictIncreasing, Monotony.StrictDecreasing)

    # Equal / Increasing / Decreasing are all monotonic in non-strict mode.
    return True


def _chunked_monotonic(arr: np.ndarray, strict: bool, chunk_size: int) -> bool:
    """Chunked check for ``mod == "monotonic"`` when ``arr.size > chunk_size``."""
    arr_length = arr.size

    # Pairs crossing every chunk seam, classified by sign pattern.
    seam_idx = np.arange(chunk_size, arr_length, chunk_size)
    bound_state = _classify_seams(arr, seam_idx, strict)
    if bound_state is None:
        return False

    def Chunk() -> Iterable[np.ndarray]:
        for i in range(0, arr_length, chunk_size):
            chunk_end = min(i + chunk_size, arr_length)
            yield arr[i:chunk_end]  # create chunk view

    if strict:
        for chunk in Chunk():
            if chunk.size < 2:
                continue  # single element: its seams were already checked
            if _classify_chunk(chunk, strict=True) is not bound_state:
                return False

        return True

    # Non-strict: Equal seams defer the direction decision to the chunks.
    monotony = bound_state  # Equal, Increasing or Decreasing
    for chunk in Chunk():
        if chunk.size < 2:
            continue

        sub_monotony = _classify_chunk(chunk, strict=False)
        if sub_monotony is None:
            return False  # mixed signs inside a chunk -> not monotone

        if monotony is Monotony.Equal:
            if sub_monotony is not Monotony.Equal:
                monotony = sub_monotony  # first one-directional chunk decides
        elif sub_monotony is not Monotony.Equal and sub_monotony is not monotony:
            return False  # direction flip between chunks -> not monotone

    # Still Equal: every value is equal, which is monotonic in non-strict mode.
    return True


def monotonic(arr: np.ndarray, strict: bool = False, chunk_size: int = 1048576, mod: Literal["increasing", "decreasing", "monotonic"] = "monotonic") -> bool:
    """
    Check whether a 1-d array is monotonic.

    Parameters
    ----------
    arr : np.ndarray
        1-d input array.
    strict : bool
        True for strict monotonicity.
    chunk_size : int
        Threshold above which the chunked path is used.
    mod : Literal["increasing", "decreasing", "monotonic"]
        Direction of monotonicity.

    Returns
    -------
    bool
        True if the condition holds.

    Raises
    ------
    ValueError
        On non-finite values, wrong dimensionality, empty input, a
        non-numeric dtype, or an invalid ``mod`` / ``chunk_size``.

    Notes
    -----
    A constant (all-equal) sequence is monotonic in non-strict mode (``Equal``
    satisfies every direction) and not strictly monotonic in strict mode.
    The input dtype is preserved: integers are compared exactly and floats
    (float16/32/64) are compared without an upcast copy.
    """
    arr = np.asarray(arr)

    if arr.dtype.kind not in "iufb":
        raise ValueError(
            f"monotonic: expected a real numeric array, got dtype {arr.dtype}"
        )

    if arr.ndim == 0:
        raise ValueError("Signal must have at least 1 dimension, got 0-d array")

    if arr.size == 1:
        # A single value is trivially monotonic in every direction. Handle it
        # before the squeeze turns a one-element array into 0-d.
        if arr.dtype.kind == "f" and not np.all(np.isfinite(arr)):
            raise ValueError("NaN or Inf found")
        return True

    arr = arr.squeeze()
    require_ndim(arr, {1}, "monotonic")

    if arr.dtype.kind == "f" and not np.all(np.isfinite(arr)):
        raise ValueError("NaN or Inf found")

    arr_length = arr.size

    if arr_length == 0:
        raise ValueError("The length of arr shouldn't be 0")

    if chunk_size < 1:
        raise ValueError(f"chunk_size must be a positive integer, got {chunk_size}")

    match mod:
        case "monotonic":
            if arr_length <= chunk_size:
                return _single_shot_monotonic(arr, strict)
            return _chunked_monotonic(arr, strict, chunk_size)

        case "decreasing":
            return _monotonic(arr, strict, chunk_size, "decreasing")

        case "increasing":
            return _monotonic(arr, strict, chunk_size, "increasing")

        case _:
            raise ValueError(f"Invalid mod: {mod}")


def is_monotonic(arr: np.ndarray, strict: bool = False, chunk_size: int = 1048576) -> bool:
    """
    Check whether a 1-d array is monotonic in either direction.

    Parameters
    ----------
    arr : np.ndarray
        1-d input array.
    strict : bool
        True for strict monotonicity.
    chunk_size : int
        Threshold above which the chunked path is used.

    Returns
    -------
    bool
        True if non-increasing or non-decreasing (a constant sequence
        qualifies as both in non-strict mode).
    """
    return monotonic(arr, strict=strict, chunk_size=chunk_size, mod="monotonic")


def _monotonic(arr: np.ndarray, strict: bool, chunk_size: int, mod: Literal["increasing", "decreasing"]) -> bool:
    arr_length = arr.size

    if arr_length <= chunk_size:  # below the safe line
        monotony = _classify_chunk(arr, strict)
        if monotony is None:
            return False

        if strict:
            return (
                monotony is Monotony.StrictIncreasing
                if mod == "increasing"
                else monotony is Monotony.StrictDecreasing
            )

        # Non-strict: a constant sequence (Equal) satisfies every direction.
        return (
            monotony in (Monotony.Increasing, Monotony.Equal)
            if mod == "increasing"
            else monotony in (Monotony.Decreasing, Monotony.Equal)
        )

    else:  # the length is too big
        def Chunk() -> Iterable[np.ndarray]:
            for i in range(0, arr_length, chunk_size):
                chunk_end = min(i + chunk_size, arr_length)
                yield arr[i:chunk_end]  # create chunk view

        last = None
        for chunk in Chunk():
            if chunk.size >= 2:
                monotony = _classify_chunk(chunk, strict)
                if monotony is None:
                    return False

                if strict:
                    ok = (
                        monotony is Monotony.StrictIncreasing
                        if mod == "increasing"
                        else monotony is Monotony.StrictDecreasing
                    )
                else:
                    ok = (
                        monotony in (Monotony.Increasing, Monotony.Equal)
                        if mod == "increasing"
                        else monotony in (Monotony.Decreasing, Monotony.Equal)
                    )

                if not ok:
                    return False

            if last is None:
                last = chunk[-1]

            else:
                if mod == "decreasing":
                    if strict:
                        if chunk[0] >= last:
                            return False

                    else:
                        if chunk[0] > last:
                            return False

                else:
                    if strict:
                        if chunk[0] <= last:
                            return False

                    else:
                        if chunk[0] < last:
                            return False

                last = chunk[-1]

        return True
