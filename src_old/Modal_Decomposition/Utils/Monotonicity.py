"""
First: Calculate the length of the array.
Second: Choose the method: Chunk or not Chunk or Streaming.
"""

import numpy as np
from typing import Iterable, Literal
from .Check import CheckDim
from ..Base.ConstDefine import Monotony

__all__ = ["monotonic"]

CheckOneDim = CheckDim(1)

def monotonic(arr: np.ndarray, strict: bool = False, chunk_size: int = 1048576, mod: Literal["increasing", "decreasing", "monotonic"]="monotonic") -> bool:
    """
    Short sequence use python check, Long sequence use numpy.
    :param arr:
    :param strict:
    :param chunk_size: 1024 * 1024
    :return:
    """

    dim_sure, arr = CheckOneDim(arr)

    if not np.all(np.isfinite(arr)):
        raise ValueError("NaN or Inf found")

    # check dim
    if not dim_sure:
        raise ValueError("The dim of arr must be 1-dim!")

    # judge length
    arr_length = arr.size

    if arr_length == 0:
        raise ValueError("The length of arr shouldn't be 0")

    if arr_length == 1:
        return True

    match mod:
        case "monotonic":
            if arr_length <= chunk_size:  # TODO: should test when the short sequence, python version's speed and numpy's speed.
                diff = np.diff(arr)
                if strict:
                    if np.all(diff < 0) or np.all(diff > 0):
                        return True
                    else:
                        return False

                else:
                    if np.all(diff <= 0) or np.all(diff >= 0):
                        return True
                    else:
                        return False

            else:
                if chunk_size * 1.5 <= arr_length < chunk_size * 2:
                    chunk_size = arr_length // 2

                bound_monotony = []
                for i in range(chunk_size, arr_length, chunk_size):
                    bound_monotony.extend([arr[i - 1], arr[i]])

                bound_monotony_diff = np.diff(bound_monotony)
                if strict:
                    if np.all(bound_monotony_diff < 0):
                        monotony = Monotony.StrictDecreasing
                    elif np.all(bound_monotony_diff > 0):
                        monotony = Monotony.StrictIncreasing
                    else:
                        return False

                else:
                    if np.all(bound_monotony_diff <= 0):
                        monotony = Monotony.Decreasing
                    elif np.all(bound_monotony_diff >= 0):
                        monotony = Monotony.Increasing
                    else:
                        return False

                def Chunk() -> Iterable[np.ndarray]:
                    for i in range(0, arr_length, chunk_size):
                        chunk_end = min(i + chunk_size, arr_length)
                        yield arr[i:chunk_end]  # create chunk view

                for chunk in Chunk():
                    match monotony:
                        case Monotony.Increasing:
                            if not np.all(np.diff(chunk) >= 0):
                                return False

                        case Monotony.Decreasing:
                            if not np.all(np.diff(chunk) <= 0):
                                return False

                        case Monotony.StrictDecreasing:
                            if not np.all(np.diff(chunk) < 0):
                                return False

                        case Monotony.StrictIncreasing:
                            if not np.all(np.diff(chunk) > 0):
                                return False
                return True

        case "decreasing":
            return _monotonic(arr, strict, chunk_size, "decreasing")

        case "increasing":
            return _monotonic(arr, strict, chunk_size, "increasing")

def _monotonic(arr: np.ndarray, strict: bool, chunk_size: int, mod: Literal["increasing", "decreasing"]) -> bool:
    arr_length = arr.size

    if arr_length <= chunk_size:  # below the safe line
        if mod == "decreasing":
            if strict:
                return np.all(np.diff(arr) < 0)  # strictly increasing

            else:
                return np.all(np.diff(arr) <= 0)  # easily increasing
        else:
            if strict:
                return np.all(np.diff(arr) > 0)  # strictly increasing

            else:
                return np.all(np.diff(arr) >= 0)  # easily increasing

    else:  # the length is too big
        def Chunk() -> Iterable[np.ndarray]:
            for i in range(0, arr_length, chunk_size):
                chunk_end = min(i + chunk_size, arr_length)
                yield arr[i:chunk_end]  # create chunk view

        last = None
        for chunk in Chunk():
            # two methods: 1: use 'np.diff' calculate the entire chunk | 2: use 'for' calculate the chunk step by step.
            chunk_diff = np.diff(chunk)

            if mod == "decreasing":
                if strict:
                    sub_chunk_monotony = np.all(chunk_diff < 0)

                else:
                    sub_chunk_monotony = np.all(chunk_diff <= 0)

            else:
                if strict:
                    sub_chunk_monotony = np.all(chunk_diff > 0)

                else:
                    sub_chunk_monotony = np.all(chunk_diff >= 0)

            if not sub_chunk_monotony:
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