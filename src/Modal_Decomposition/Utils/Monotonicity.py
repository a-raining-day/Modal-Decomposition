"""
First: Calculate the length of the array.
Second: Choose the method: Chunk or not Chunk or Streaming.
"""

import numpy as np

def monotonic_increasing(arr: np.ndarray, strict: bool = False, chunk_size: int = 1000000) -> bool:
    """
    Judge the arr whether monotonic increasing.

    Attention: If length of arr is 1, return True. If 0, Raise Error. Nan and Inf couldn't be in arr.

    :param arr: the numpy.ndarray.
    :param strict: dose include equal? Default include. True -> '>'(strict monotonic) | False -> '>='(easy monotonic)
    :param chunk_size: the size of the chunk. when the length of arr is too long, will use it.
    :return:
    """

    if not isinstance(arr, np.ndarray):
        try:
            arr = np.asarray(arr)

        except:
            raise TypeError(f"The type of the arr must be np.ndarray, not {type(arr)}.")

    if not np.all(np.isfinite(arr)):
        raise ValueError("Nan or Inf in arr")

    if arr.ndim == 0:
        raise ValueError("The dim of arr at least 1-dim")

    elif arr.ndim != 1:  # may be (n,) -> 1-dim or (1, n) -> 2-dim
        if 1 not in arr.shape:  # now it's should be (1, n), and it must be (1, n)
            raise ValueError(f"The dim of the arr must be 1. not {arr.ndim}")

        else:
            arr = arr.reshape(-1)

    arr_length = arr.size

    if arr_length == 0:
        raise ValueError("The length of arr shouldn't be 0")

    if arr_length == 1:
        return True

    if arr_length <= chunk_size:  # below the safe line
        if strict:
            return np.all(np.diff(arr) > 0)  # strictly increasing

        else:
            return np.all(np.diff(arr) >= 0)  # easily increasing

    else:  # the length is too big
        def Chunk() -> iter:
            for i in range(0, arr_length, chunk_size):
                chunk_end = min(i + chunk_size, arr_length)
                yield arr[i:chunk_end]  # create chunk view

        last = None
        for chunk in Chunk():
            # two methods: 1: use 'np.diff' calculate the entire chunk | 2: use 'for' calculate the chunk step by step.
            chunk_diff = np.diff(chunk)

            if strict:
                sub_chunk_increasing = np.all(chunk_diff > 0)

            else:
                sub_chunk_increasing = np.all(chunk_diff >= 0)

            if not sub_chunk_increasing:
                return False

            if last is None:
                last = chunk[-1]

            else:
                if strict:
                    if chunk[0] <= last:
                        return False

                else:
                    if chunk[0] < last:
                        return False

                last = chunk[-1]

        return True


def monotonic_decreasing(arr: np.ndarray, strict: bool = False, chunk_size: int = 1000000) -> bool:
    """
    Judge the arr whether monotonic decreasing.

    Attention: If length of arr is 1, return True. If 0, Raise Error. Nan and Inf couldn't be in arr.

    :param arr: the numpy.ndarray.
    :param strict: dose include equal? Default include. True -> '<'(strict monotonic) | False -> '<='(easy monotonic)
    :param chunk_size: the size of the chunk. when the length of arr is too long, will use it.
    :return:
    """

    if not isinstance(arr, np.ndarray):
        try:
            arr = np.asarray(arr)

        except:
            raise TypeError(f"The type of the arr must be np.ndarray, not {type(arr)}.")

    if not np.all(np.isfinite(arr)):
        raise ValueError("Nan or Inf in arr")

    if arr.ndim == 0:
        raise ValueError("The dim of arr at least 1-dim")

    elif arr.ndim != 1:  # may be (n,) -> 1-dim or (1, n) -> 2-dim
        if 1 not in arr.shape:  # now it's should be (1, n), and it must be (1, n)
            raise ValueError(f"The dim of the arr must be 1. not {arr.ndim}")

        else:
            arr = arr.reshape(-1)

    arr_length = arr.size

    if arr_length == 0:
        raise ValueError("The length of arr shouldn't be 0")

    if arr_length == 1:
        return True

    if arr_length <= chunk_size:  # below the safe line
        if strict:
            return np.all(np.diff(arr) < 0)  # strictly increasing

        else:
            return np.all(np.diff(arr) <= 0)  # easily increasing

    else:  # the length is too big
        def Chunk() -> iter:
            for i in range(0, arr_length, chunk_size):
                chunk_end = min(i + chunk_size, arr_length)
                yield arr[i:chunk_end]  # create chunk view

        last = None
        for chunk in Chunk():
            # two methods: 1: use 'np.diff' calculate the entire chunk | 2: use 'for' calculate the chunk step by step.
            chunk_diff = np.diff(chunk)

            if strict:
                sub_chunk_increasing = np.all(chunk_diff < 0)

            else:
                sub_chunk_increasing = np.all(chunk_diff <= 0)

            if not sub_chunk_increasing:
                return False

            if last is None:
                last = chunk[-1]

            else:
                if strict:
                    if chunk[0] >= last:
                        return False

                else:
                    if chunk[0] > last:
                        return False

                last = chunk[-1]

        return True