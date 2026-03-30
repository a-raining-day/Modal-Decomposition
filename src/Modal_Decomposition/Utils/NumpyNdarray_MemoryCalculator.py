"""
Show the memory of np.ndarray.

1: Only of item(even it's view)
2: For all, check the base memory recursively.
"""

import numpy as np
from typing import Union
import warnings


def view_memory(arr: np.ndarray) -> int:
    """calculate the view's memory"""
    if not isinstance(arr, np.ndarray):
        raise TypeError(f"Except np.ndarray, not {type(arr)}")

    return arr.nbytes

def root_memory(arr: np.ndarray, max_depth: int = 10) -> int:
    """
    calculate the view's root's memory

    :param max_depth: the max depth of the base chain. Default 10.
    """
    if not isinstance(arr, np.ndarray):
        raise TypeError(f"Except np.ndarray, not {type(arr)}")

    while_count = 0
    while arr.base is not None:
        arr = arr.base
        while_count += 1
        if while_count >= max_depth + 1:
            # warnings.warn(f"The time is too long when getting the view's root memory, get {current.nbytes} now.")
            warnings.warn("it's cost much time when finding the view's root's memory. return current nbytes.")
            return arr.nbytes

    return arr.nbytes