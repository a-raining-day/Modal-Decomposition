"""
Python version:  (must)
    3.10.11

Lib and Version:  (if None write None)
    numpy - 2.2.6
	numba - 0.64.0

Only accessed by:  (must)
    All(for modal decomposition more)

Modify:  (must)
    2026.3.25

Description: (if None write None)
    This file stored the helpful function which can be used in other modal decomposition methods.

Modify:
    Optimize the use of numba, accelerate the speed of import
"""

import numpy as np
from typing import Literal, Callable

def is_increasing_1(S) -> bool:
    diff = np.ediff1d(S)
    epsilon = 1e-8
    return np.all(diff > epsilon)

def give_is_increasing_2() -> Callable:
    from numba import njit

    @njit
    def is_increasing_2(S, rtol=1e-8, atol=1e-8):
        for i in range(len(S)-1):
            diff = S[i+1] - S[i]
            if diff <= atol + rtol * abs(S[i]):
                return False
        return True

    return is_increasing_2

def is_increasing(S, threshold=2, tolerance: Literal["high", "mid", "low"]="high") -> bool:
    def count_extrema(x):
        interior = x[1:-1]
        left = x[:-2]
        right = x[2:]
        maxima = (interior > left) & (interior > right)
        minima = (interior < left) & (interior < right)
        return np.sum(maxima) + np.sum(minima)

    if tolerance == "high":
        ans_3 = count_extrema(S) <= threshold
        return ans_3

    elif tolerance == "mid":
        diff = np.diff(S)

        sign_changes = np.diff(np.sign(diff))

        count = np.sum(np.abs(sign_changes) == 2)

        ans_1 = count <= threshold  # one way to check
        return ans_1