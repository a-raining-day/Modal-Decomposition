"""
Python version:
    3.10.11

Role:  (if None write None)
    Entrance of Utils

Lib and Version:  (if None write None)
    numpy - 2.2.6

Only accessed by:  (must)
    All

Description: (if None write None)
    As the entrance of the utils sub-lib.

Modify:  (must)
    2026.3.30 - Create.
    2026.7.2  - Change the realize of CheckDim(for check dimension)
"""

import numpy as np
from typing import Tuple
from warnings import warn

from .EnvironmentMemory import *
from .Monotonicity import *
from .Check import *


def Check_ST_and_Transform(S: list | np.ndarray, T: list | np.ndarray = None, dim: int = 1, REVERSE: bool = False, RAISE: bool = True) -> Tuple[bool, bool, np.ndarray, np.ndarray, int]:
    CheckDimension = CheckDim(dim, RAISE)

    DimSure, S = CheckDimension(S)
    if not DimSure:
        if RAISE:
            raise ValueError(f"The dim of S is not equal to {dim}!")

    N = S.size

    if T is None:
        T = np.arange(N)
        if RAISE:
            raise ValueError(f"Warn: T is None，default T = [0, 1, 2, ..., {N - 1}]")

    else:
        diff = np.diff(T)

        if 0 in diff:
            raise ValueError(f"There two value at the same time!")

        if not np.all(diff > 0):
            warn("T should be monotonic increasing! The Signal Values has been queued by Time-Axis.")

            idx = np.argsort(T)
            if REVERSE:
                idx = idx[::-1]

            S, T = S[idx], T[idx]

    uniform, S, T = Check_Time(S, T)

    return uniform, DimSure, S, T, N