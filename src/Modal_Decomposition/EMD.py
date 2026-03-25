"""
Python version:  (must)
    3.10.11

Lib and Version:  (if None write None)
    EMD-signal - 1.9.0
	numpy - 2.2.6

Only accessed by:  (must)
    Only __init__.py

Modify:  (must)
    2026.3.25

Description: (if None write None)
    Realize the EMD
"""

from PyEMD import EMD
import numpy as np
from .COLOR.colorful_print import printc
from typing import Tuple

EMD_cls = EMD

def emd(S, T=None, spline_kind: str = "cubic", nbsym: int = 2, max_imf=-1, fs=None) -> Tuple[np.ndarray, np.ndarray]:
    """
    :param S:
    :param T:
    :param spline_kind:
    :param nbsym:
    :param max_imf:
    :param fs:
    :return: IMFs (2-dim), Res: None
    """
    if not isinstance(S, np.ndarray):
        S = np.array(S)

    N = len(S)

    if T is None:
        if fs is not None:
            dt = 1.0 / fs  # smaple for time axis
            T = np.arange(N) * dt
        else:
            T = np.arange(N)  # default fs = 1
            printc(f"Warn: T is None，default T = [0, 1, 2, ..., {N - 1}]", color="red")

    EMD_cls = EMD(spline_kind, nbsym)

    IMFs = EMD_cls.emd(S, T, max_imf=max_imf)

    return IMFs[:-1, :], IMFs[-1, :]