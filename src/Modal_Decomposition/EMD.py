"""
Python version:  (must)
    3.10.11

Lib and Version:  (if None write None)
    EMD-S - 1.9.0
	numpy - 2.2.6

Only accessed by:  (must)
    Only __init__.py

Description: (if None write None)
    Realize the EMD

Modify:  (must)
    2026.3.25 - Create.
"""

from PyEMD import EMD
import numpy as np
from typing import Tuple, Union


def emd(S: Union[list, np.ndarray], T: Union[list, np.ndarray]=None, spline_kind: str = "cubic", nbsym: int = 2, max_imf=-1, verbose: bool=False)\
        -> Tuple[np.ndarray, np.ndarray]:
    """
    EMD: Empirical Mode Decomposition

    :param S: Signal (1-dim)
    :param T: Time axis (1-dim)
    :param spline_kind: the kind of spline. default cubic.
    :param nbsym:
    :param max_imf: the max num of IMFs
    :param verbose: True will print info, else no.
    :return: IMFs (2-dim), Res (1-dim)
    """
    if not isinstance(S, np.ndarray):
        S = np.array(S)

    if S.ndim == 0:
        raise ValueError("The dim of the S must be 1-dim, not 0")

    elif S.ndim > 1:
        if 1 in S.shape:
            S = S.reshape(-1)

        else:
            raise ValueError(f"The dim of S must be 1-dim, not {S.ndim}")

    N = len(S)

    if T is None:  # if T is None, default generate uniform T-axis.
        T = np.arange(N)  # default fs = 1
        if verbose:
            print(f"Warn: T is None，default T = [0, 1, 2, ..., {N - 1}]")

    else:
        if not isinstance(T, np.ndarray):
            T = np.array(T)

    EMD_cls = EMD(spline_kind, nbsym)

    IMFs = EMD_cls.emd(S, T, max_imf=max_imf)

    Res = IMFs[-1, :]
    IMFs = IMFs[:-1, :]

    if IMFs.ndim == 1:
        IMFs = IMFs.reshape(1, -1)

    elif IMFs.ndim == 0:
        IMFs = np.zeros((1, Res.shape[1]))

    return IMFs, Res