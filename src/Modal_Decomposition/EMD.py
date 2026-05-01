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
    2026.5.1  - Change the position of import PyEMD.
"""

import numpy as np
from typing import Tuple, Union
from .Utils import Check_Time_and_Signal

def emd(S: Union[list, np.ndarray], T: Union[list, np.ndarray]=None, spline_kind: str = "cubic", nbsym: int = 2, max_imf=-1, verbose: bool=False)\
        -> Tuple[np.ndarray, np.ndarray, None]:
    """
    EMD: Empirical Mode Decomposition

    :param S: Signal (1-dim)
    :param T: Time axis (1-dim)
    :param spline_kind: the kind of spline. default cubic.
    :param nbsym:
    :param max_imf: the max num of IMFs
    :param verbose: True will print info, else no.
    :return: IMFs (n_IMFs, N), Res (N,), None
    """
    from PyEMD import EMD

    S, T, N = Check_Time_and_Signal(S, T, verbose)

    EMD_cls = EMD(spline_kind, nbsym)

    IMFs = EMD_cls.emd(S, T, max_imf=max_imf)

    Res = IMFs[-1, :]
    IMFs = IMFs[:-1, :]

    if IMFs.ndim == 1:
        IMFs = IMFs.reshape(1, -1)

    elif IMFs.ndim == 0:
        IMFs = np.zeros((1, Res.shape[1]))

    return IMFs, Res, None