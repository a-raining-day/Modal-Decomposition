"""
Python version:  (must)
    3.10.11

Lib and Version:  (if None write None)
    ewtpy - 0.2
	numpy - 2.2.6

Only accessed by:  (must)
    Only __init__.py

Description: (if None write None)
    Realize the EWT

Modify:  (must)
    2026.3.25 - Create.
    2026.5.1  - Use dict to store other informations.
"""

from typing import Any, Union
import numpy as np
from .Utils import Check_Time_and_Signal

def ewt \
(
    S: Union[list, np.ndarray],
    N: int = 5,
    log: int = 0,
    detect: str = "locmax",
    completion: int = 0,
    reg: str = 'average',
    lengthFilter: int = 10,
    sigmaFilter: int = 5,
) -> Union[np.ndarray, np.ndarray, dict[str, Any]]:
    """
    EWT: Empirical Wavelet Transform

    :param S: Signal (1-dim)
    :param N:
    :param log:
    :param detect:
    :param completion:
    :param reg:
    :param lengthFilter:
    :param sigmaFilter:
    :return: IMFs(N, len(S)), Res(N,), Info(dict)
    """
    from ewtpy import EWT1D

    S, _, _ = Check_Time_and_Signal(S)

    ewt, mfb, boundaries = EWT1D(S, N, log, detect, completion, reg, lengthFilter, sigmaFilter)
    ewt: np.ndarray = ewt.T
    mfb = mfb.T

    Info = \
    {
        "MFB": mfb,
        "Boundaries": boundaries
    }

    return ewt[:-1, :], ewt[-1, :], Info