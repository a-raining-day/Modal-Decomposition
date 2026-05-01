"""
Python version:  (must)
    3.10.11

Lib and Version:  (if None write None)
    numpy - 2.2.6

Only accessed by:  (must)
    Only __init__.py

Description: (if None write None)
    Realize the EEMD.

Modify:  (must)
    2026.3.25 - Create.
    2026.3.27 - Change the EEMD class' usage. EEMD(parallel=True) (default) -> EEMD(parallel=False). Now, it will not use parallel default.
    2026.4.2  - Finish the Optimization of the EEMD. Correct the logic.
    2026.5.1  - Fix: use PyEMD.EEMD to ensure standard algorithm.
"""

import numpy as np
from typing import Union, Tuple, Optional
from .Utils import Check_Time_and_Signal


def eemd(
    S: Union[list, np.ndarray],
    T: Optional[Union[list, np.ndarray]] = None,
    trials: int = 100,
    noise_width: float = 0.05,
    max_imf: int = -1,
    parallel: bool = False,
    **kwargs
) -> Tuple[np.ndarray, np.ndarray, None]:
    """
    EEMD: Ensemble Empirical Mode Decomposition
    (Standard implementation via PyEMD)

    :param S: Signal (1-dim)
    :param T: Time axis (1-dim). Default uniform.
    :param trials: Number of white noise realizations (ensemble size).
    :param noise_width: Standard deviation of added white noise relative to S std.
    :param max_imf: Max number of IMFs to extract. -1 for all.
    :param parallel: Parallelize the EEMD.
    :param kwargs: Additional parameters passed to PyEMD.EEMD.
    :return: IMFs (n_IMFs, N), Res (N,)
    """
    from PyEMD import EEMD

    S, T, N = Check_Time_and_Signal(S, T)

    decomposer = EEMD(trials=trials, noise_width=noise_width, parallel=parallel, **kwargs)
    result = decomposer.eemd(S, T, max_imf=max_imf)

    IMFs = result[:-1, :]   # shape [n_imfs, N]
    Res  = result[-1, :]    # shape [N,]

    return IMFs, Res, None