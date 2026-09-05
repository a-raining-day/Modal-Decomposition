"""
Python version:  (must)
    3.10.11

Lib and Version:  (if None write None)
    vmdpy - 0.2

Only accessed by:  (must)
    Only __init__.py

Description: (if None write None)
    Realize VMD by using vmdpy lib.

Modify:  (must)
    2026.3.25 - Create.
    2026.5.1  - Change the construction of return.
    2026.5.2  - Fix the Res, From "None" -> "np.zeros".
"""

from typing import Union, Tuple
import numpy as np
from .Utils import Check_Time_and_Signal


def vmd(S: Union[list, np.ndarray], alpha = 2000, tau = 0.0, K = 2, DC = 0, init = 1, tol = 1e-7) -> Tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """
    VMD: Variational Mode Decomposition

    :param S: Signal (1-dim)
    :param alpha: broadband constraints
    :param tau: noise tolerance
    :param K: num of IMFs
    :param DC: is included directional component
    :param init: way of initial
    :param tol: convergence threshold
    :return: IMFs (2-dim), No Res(zeros), Dict[u_hat (2-dim), omega (1-dim)]
    """
    from vmdpy import VMD

    S, _, _ = Check_Time_and_Signal(S)

    u, u_hat, omega = VMD(S, alpha, tau, K, DC, init, tol)

    Info = \
    {
        "u_hat": u_hat,
        "omega": omega
    }

    return u, np.zeros(u.shape[1]), Info