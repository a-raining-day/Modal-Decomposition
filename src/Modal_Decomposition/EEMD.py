"""
Python version:  (must)
    3.10.11

Lib and Version:  (if None write None)
    EMD-signal - 1.9.0

Only accessed by:  (must)
    Only __init__.py

Description: (if None write None)
    Realize the EEMD.

Modify:  (must)
    2026.3.25 - I don't know, because when I'm find I don't write this line.
    2026.3.27 - Change the EEMD class' usage. EEMD(parallel=True) (default) -> EEMD(parallel=False). Now, it will not use parallel default.
"""
import numpy as np
from PyEMD import EEMD
from typing import Union, Tuple

Origin_EEMD = EEMD

EEMD = EEMD(parallel=False)
def eemd(S: Union[list, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """
    :param S: Signal (1-dim)
    :return: IMFs (2-dim), Res (1-dim)
    """
    if not isinstance(S, np.ndarray):
        S = np.array(S)

    IMFs = EEMD.eemd(S)
    Res = EEMD.residue

    return IMFs, Res