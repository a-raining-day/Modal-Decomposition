"""
Python version:  (must)
    3.10.11

Lib and Version:  (if None write None)
    numpy - 2.2.6
	time - 1.39.2
	tqdm - 4.67.3

Only accessed by:  (must)
    Only __init__.py

Description: (if None write None)
    Realize the CEEMD.

Modify:  (must)
    2026.3.25 - Create
    2026.4.2  - Finish the Optimization of the CEEMD.
"""

from typing import Union, Tuple, Optional
from time import sleep
import numpy as np
from .EMD import emd
from .Utils import monotonic_increasing, monotonic_decreasing, Check_Time_and_Signal
from warnings import warn


def ceemd(S: Union[list, np.ndarray], T: Union[list, np.ndarray]=None, N_whitenoise=37, beta=0.3, max_imf: Optional[int]=None, dead_line: int=10, verbose: bool=False) \
        -> Tuple[np.ndarray, np.ndarray, None]:
    """
    CEEMD: Complementary Ensemble Empirical Mode Decomposition

    :param S: Signal (1-dim)
    :param T: the time axis.
    :param N_whitenoise: the num of the added whitenoise.
    :param beta:
    :param max_imf: -1, None or other int | -1 means decompose completely, None means give a int auto, other int means the num of the IMFs
    :param dead_line: Sometime it'll be in unuseful cycle, when the average of the N's sequence with added whitenoise is empty([]). It'll be forced exit when the time of the cycle above the deadline.
    :param verbose:
    :return: IMFs (n_IMFs, N), Res (N,), None
    """
    if beta <= 0:
        raise ValueError("The beta should > 0")

    if N_whitenoise <= 0 or not isinstance(N_whitenoise, int):
        raise TypeError("N_whitenoise must be int type or > 0")

    S, T, N = Check_Time_and_Signal(S, T, verbose)

    if not np.all(np.diff(T) > 0):
        raise ValueError("T should be monotonic increasing!")

    if np.any(np.allclose(np.diff(np.diff(T)), 2)):
        warn("The T is not uniform! Some error may happen.")

    if max_imf is None:
        max_imf = int(np.log2(len(S))) + 2

    Res = S
    IMFs = []

    # std_dev = np.std(S)

    count = 0
    dead_cycle = 0
    while True:
        std_dev = np.std(Res)

        _IMFs = []
        for n in range(N_whitenoise):
            white_noise = np.random.normal(0, std_dev * beta, N)
            S_plus = Res + white_noise
            S_minus = Res - white_noise
            _IMFs_plus, _ = emd(S_plus, T, max_imf=1)
            _IMFs_minus, _ = emd(S_minus, T, max_imf=1)

            if not _IMFs_plus or not _IMFs_minus:
                continue

            _IMFs.append((_IMFs_plus[0] + _IMFs_minus[0]) / 2.0)

        if not _IMFs:
            dead_cycle += 1
            continue

        if dead_cycle >= dead_line:
            raise RuntimeError("Trapped in a vicious cycle")
        else:
            dead_cycle = 0

        IMF = np.mean(_IMFs, axis=0)

        IMFs.append(IMF)

        Res -= IMF
        count += 1

        if verbose:
            if count % 10 == 0:
                print(f"has get {count} IMFs...")
                sleep(1)

        if max_imf != -1:
            if count >= max_imf or monotonic_increasing(Res) or monotonic_decreasing(Res):
                return np.array(IMFs), np.array(Res), None

        else:
            if monotonic_increasing(Res) or monotonic_decreasing(Res):
                return np.array(IMFs), np.array(Res), None
