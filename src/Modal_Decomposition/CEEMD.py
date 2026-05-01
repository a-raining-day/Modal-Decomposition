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
    2026.5.1  - Fix the error of the decomposition.
"""

from typing import Union, Tuple, Optional
import numpy as np
from .EMD import emd
from .Utils import monotonic_increasing, monotonic_decreasing, Check_Time_and_Signal
from warnings import warn


def ceemd(S: Union[list, np.ndarray], T: Union[list, np.ndarray] = None, N_whitenoise=37, beta=0.2, max_imf: Optional[int] = None, dead_line: int = 10, verbose: bool = False) \
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

    imfs = []
    residual = S.copy()
    k = 0
    dead_cycle = 0

    while True:
        if max_imf != -1 and k >= max_imf:
            break

        imf_candidates = []
        std_dev = np.std(residual)

        for i in range(N_whitenoise):
            white_noise = np.random.normal(0, std_dev * beta, N)
            S_plus = residual + white_noise
            S_minus = residual - white_noise

            imfs_plus, _, _ = emd(S_plus, T, max_imf=1)
            imfs_minus, _, _ = emd(S_minus, T, max_imf=1)

            if len(imfs_plus) > 0 and len(imfs_minus) > 0:
                imf_candidate = (imfs_plus[0] + imfs_minus[0]) / 2.0
                imf_candidates.append(imf_candidate)

        if not imf_candidates:
            dead_cycle += 1
            if dead_cycle >= dead_line:
                raise RuntimeError("Trapped in a vicious cycle")
            continue

        imf = np.mean(imf_candidates, axis=0)
        imfs.append(imf)
        residual = residual - imf
        k += 1

        from scipy.signal import argrelextrema
        peaks = argrelextrema(residual, np.greater)[0]
        valleys = argrelextrema(residual, np.less)[0]

        if monotonic_increasing(residual) or monotonic_decreasing(residual) or len(peaks) + len(valleys) < 3:
            break

    return np.array(imfs), np.array(residual), None