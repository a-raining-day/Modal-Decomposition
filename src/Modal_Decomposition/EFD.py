"""
Python version:  (must)
    3.10.11

Lib and Version:  (if None write None)
    numpy - 2.2.6
	scipy - 1.15.3
	matplotlib - 3.10.8

Only accessed by:  (must)
    Only __init__.py

Modify:  (must)
    2026.3.25

Description: (if None write None)
    Realize the EFD.
    Optimize the use of scipy.signal
"""

import numpy as np
from .COLOR import printc

def EFD(S, T=None, fs=None):
    """
    :param S: Signal (1-dim)
    :param T: Time axis (uniform spacing)
    :return: IMFs: np.ndarray (2-dim) | Res: np.ndarray (1-dim)
    """
    from scipy.signal import find_peaks

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

    if len(T) != N:
        raise ValueError(f"len of T: ({len(T)}) doesn't match ({N})")

    dt = np.diff(T)
    if not np.allclose(dt, dt[0], rtol=1e-10, atol=1e-14):
        raise ValueError("Time series should be uniform intervals")

    # FFT
    fft = np.fft.fft
    F = fft(S)
    freq = np.fft.fftfreq(len(T), d=T[1]-T[0])

    f = np.abs(F)
    idx = np.where(freq >= 0)[0]
    freq_positive = freq[idx]
    spectrum = f[idx]

    # boundary
    part_max, _ = find_peaks(spectrum)
    part_min, _ = find_peaks(-spectrum)
    boundary = np.concatenate([np.array([0]), part_min, part_max, np.array([len(freq_positive) - 1])])
    boundary = np.unique(boundary)
    boundary = np.sort(boundary)

    IMFs = []

    freq_set = []
    spectrum_set = []

    for b in range(1, len(boundary)):
        down_bound = boundary[b - 1]
        up_bound = boundary[b]

        filter_mask = np.zeros(N)
        filter_mask[N // 2 + down_bound : N // 2 + up_bound + 1] = 1
        filter_mask[N // 2 - up_bound : N // 2 - down_bound + 1] = 1

        if down_bound == 0:
            filter_mask[N // 2 + 1 : N // 2 + up_bound + 1] = 1

        F_band = F * filter_mask

        IMF = np.fft.ifft(F_band)
        IMFs.append(np.real(IMF))

    sum_IMFs = np.sum(IMFs, axis=0)
    Res: np.ndarray = np.array(S - sum_IMFs)
    IMFs: np.ndarray = np.array(IMFs)

    return IMFs, Res