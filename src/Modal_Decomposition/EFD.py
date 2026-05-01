"""
Python version:  (must)
    3.10.11

Lib and Version:  (if None write None)
    numpy - 2.2.6
	scipy - 1.15.3
	matplotlib - 3.10.8

Only accessed by:  (must)
    Only __init__.py

Description: (if None write None)
    Realize the EFD.
    Optimize the use of scipy.S

Modify:  (must)
    2026.3.25 - Create
    2026.4.2  - Finish the Optimization of the EFD. Del the origin efd function.
    2026.4.7  - Correct the error of the use of the np.concentrate and the construction of the wn.
"""

import numpy as np
from typing import Union, Tuple
from .Utils import Check_Time_and_Signal


def efd(S: Union[list, np.ndarray], T: Union[list, np.ndarray]=None, max_IMFs: int=-1, verbose: bool=False) -> Tuple[np.ndarray, np.ndarray, None]:
    """
    EFD: Empirical Fourier Decomposition

    :param S: Signal (1-dim)
    :param T: Time axis (1-dim)
    :param max_IMFs: the num of the IMFs. -1 means return all IMFs
    :param verbose: if print the info?
    :return: IMFs (n_IMFs, N), Res: (N,), None
    """

    from scipy.signal import argrelmax

    if not isinstance(max_IMFs, int):
        raise TypeError("The type of the max_IMFs must be int!")

    if max_IMFs != -1 and max_IMFs <= 0:
        if max_IMFs <= 0:
            raise ValueError("Invalid value! Do you want use -1?")

    S, T, N = Check_Time_and_Signal(S, T, verbose)

    dt = np.diff(T)
    if not np.allclose(dt, dt[0], rtol=1e-10, atol=1e-14):
        raise ValueError("Time series should be uniform intervals")

    # make seq 0-mean value
    MEAN = np.mean(S)
    S: np.ndarray = S - np.mean(S)

    F = np.fft.fft(S)
    # fs = np.fft.fftfreq(N // 2 + 1, d=T[1]-T[0])  # only positive freq S
    magnitude = np.abs(F)
    phase = np.angle(F)

    edge_magnitude = magnitude[: N // 2 + 1].copy()[1:] * 2  # filter the dc(loc[0])
    uniform_freq = np.linspace(0, np.pi, N // 2)
    freq_N = len(uniform_freq)

    local_maximum_points_tuple = argrelmax(edge_magnitude)
    local_maximum_points = local_maximum_points_tuple[0]
    local_maximum = edge_magnitude[local_maximum_points]

    if max_IMFs != -1:
        local_maximum_zip = [(point, value) for point, value in zip(local_maximum_points, local_maximum)]
        local_maximum_zip = sorted(local_maximum_zip, reverse=True, key=lambda x: x[1])
        local_maximum_points = list(map(lambda x: x[0], local_maximum_zip[:max_IMFs]))

    local_maximum_points = np.concatenate(([0], local_maximum_points, [freq_N - 1]))

    local_maximum_points = np.unique(local_maximum_points)
    local_maximum_points = np.sort(local_maximum_points)

    wn = []  # the zero phase filter
    for p in range(len(local_maximum_points) - 1):
        next_point = local_maximum_points[p + 1]
        current_point = local_maximum_points[p]

        if edge_magnitude[current_point] == edge_magnitude[next_point]:
            wn.append(current_point)

        else:
            # wn.append(np.argmin(edge_magnitude[current_point:next_point + 1]))
            wn.append(current_point + np.argmin(edge_magnitude[current_point:next_point + 1]))
    wn = np.concatenate(([0], wn, [freq_N - 1]))

    filters_arr = []
    for edge in range(1, len(wn)):
        filter_single = np.zeros(freq_N)

        current_point = wn[edge]
        last_point = wn[edge - 1]

        filter_single[last_point: current_point + 1] = 1
        filters_arr.append(filter_single)

    IMFs = []
    M = N // 2 + 1
    # the spectrum is symmetry, so we just need positive, and next, only need concentrate symmetrically.
    for _filter in filters_arr:
        filtered_magnitude = np.zeros(M, dtype=float)
        filtered_magnitude[1:] = magnitude[1:M] * _filter
        temp_spectrum = filtered_magnitude * np.exp(1j * phase[:M])

        full_spectrum = np.zeros(N, dtype=complex)
        full_spectrum[:M] = temp_spectrum  # complex temp_spectrum
        for i in range(1, M - 1):
            full_spectrum[N - i] = np.conj(temp_spectrum[i])

        IMFs.append(np.fft.ifft(full_spectrum).real)

    IMFs = np.array(IMFs)
    return IMFs, S - np.sum(IMFs, axis=0) + MEAN, None