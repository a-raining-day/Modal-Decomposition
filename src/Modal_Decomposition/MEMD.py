"""

Python version:  (must)
    3.10.11

Lib and Version:  (if None write None)
    numpy - 2.2.6

Only accessed by:  (must)
    Only __init__.py

Description: (if None write None)
    Realize the MEMD

Modify:  (must)
    2026.3.25 - Create.
    2026.5.1  - Fix the problem when calculating projections.
"""

import numpy as np
from typing import Union, Tuple


def memd(S: Union[list, np.ndarray], d=None, k=None, max_imf=None, sd_thresh=0.2, max_iter=10, spline_kind: str="linear") -> Tuple[np.ndarray, np.ndarray]:
    """
    MEMD: Multimodal Empirical Mode Decomposition

    :param S: Signal (2-dim), (d, N) | d -> channels,N -> time points
    :param d: channels or dimensions
    :param k: number of directional vector,default: d * 128
    :param max_imf: max num of IMFs
    :param sd_thresh: therahold
    :param max_iter: max iterations of each IMF
    :param spline_kind: spline kind.
    :return: IMFs (IMFs_num, d, N), Res (2-dim)
    """

    if not isinstance(S, np.ndarray):
        S = np.array(S)

    if S.ndim != 2:
        raise ValueError(f"dim of S must be 2-dim, not {S.ndim}!")

    if d is None:
        d = S.shape[0]

    if S.shape[0] != d:
        raise ValueError(f"shape of S should be: ({d}, N)，but get -> {S.shape}")

    N = S.shape[1]
    T = np.arange(N)

    if k is None:
        k = d * 128

    if max_imf is None:
        max_imf = int(np.log2(N))

    vectors = generate_hammersley_points(k, d)  # dhape: (d, k)

    # projections = np.dot(S.T, vectors)  # shape: (N, k)

    imfs = []
    residue = S.copy()

    for imf_idx in range(max_imf):
        h = residue.copy()

        for iter_num in range(max_iter):
            h_old = h
            mean_envelope = compute_local_mean(h, vectors, T, spline_kind=spline_kind)

            h_new = h - mean_envelope
            sd = np.sum((h_old - h_new) ** 2) / np.sum(h_old ** 2)
            h = h_new

            if sd < sd_thresh or iter_num == max_iter - 1:
                break

        imfs.append(h.copy())

        residue = residue - h

        if should_stop(residue):
            break

    imfs_array = np.array(imfs)  # shape: (n_imfs, d, N)

    return imfs_array, residue


def generate_hammersley_points(k, d):
    vectors = np.zeros((d, k))

    primes = generate_primes(d - 1)

    for i in range(k):
        point = np.zeros(d)

        point[0] = radical_inverse_vdc(i)

        for j in range(1, d):
            base = primes[j - 1] if j - 1 < len(primes) else primes[-1]
            point[j] = radical_inverse(i, base)

        norm = np.linalg.norm(point)
        if norm > 0:
            point = point / norm

        vectors[:, i] = point

    return vectors


def radical_inverse_vdc(index):
    """Van der Corput"""
    bits = index
    bits = (bits << 16) | (bits >> 16)
    bits = ((bits & 0x55555555) << 1) | ((bits & 0xAAAAAAAA) >> 1)
    bits = ((bits & 0x33333333) << 2) | ((bits & 0xCCCCCCCC) >> 2)
    bits = ((bits & 0x0F0F0F0F) << 4) | ((bits & 0xF0F0F0F0) >> 4)
    bits = ((bits & 0x00FF00FF) << 8) | ((bits & 0xFF00FF00) >> 8)
    return float(bits) / 2 ** 32


def radical_inverse(index, base):
    result = 0.0
    f = 1.0 / base
    i = index
    while i > 0:
        result += f * (i % base)
        i = i // base
        f = f / base
    return result


def generate_primes(n):
    if n <= 0:
        return []

    primes = []
    num = 2
    while len(primes) < n:
        is_prime = True
        for p in primes:
            if p * p > num:
                break
            if num % p == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(num)
        num += 1

    return primes


def compute_local_mean(signal, vectors, T, spline_kind: str="linear"):
    """
    S: now S (d, N)
    vectors: directional vector (d, k)
    projections: projecttion of origin S (N, k)
    T
    """
    from scipy.signal import find_peaks
    from scipy.interpolate import interp1d

    d, N = signal.shape
    k = vectors.shape[1]

    current_projections = np.dot(signal.T, vectors)

    mean_envelope = np.zeros((d, N))

    for dir_idx in range(k):
        proj = current_projections[:, dir_idx]

        max_indices, _ = find_peaks(proj)
        min_indices, _ = find_peaks(-proj)

        if len(max_indices) == 0:
            max_indices = np.array([0, N - 1])
        else:
            if max_indices[0] != 0:
                max_indices = np.concatenate([[0], max_indices])
            if max_indices[-1] != N - 1:
                max_indices = np.concatenate([max_indices, [N - 1]])

        if len(min_indices) == 0:
            min_indices = np.array([0, N - 1])
        else:
            if min_indices[0] != 0:
                min_indices = np.concatenate([[0], min_indices])
            if min_indices[-1] != N - 1:
                min_indices = np.concatenate([min_indices, [N - 1]])

        for ch in range(d):
            max_values = signal[ch, max_indices]
            if len(max_indices) >= 4:
                try:
                    max_spline = interp1d(max_indices, max_values, kind=spline_kind,fill_value='extrapolate')
                    max_env = max_spline(T)
                except:
                    max_env = np.interp(T, max_indices, max_values)
            else:
                max_env = np.interp(T, max_indices, max_values)

            min_values = signal[ch, min_indices]
            if len(min_indices) >= 4:
                try:
                    min_spline = interp1d(min_indices, min_values, kind=spline_kind,fill_value='extrapolate')
                    min_env = min_spline(T)
                except:
                    min_env = np.interp(T, min_indices, min_values)
            else:
                min_env = np.interp(T, min_indices, min_values)

            mean_envelope[ch] += (max_env + min_env) / 2

    mean_envelope = mean_envelope / k

    return mean_envelope

def should_stop(residue):
    from scipy.signal import find_peaks

    d, N = residue.shape

    for ch in range(d):
        diff = np.diff(residue[ch])
        if np.all(diff >= 0) or np.all(diff <= 0):
            return True

        max_indices, _ = find_peaks(residue[ch])
        min_indices, _ = find_peaks(-residue[ch])
        if len(max_indices) + len(min_indices) <= 2:
            return True

    return False
