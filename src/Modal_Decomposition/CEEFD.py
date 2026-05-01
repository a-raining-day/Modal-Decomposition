"""
Python version:  (must)
    3.10.11

Lib and Version:  (if None write None)
    EMD-S - 1.9.0

Only accessed by:  (must)
    Only __init__.py

Description: (if None write None)
    Realize the CEEFD and CEEMDAN.

Modify:  (must)
    2026.3.25 - Create
    2026.3.30 - Desperate the CEEMDAN and the CEEFD, and rename the Cyclic_CEEFD as CEEFD, del the origin CEEFD.
    2026.4.7  - Fix the problem when the freq_bins include 0 will make the idx out of list. And change _extract_imf to staticmethod.
"""

import numpy as np
from typing import Union, Tuple


class ceefd:
    """CEEFD: Cyclic Envelop Empirical Fourier Decomposition"""

    def __init__(self, fs=1.0, min_peak_distance=10, envelop_iter=3):
        self.fs = fs
        self.min_peak_distance = min_peak_distance
        self.envelop_iter = envelop_iter

    def __call__(self, S):
        return self.decompose(S)

    def _compute_spectral_envelope(self, mag_spectrum):
        from scipy.signal import windows

        n = len(mag_spectrum)
        envelope = np.copy(mag_spectrum)
        window_size = int(0.05 * n)
        window = windows.boxcar(window_size)

        for _ in range(self.envelop_iter):
            envelope = np.convolve(envelope, window, mode='same') / window_size
            envelope = np.maximum(envelope, mag_spectrum)
        return envelope

    @staticmethod
    def _extract_imf(signal, freq_bins):
        N = len(signal)
        fft_signal = np.fft.fft(signal)

        mask = np.zeros(N, dtype=bool)
        mask[freq_bins] = True
        # mask[N - np.array(freq_bins)] = True -> \  # 2026.4.7
        mask[(N - np.array(freq_bins)) % N] = True

        imf_fft = fft_signal * mask
        imf = np.fft.ifft(imf_fft).real

        return imf, mask

    def decompose(self, S: Union[list, np.ndarray], T: Union[list, np.ndarray]=None) -> Tuple[np.ndarray, np.ndarray, None]:
        """
        CEEFD: Cyclic Envelope Empirical Fourier Decomposition

        :param S: Signal
        :param T: Time-axis, accepted formation is Unix array. Default uniformly sample.
        :return: IMFs (2-dim), Res (1-dim), None
        """

        from scipy.signal import find_peaks

        if not isinstance(S, np.ndarray):
            S = np.array(S)

        if S.ndim == 0:
            raise ValueError("The dim of the S must be 1-dim, not 0")

        elif S.ndim > 1:
            if 1 in S.shape:
                S = S.reshape(-1)

            else:
                raise ValueError(f"The dim of S must be 1-dim, not {S.ndim}")

        N = len(S)

        if T is None:  # if T is None, default generate uniform T-axis.
            T = np.arange(N)  # default fs = 1
            print(f"Warn: T is None，default T = [0, 1, 2, ..., {N - 1}]")

        else:
            if not isinstance(T, np.ndarray):
                T = np.array(T)

        fft_signal = np.fft.fft(S)
        mag_spectrum = np.abs(fft_signal[:N // 2 + 1])

        envelope = self._compute_spectral_envelope(mag_spectrum)

        peaks, properties = find_peaks(envelope, distance=self.min_peak_distance)

        if len(peaks) == 0:
            return [S], []

        boundaries = [0]
        for i in range(len(peaks) - 1):
            boundary = (peaks[i] + peaks[i + 1]) // 2
            boundaries.append(boundary)
        boundaries.append(N // 2)

        imfs = []
        freq_masks = []

        for i in range(len(boundaries) - 1):
            start_bin = boundaries[i]
            end_bin = boundaries[i + 1]

            if end_bin - start_bin < 2:
                continue

            freq_bins = list(range(start_bin, end_bin))
            imf, mask = self._extract_imf(S, freq_bins)
            imfs.append(imf)
            freq_masks.append(mask)

        residual = S.copy()
        for imf in imfs:
            residual -= imf

        if np.abs(residual).max() > 1e-10:
            imfs.append(residual)

        return np.array(imfs), residual, None