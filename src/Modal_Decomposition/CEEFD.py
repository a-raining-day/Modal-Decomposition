"""
Cyclic Envelop Empirical Fourier Decomposition

Segments the spectral envelope at its peaks and reconstructs modes from
the segmented spectrum.

References
----------
10.3969/j.issn.1001-4551.2023.07.001
"""

import numpy as np
from dataclasses import dataclass
from typing import ClassVar

from .Base import Config, Decomposer, DecompositionResult
from ._Registry import register_class
from .Utils import Check_Time_and_Signal

__all__ = ["CEEFD", "CEEFDConfig"]


@dataclass(frozen=True, kw_only=True)
class CEEFDConfig(Config):
    """
    Effective parameters of a CEEFD run.
    """
    fs: int | float
    min_peak_distance: int
    envelop_iter: int


@register_class("CEEFD")
class CEEFD(Decomposer):
    name: ClassVar[str] = "CEEFD"

    def __init__(
        self,
        fs: int | float = 1.0,
        min_peak_distance: int = 10,
        envelop_iter: int = 3,
    ):
        """
        Parameters
        ----------
        fs : int | float
            Sampling frequency.
        min_peak_distance : int
            Minimum bin distance between envelope peaks.
        envelop_iter : int
            Number of envelope smoothing iterations.
        """
        self.fs = fs
        self.min_peak_distance = min_peak_distance
        self.envelop_iter = envelop_iter

    def decompose(self, S, T=None) -> DecompositionResult:
        """
        Decompose the signal into IMFs and a residual.
        """
        from scipy.signal import find_peaks

        S, T, N = Check_Time_and_Signal(S, T, ndim={1}, method=self.name)

        fft_signal = np.fft.fft(S)
        mag_spectrum = np.abs(fft_signal[:N // 2 + 1])

        envelope = self._compute_spectral_envelope(mag_spectrum)

        peaks, properties = find_peaks(envelope, distance=self.min_peak_distance)

        if len(peaks) == 0:
            return DecompositionResult(
                S.reshape(1, -1),
                np.zeros_like(S),
                {
                    "fft_signal": fft_signal,
                    "mag_spectrum": mag_spectrum,
                    "peaks": peaks,
                    "envelope": envelope,
                },
                CEEFDConfig(
                    fs=self.fs,
                    min_peak_distance=self.min_peak_distance,
                    envelop_iter=self.envelop_iter,
                ),
            )

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

        IMFs = np.array(imfs)
        if IMFs.shape[0] == 0:
            IMFs = np.empty((0, N), dtype=np.float64)

        return DecompositionResult(
            IMFs,
            residual,
            {
                "boundaries": boundaries,
                "fft_signal": fft_signal,
                "mag_spectrum": mag_spectrum,
                "peaks": peaks,
                "envelope": envelope,
            },
            CEEFDConfig(
                fs=self.fs,
                min_peak_distance=self.min_peak_distance,
                envelop_iter=self.envelop_iter,
            ),
        )

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
        mask[(N - np.array(freq_bins)) % N] = True

        imf_fft = fft_signal * mask
        imf = np.fft.ifft(imf_fft).real

        return imf, mask
