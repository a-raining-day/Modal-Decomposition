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
    2026.7.7  - Reconstruct with abstract class. And correct the reference.
"""

import numpy as np

from ._Registry import register_class, register_function
from .Utils import Check_ST_and_Transform
from .Base import Decomposer, DecompositionResult, Name, Reference, Config, CEEFDConfig

@register_class("CEEFD")
class CEEFD(Decomposer):
    name = "CEEFD"

    def __init__(self, fs: int | float, min_peak_distance: int, envelop_iter: int, T: list | np.ndarray=None, dim: int = 1, RAISE: bool = True):
        self.fs = fs
        self.min_peak_distance = min_peak_distance
        self.envelop_iter = envelop_iter
        self.T = T
        self.dim = dim
        self.RAISE = RAISE
        self.config: Config = Config()

    def decompose(self, S: list | np.ndarray, **kwargs) -> DecompositionResult:
        """
        CEEFD: Cyclic Envelope Empirical Fourier Decomposition

        :param S: Signal
        :return: IMFs(2-dim), Res(1-dim), Info(boundaries(Optional), fft_signal, mag_spectrum, peaks, envelop), Config(fs, min_peak_distance, envelop_iter)
        """
        try:
            from scipy.signal import find_peaks
        except ImportError:
            raise ModuleNotFoundError("Scipy module not available")

        uniform, DimSure, S, T, N = Check_ST_and_Transform(S, self.T, self.dim, self.RAISE)

        fft_signal = np.fft.fft(S)
        mag_spectrum = np.abs(fft_signal[:N // 2 + 1])

        envelope = self._compute_spectral_envelope(mag_spectrum)

        peaks, properties = find_peaks(envelope, distance=self.min_peak_distance)

        if len(peaks) == 0:
            self.config = CEEFDConfig(self.fs, self.min_peak_distance, self.envelop_iter)
            return DecompositionResult \
            (
                S, np.zeros_like(S),
                {
                    "fft_signal": fft_signal, "mag_spectrum": mag_spectrum,
                    "peaks": peaks,
                    "envelope": envelope,
                    "dim_sure": DimSure,
                    "uniform": uniform,
                },
                self.config
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

        if np.abs(residual).max() > 1e-10:
            imfs.append(residual)

        self.config = CEEFDConfig(self.fs, self.min_peak_distance, self.envelop_iter)
        return DecompositionResult \
        (
            np.array(imfs), residual,
            {
                "Boundaries": boundaries,
                "fft_signal": fft_signal, "mag_spectrum": mag_spectrum,
                "peaks": peaks,
                "envelope": envelope,
                "dim_sure": DimSure,
                "uniform": uniform,
            },
            self.config
        )

    def _compute_spectral_envelope(self, mag_spectrum):
        try:
            from scipy.signal import windows
        except ImportError:
            raise ModuleNotFoundError("Scipy module not available")

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

@register_function("fast_CEEFD")
def fast_CEEFD(S: list | np.ndarray, T: list | np.ndarray = None, fs: float = 1.0, min_peak_distance: int = 10, envelop_iter: int = 3, dim: int = 1, RAISE: bool = True) -> DecompositionResult:
    ceefd = CEEFD(fs=fs, min_peak_distance=min_peak_distance, envelop_iter=envelop_iter, T=T, dim=dim, RAISE=RAISE)
    return ceefd.decompose(S)