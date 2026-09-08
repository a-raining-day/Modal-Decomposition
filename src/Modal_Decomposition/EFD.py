"""
Empirical Fourier Decomposition

Partitions the amplitude spectrum at local maxima and reconstructs modes
from the segmented spectrum.

References
----------
10.1016/j.ymssp.2021.108155
"""

import numpy as np
from dataclasses import dataclass
from typing import ClassVar

from .Base import Config, Decomposer, DecompositionResult
from ._Registry import register_class
from .Utils import Check_Time_and_Signal

__all__ = ["EFD", "EFDConfig"]


@dataclass(frozen=True, kw_only=True)
class EFDConfig(Config):
    """
    Effective parameters of an EFD run.
    """
    max_IMFs: int


@register_class("EFD")
class EFD(Decomposer):
    name: ClassVar[str] = "EFD"

    def __init__(self, max_IMFs: int = -1):
        """
        Parameters
        ----------
        max_IMFs : int
            Maximum number of IMFs; -1 returns all IMFs.
        """
        self.max_IMFs = max_IMFs

        if not isinstance(max_IMFs, int):
            raise TypeError("The type of the max_IMFs must be int!")

        if max_IMFs != -1 and max_IMFs <= 0:
            raise ValueError("Invalid value! Do you want use -1?")

    def decompose(self, S, T=None) -> DecompositionResult:
        """
        Decompose the signal into IMFs and a residual.
        """
        from scipy.signal import argrelmax

        S, T, N = Check_Time_and_Signal(S, T, ndim={1}, method=self.name)

        # make seq 0-mean value
        MEAN = np.mean(S)
        S = S - np.mean(S)

        F = np.fft.fft(S)
        magnitude = np.abs(F)
        phase = np.angle(F)

        edge_magnitude = magnitude[: N // 2 + 1].copy()[1:] * 2  # filter the dc(loc[0])
        uniform_freq = np.linspace(0, np.pi, N // 2)
        freq_N = len(uniform_freq)

        local_maximum_points_tuple = argrelmax(edge_magnitude)
        local_maximum_points = local_maximum_points_tuple[0]
        local_maximum = edge_magnitude[local_maximum_points]

        if self.max_IMFs != -1:
            local_maximum_zip = [(point, value) for point, value in zip(local_maximum_points, local_maximum)]
            local_maximum_zip = sorted(local_maximum_zip, reverse=True, key=lambda x: x[1])
            # The segmentation below yields (internal maxima + 2) bands, so
            # m IMFs need (m - 2) internal maxima. The old code kept
            # max_IMFs maxima and produced up to max_IMFs + 2 modes,
            # violating the documented "maximum number of IMFs" contract.
            internal = local_maximum_zip[: max(0, self.max_IMFs - 2)]
            local_maximum_points = sorted(p for p, _ in internal)

        bounds = np.concatenate(([0], local_maximum_points, [freq_N - 1])).astype(np.int64)
        bounds = np.unique(bounds)
        bounds = np.sort(bounds)

        if self.max_IMFs == 1:
            # exactly one mode: a single band over the whole spectrum
            wn = np.array([0, freq_N - 1])
        else:
            wn = []  # the zero phase filter edges
            for p in range(len(bounds) - 1):
                next_point = bounds[p + 1]
                current_point = bounds[p]

                if edge_magnitude[current_point] == edge_magnitude[next_point]:
                    wn.append(current_point)
                else:
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
        # the spectrum is symmetric, so we just need positive, and next, only need concentrate symmetrically.
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
        Res = S - np.sum(IMFs, axis=0) + MEAN

        info = \
        {
            "mean": MEAN,
            "fft_signal": F,
            "magnitude": magnitude,
            "phase": phase,
        }

        return DecompositionResult(
            IMFs,
            Res,
            info,
            EFDConfig(max_IMFs=self.max_IMFs),
        )
