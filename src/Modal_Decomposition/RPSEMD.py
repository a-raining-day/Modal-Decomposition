"""
Random Phase Sinusoidal Assisted Empirical Mode Decomposition

Assists EMD with phase-shifted sinusoids at the dominant frequency of the
current residual to separate close modes.

References
----------
10.1109/LSP.2016.2537376
"""

import numpy as np
from dataclasses import dataclass
from typing import ClassVar

from .Base import Config, Decomposer, DecompositionResult
from ._Registry import register_class
from .EMD import EMD
from .Utils import Check_Time_and_Signal, is_monotonic

__all__ = ["RPSEMD", "RPSEMDConfig"]


@dataclass(frozen=True, kw_only=True)
class RPSEMDConfig(Config):
    """
    Effective parameters of an RPSEMD run.
    """
    f: float | None
    M: int
    max_imf: int
    fs: float
    spline_kind: str
    nbsym: int


@register_class("RPSEMD")
class RPSEMD(Decomposer):
    name: ClassVar[str] = "RPSEMD"

    def __init__(
        self,
        f: float | None = None,
        M: int = 4,
        max_imf: int | None = None,
        fs: float = 1.0,
        spline_kind: str = "CubicSpline",
        nbsym: int = 2,
    ):
        """
        Parameters
        ----------
        f : float | None
            Fixed assisting frequency; None re-estimates it from the
            dominant spectrum peak of the current residual.
        M : int
            Number of phase shifts of the assisting sinusoid.
        max_imf : int | None
            Maximum number of IMFs; None selects log2(N).
        fs : float
            Sampling frequency used for frequency estimation.
        spline_kind : str
            Envelope backend of the internal native EMD: one of
            "CubicSpline" / "PCHIP" / "linear" (canonical ``Utils.Spline``
            names; "linear" = np.interp).
        nbsym : int
            Number of mirrored extrema for the internal EMD.
        """
        self.f = f
        self.M = M
        self.max_imf = max_imf
        self.fs = fs
        self.spline_kind = spline_kind
        self.nbsym = nbsym

    def decompose(self, S, T=None) -> DecompositionResult:
        """
        Decompose the signal into IMFs and a residual.
        """
        S, T, _N = Check_Time_and_Signal(S, T, ndim={1}, method=self.name)

        IMFs = []
        Res = S

        max_imf = self.max_imf
        if max_imf is None:
            max_imf = int(np.log2(len(S)))

        count = 0
        while True:
            if self.f is None:
                N = len(Res)
                fft_vals = np.fft.rfft(Res)
                freqs = np.fft.rfftfreq(N, 1 / self.fs)

                magnitude = np.abs(fft_vals[1:])  # filter the dc
                if len(magnitude) > 0:
                    main_freq_idx = np.argmax(magnitude) + 1
                    current_f = freqs[main_freq_idx]
                else:
                    current_f = self.fs / N
            else:
                current_f = self.f

            phi = np.array([2.0 * np.pi * i / self.M for i in range(self.M)])  # generate the sin wave
            orders = []

            for m in range(self.M):
                Am_t = np.sin(2 * np.pi * T * current_f + phi[m])
                Xm_t = Res + Am_t

                _IMFs = EMD(
                    spline_kind=self.spline_kind,
                    nbsym=self.nbsym,
                    max_imf=1,
                ).decompose(Xm_t, T).IMFs

                if _IMFs.shape[0] > 0:
                    orders.append(_IMFs[0])

            if len(orders) == 0:
                break

            IMF = np.mean(orders, axis=0)
            IMFs.append(IMF)
            Res = Res - IMF

            count += 1

            if max_imf != -1 and count >= max_imf or len(Res) < 4 or np.std(Res) < 1e-10 or is_monotonic(Res):
                break

        IMFs = np.array(IMFs)
        if IMFs.shape[0] == 0:
            IMFs = np.empty((0, len(S)), dtype=np.float64)

        return DecompositionResult(
            IMFs,
            np.array(Res),
            {},
            RPSEMDConfig(
                f=self.f,
                M=self.M,
                max_imf=max_imf,
                fs=self.fs,
                spline_kind=self.spline_kind,
                nbsym=self.nbsym,
            ),
        )
