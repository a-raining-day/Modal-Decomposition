"""
Empirical Mode Decomposition

Decomposes a 1-d signal into intrinsic mode functions (IMFs) and a
residual using the sifting process of PyEMD.

References
----------
10.1098/rspa.1998.0193
"""

import numpy as np
from dataclasses import dataclass
from typing import ClassVar, Literal

from .Base import Config, Decomposer, DecompositionResult
from ._Registry import register_class
from .Utils import Check_Time_and_Signal

__all__ = ["EMD", "EMDConfig"]


@dataclass(frozen=True, kw_only=True)
class EMDConfig(Config):
    """
    Effective parameters of an EMD run.
    """
    nbsym: int
    spline_kind: str
    max_imf: int


@register_class("EMD")
class EMD(Decomposer):
    name: ClassVar[str] = "EMD"

    def __init__(
        self,
        nbsym: int = 2,
        spline_kind: Literal["akima", "cubic", "pchip", "cubic_hermite", "slinear", "quadratic", "linear"] = "cubic",
        max_imf: int = -1,
    ):
        """
        Parameters
        ----------
        nbsym : int
            Number of extrema mirrored at the signal boundaries.
        spline_kind : str
            Interpolation kind for envelope fitting.
        max_imf : int
            Maximum number of IMFs; -1 decomposes completely.
        """
        self.nbsym = nbsym
        self.spline_kind = spline_kind
        self.max_imf = max_imf

    def decompose(self, S, T=None) -> DecompositionResult:
        """
        Decompose the signal into IMFs and a residual.
        """
        from PyEMD import EMD as PyEMD_EMD

        S, T, _N = Check_Time_and_Signal(S, T, ndim={1}, method=self.name)

        arr = np.asarray(
            PyEMD_EMD(spline_kind=self.spline_kind, nbsym=self.nbsym).emd(
                S, T, max_imf=self.max_imf
            ),
            dtype=np.float64,
        )

        Res = arr[-1, :]
        IMFs = arr[:-1, :]

        if IMFs.ndim == 1:
            IMFs = IMFs.reshape(1, -1)
        elif IMFs.ndim == 0:
            IMFs = np.zeros((1, Res.shape[0]))

        return DecompositionResult(
            IMFs,
            Res,
            {},
            EMDConfig(
                nbsym=self.nbsym,
                spline_kind=self.spline_kind,
                max_imf=self.max_imf,
            ),
        )
