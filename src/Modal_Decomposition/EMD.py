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
        spline_kind: Literal["akima", "cubic", "pchip", "cubic_hermite",
                             "slinear", "quadratic", "linear"] = "cubic",
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

        # default_T=False: the PyEMD sifter rebuilds its own timeline and
        # ignores a caller-provided T under the default extrema detection, so
        # never allocate the N-length default time axis (see tests/comparison).
        S, T, _N = Check_Time_and_Signal(
            S, T, ndim={1}, method=self.name, default_T=False
        )

        arr = np.asarray(
            PyEMD_EMD(spline_kind=self.spline_kind, nbsym=self.nbsym).emd(
                S, T, max_imf=self.max_imf
            ),
            dtype=np.float64,
        )

        # Normalize the PyEMD output to (IMFs, Res); handle the degenerate
        # 1-D/0-D cases before slicing so they cannot raise.
        if arr.ndim >= 2:
            Res = arr[-1, :]
            IMFs = arr[:-1, :]
        elif arr.ndim == 1:
            IMFs = arr.reshape(1, -1)
            Res = np.zeros(arr.shape[0], dtype=np.float64)
        else:  # 0-d: no usable components
            IMFs = np.zeros((1, _N), dtype=np.float64)
            Res = np.zeros(_N, dtype=np.float64)

        if IMFs.ndim == 1:
            IMFs = IMFs.reshape(1, -1)

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
