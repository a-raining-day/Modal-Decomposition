"""
Empirical Wavelet Transform

Builds adaptive wavelets on the detected spectrum support, implemented by
ewtpy.

References
----------
10.48550/arXiv.2304.06274
"""

import numpy as np
from dataclasses import dataclass
from typing import ClassVar

from .Base import Config, Decomposer, DecompositionResult
from ._Registry import register_class
from .Utils import Check_Time_and_Signal

__all__ = ["EWT", "EWTConfig"]


@dataclass(frozen=True, kw_only=True)
class EWTConfig(Config):
    """
    Effective parameters of an EWT run.
    """
    N: int
    log: int
    detect: str
    completion: int
    reg: str
    lengthFilter: int
    sigmaFilter: int


@register_class("EWT")
class EWT(Decomposer):
    name: ClassVar[str] = "EWT"

    def __init__(
        self,
        N: int = 5,
        log: int = 0,
        detect: str = "locmax",
        completion: int = 0,
        reg: str = "average",
        lengthFilter: int = 10,
        sigmaFilter: int = 5,
    ):
        """
        Parameters
        ----------
        N : int
            Number of modes.
        log : int
            Logarithm of the number of Fourier bounds.
        detect : str
            Boundary detection method.
        completion : int
            Whether to complete the boundary set.
        reg : str
            Regularization for the empirical filters.
        lengthFilter : int
            Length filter for boundaries.
        sigmaFilter : int
            Sigma filter for boundaries.
        """
        self.N = N
        self.log = log
        self.detect = detect
        self.completion = completion
        self.reg = reg
        self.lengthFilter = lengthFilter
        self.sigmaFilter = sigmaFilter

    def decompose(self, S, T=None) -> DecompositionResult:
        """
        Decompose the signal into modes and a residual.

        The time axis is validated but not used by the algorithm.
        """
        from ewtpy import EWT1D

        S, T, _N = Check_Time_and_Signal(S, T, ndim={1}, method=self.name)

        ewt, mfb, boundaries = EWT1D(
            S, self.N, self.log, self.detect, self.completion,
            self.reg, self.lengthFilter, self.sigmaFilter,
        )
        ewt = np.asarray(ewt).T
        mfb = np.asarray(mfb).T

        return DecompositionResult(
            ewt[:-1, :],
            ewt[-1, :],
            {
                "mfb": mfb,
                "boundaries": boundaries,
            },
            EWTConfig(
                N=self.N,
                log=self.log,
                detect=self.detect,
                completion=self.completion,
                reg=self.reg,
                lengthFilter=self.lengthFilter,
                sigmaFilter=self.sigmaFilter,
            ),
        )
