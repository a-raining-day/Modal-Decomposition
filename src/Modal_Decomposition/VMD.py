"""
Variational Mode Decomposition

Solves the variational problem of band-limited modes, implemented by
vmdpy.

References
----------
10.1109/TSP.2013.2288675
"""

import numpy as np
from dataclasses import dataclass
from typing import ClassVar

from .Base import Config, Decomposer, DecompositionResult
from ._Registry import register_class
from .Utils import Check_Time_and_Signal

__all__ = ["VMD", "VMDConfig"]


@dataclass(frozen=True, kw_only=True)
class VMDConfig(Config):
    """
    Effective parameters of a VMD run.
    """
    alpha: float
    tau: float
    K: int
    DC: int
    init: int
    tol: float


@register_class("VMD")
class VMD(Decomposer):
    name: ClassVar[str] = "VMD"

    def __init__(
        self,
        alpha: float = 2000,
        tau: float = 0.0,
        K: int = 2,
        DC: int = 0,
        init: int = 1,
        tol: float = 1e-7,
    ):
        """
        Parameters
        ----------
        alpha : float
            Balancing parameter of the data-fidelity constraint.
        tau : float
            Time-step of the dual ascent; 0 for noise slack.
        K : int
            Number of modes.
        DC : int
            Keep the first mode at DC (0-freq) when true.
        init : int
            Initialization of the center frequencies (0, 1, or 2).
        tol : float
            Convergence tolerance.
        """
        self.alpha = alpha
        self.tau = tau
        self.K = K
        self.DC = DC
        self.init = init
        self.tol = tol

    def decompose(self, S, T=None) -> DecompositionResult:
        """
        Decompose the signal into modes.

        VMD has no residual concept: ``Res`` is None.
        """
        from vmdpy import VMD as vmdpy_VMD

        S, T, _N = Check_Time_and_Signal(S, T, ndim={1}, method=self.name)

        u, u_hat, omega = vmdpy_VMD(S, self.alpha, self.tau, self.K, self.DC, self.init, self.tol)

        return DecompositionResult(
            u,
            None,
            {
                "u_hat": u_hat,
                "omega": omega,
            },
            VMDConfig(
                alpha=self.alpha,
                tau=self.tau,
                K=self.K,
                DC=self.DC,
                init=self.init,
                tol=self.tol,
            ),
        )
