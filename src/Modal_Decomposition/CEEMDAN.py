"""
Complete Ensemble Empirical Mode Decomposition with Adaptive Noise

Ensemble decomposition with adaptive noise at each stage, implemented by
PyEMD.

References
----------
10.1109/ICASSP.2011.5947265
"""

import numpy as np
from dataclasses import dataclass
from typing import ClassVar, Literal

from .Base import Config, Decomposer, DecompositionResult
from ._Registry import register_class
from .Utils import Check_Time_and_Signal, resolve_seed

__all__ = ["CEEMDAN", "CEEMDANConfig"]


@dataclass(frozen=True, kw_only=True)
class CEEMDANConfig(Config):
    """
    Effective parameters of a CEEMDAN run.
    """
    trials: int
    noise_scale: float
    nbsym: int
    max_imf: int
    range_thr: float
    total_power_thr: float
    noise_kind: str
    seed: int | None
    spline_kind: str
    extrema_detection: str
    parallel: bool
    processes: int | None


@register_class("CEEMDAN")
class CEEMDAN(Decomposer):
    name: ClassVar[str] = "CEEMDAN"

    def __init__(
        self,
        trials: int = 100,
        noise_scale: float = 1.0,
        nbsym: int = 2,
        max_imf: int = -1,
        range_thr: float = 0.01,
        total_power_thr: float = 0.05,
        noise_kind: Literal["normal", "uniform"] = "normal",
        seed: int | None = None,
        spline_kind: Literal["akima", "cubic", "pchip", "cubic_hermite",
                             "slinear", "quadratic", "linear"] = "cubic",
        extrema_detection: Literal["simple", "parabol"] = "simple",
        parallel: bool = False,
        processes: int | None = None,
    ):
        """
        Parameters
        ----------
        trials : int
            Number of ensemble realizations.
        noise_scale : float
            Amplitude of the added noise.
        nbsym : int
            Number of extrema mirrored at the boundaries.
        max_imf : int
            Maximum number of IMFs; -1 decomposes completely.
        range_thr : float
            Fraction of the signal range used as a stopping threshold.
        total_power_thr : float
            Residual-power stopping threshold.
        noise_kind : Literal["normal", "uniform"]
            Distribution of the added noise.
        seed : int | None
            Local random seed. Overridden by a global seed when both are set.
        spline_kind : str
            Interpolation kind for envelope fitting.
        extrema_detection : Literal["simple", "parabol"]
            Extrema detection method.
        parallel : bool
            Enable multiprocessing.
        processes : int | None
            Number of processes when ``parallel`` is True.
        """
        self.trials = trials
        self.noise_scale = noise_scale
        self.nbsym = nbsym
        self.max_imf = max_imf
        self.range_thr = range_thr
        self.total_power_thr = total_power_thr
        self.noise_kind = noise_kind
        self.seed = seed
        self.spline_kind = spline_kind
        self.extrema_detection = extrema_detection
        self.parallel = parallel
        self.processes = processes

    def decompose(self, S, T=None) -> DecompositionResult:
        """
        Decompose the signal into IMFs and a residual.
        """
        from PyEMD import CEEMDAN as PyEMD_CEEMDAN

        S, T, _N = Check_Time_and_Signal(S, T, ndim={1}, method=self.name)

        effective_seed, _ = resolve_seed(self.seed, self.name)

        decomposer = PyEMD_CEEMDAN(
            trials=self.trials,
            seed=effective_seed,
            spline_kind=self.spline_kind,
            nbsym=self.nbsym,
            extrema_detection=self.extrema_detection,
            parallel=self.parallel,
            processes=self.processes,
            noise_scale=self.noise_scale,
            noise_kind=self.noise_kind,
            range_thr=self.range_thr,
            total_power_thr=self.total_power_thr,
        )

        IMF_Residue = decomposer.ceemdan(S, T, self.max_imf)

        IMFs = IMF_Residue[:-1, :]  # shape [n_imfs, len(S)]
        Res = IMF_Residue[-1, :]

        return DecompositionResult(
            IMFs,
            Res,
            {},
            CEEMDANConfig(
                trials=self.trials,
                noise_scale=self.noise_scale,
                nbsym=self.nbsym,
                max_imf=self.max_imf,
                range_thr=self.range_thr,
                total_power_thr=self.total_power_thr,
                noise_kind=self.noise_kind,
                seed=effective_seed,
                spline_kind=self.spline_kind,
                extrema_detection=self.extrema_detection,
                parallel=self.parallel,
                processes=self.processes,
            ),
        )
