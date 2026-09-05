"""
Python version:  (must)
    3.10.11

Lib and Version:  (if None write None)
    numpy - 2.2.6

Only accessed by:  (must)
    Only __init__.py

Description: (if None write None)
    Realize the CEEMDAN.

Modify:  (must)
    2026.3.25 - Create.
"""

from typing import Tuple, Literal
import numpy as np

from .Utils import Check_ST_and_Transform
from .Base import Decomposer, DecompositionResult, Config, CEEMDANConfig
from ._Registry import register_class, register_function


@register_class("CEEMDAN")
class CEEMDAN(Decomposer):
    name = "CEEMDAN"

    def __init__ \
    (
        self,
        trials: int,
        noise_scale: int | float,
        nbsym: int,
        max_imf: int = -1,
        range_thr: float = 0.01,
        total_power_thr: float = 0.05,
        noise_kind: Literal["normal", "uniform"] = "normal",
        noise_seed: int = 42,
        spline_kind: Literal["akima", "cubic", "pchip", "cubic_hermite", "slinear", "quadratic", "linear"] = "cubic",
        extrema_detection: Literal["simple", "parabol"] = "simple",
        T: list | np.ndarray = None,
        parallel: bool = False,
        processes: int = None,
        dim: int = 1,
        RAISE: bool = True
    ):
        """
        CEEMDAN: Complete Ensemble Empirical Mode Decomposition with Adaptive Noise

        :param S: Signal (1-dim)
        :param T: Time axis (1-dim). Default uniform, or input the Unix.
        :param max_imf: the num of the decomposed IMFs. | -1 means all.
        """

        self.trials = trials
        self.noise_scale = noise_scale
        self.nbsym = nbsym
        self.max_imf = max_imf
        self.range_thr = range_thr
        self.total_power_thr = total_power_thr
        self.noise_kind = noise_kind
        self.noise_seed = noise_seed
        self.spline_kind = spline_kind
        self.extrema_detection = extrema_detection
        self.noise_scale = noise_scale
        self.noise_kind = noise_kind
        self.T = T
        self.parallel = parallel
        self.processes = processes
        self.dim = dim
        self.RAISE = RAISE

        self.config: Config = None

    def decompose(self, S: list | np.ndarray, **kwargs) -> DecompositionResult:
        try:
            from PyEMD import CEEMDAN as decomposer
        except ImportError:
            raise ModuleNotFoundError("PyEMD module not available")

        uniform, DimSure, S, T, N = Check_ST_and_Transform(S, self.T, self.dim, self.RAISE)

        decomposer = decomposer \
        (
            trials=self.trials,
            seed=self.noise_seed,
            spline_kind=self.spline_kind,
            nbsym=self.nbsym,
            extrema_detection=self.extrema_detection,
            parallel=self.parallel,
            processes=self.processes,
            noise_scale=self.noise_scale,
            noise_kind=self.noise_kind,
            range_thr=self.range_thr,
            total_power_thr=self.total_power_thr
        )

        IMF_Residue = decomposer.ceemdan(S, T, self.max_imf)

        IMFs = IMF_Residue[:-1, :]  # shape [n_imfs, len(S)]
        Res = IMF_Residue[-1, :]

        self.config = CEEMDANConfig \
        (
            self.trials,
            self.noise_scale,
            self.nbsym,
            self.max_imf,
            self.range_thr,
            self.total_power_thr,
            self.noise_kind,
            self.noise_seed,
            self.spline_kind,
            self.extrema_detection
        )

        Info = \
        {
            "parallel": self.parallel,
            "processes": self.processes,
            "dim_sure": DimSure,
            "uniform": uniform,
        }

        return DecompositionResult(IMFs, Res, Info, self.config)

@register_function("fast_CEEMDAN")
def fast_CEEMDAN(
    S: list | np.ndarray,
    T: list | np.ndarray = None,
    trials: int = 100,
    nbsym: int = 2,
    noise_scale: float = 1.0,
    range_thr: float = 0.01,
    total_power_thr: float = 0.05,
    noise_seed: int = 42,
    spline_kind: Literal["akima", "cubic", "pchip", "cubic_hermite", "slinear", "quadratic", "linear"] = "cubic",
    extrema_detection: Literal["simple", "parabol"] = "simple",
    parallel: bool = False,
    processes: int = None,
    noise_kind: Literal["normal", "uniform"] = "normal",
    dim: int = 1,
    RAISE: bool = True
):
    decomposer = CEEMDAN \
    (
        trials=trials,
        seed=noise_seed,
        spline_kind=spline_kind,
        nbsym=nbsym,
        extrema_detection=extrema_detection,
        parallel=parallel,
        processes=processes,
        noise_scale=noise_scale,
        noise_kind=noise_kind,
        range_thr=range_thr,
        total_power_thr=total_power_thr,
        dim=dim,
        RAISE=RAISE,
        T=T
    )

    return  decomposer.decompose(S)