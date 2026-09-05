"""
Complementary Ensemble Empirical Mode Decomposition

Adds white-noise pairs to the residual and averages first-order EMD
extractions to cancel the added noise.

References
----------
10.1016/j.jhydrol.2020.124647
"""

import numpy as np
from dataclasses import dataclass
from typing import ClassVar

from .Base import Config, Decomposer, DecompositionResult
from ._Registry import register_class
from .EMD import EMD
from .Utils import Check_Time_and_Signal, monotonic, resolve_seed

__all__ = ["CEEMD", "CEEMDConfig"]


@dataclass(frozen=True, kw_only=True)
class CEEMDConfig(Config):
    """
    Effective parameters of a CEEMD run.
    """
    N_whitenoise: int
    beta: float
    max_imf: int
    dead_line: int
    seed: int | None


@register_class("CEEMD")
class CEEMD(Decomposer):
    name: ClassVar[str] = "CEEMD"

    def __init__(
        self,
        N_whitenoise: int = 37,
        beta: float = 0.2,
        max_imf: int | None = None,
        dead_line: int = 10,
        seed: int | None = None,
    ):
        """
        Parameters
        ----------
        N_whitenoise : int
            Number of white-noise realizations per IMF.
        beta : float
            Noise amplitude relative to the residual standard deviation.
        max_imf : int | None
            Maximum number of IMFs; None selects an automatic value, -1
            decomposes completely.
        dead_line : int
            Consecutive empty-candidate rounds tolerated before raising.
        seed : int | None
            Local random seed. Overridden by a global seed when both are set.
        """
        self.N_whitenoise = N_whitenoise
        self.beta = beta
        self.max_imf = max_imf
        self.dead_line = dead_line
        self.seed = seed

        if beta <= 0:
            raise ValueError("The beta should > 0")

        if N_whitenoise <= 0 or not isinstance(N_whitenoise, int):
            raise TypeError("N_whitenoise must be int type or > 0")

    def decompose(self, S, T=None) -> DecompositionResult:
        """
        Decompose the signal into IMFs and a residual.
        """
        from scipy.signal import argrelextrema

        S, T, N = Check_Time_and_Signal(S, T, ndim={1}, method=self.name)

        effective_seed, _ = resolve_seed(self.seed, self.name)
        rng = np.random.default_rng(effective_seed)

        max_imf = self.max_imf
        if max_imf is None:
            max_imf = int(np.log2(len(S))) + 2

        imfs = []
        residual = S.copy()
        k = 0
        dead_cycle = 0
        epoch = 0

        while True:
            if max_imf != -1 and k >= max_imf:
                break

            imf_candidates = []
            std_dev = np.std(residual)

            for i in range(self.N_whitenoise):
                white_noise = rng.normal(0, std_dev * self.beta, N)
                S_plus = residual + white_noise
                S_minus = residual - white_noise

                imfs_plus = EMD(max_imf=1).decompose(S_plus, T).IMFs
                imfs_minus = EMD(max_imf=1).decompose(S_minus, T).IMFs

                if imfs_plus.shape[0] > 0 and imfs_minus.shape[0] > 0:
                    imf_candidate = (imfs_plus[0] + imfs_minus[0]) / 2.0
                    imf_candidates.append(imf_candidate)

            if not imf_candidates:
                dead_cycle += 1
                if dead_cycle >= self.dead_line:
                    raise RuntimeError("Trapped in a vicious cycle")
                continue

            imf = np.mean(imf_candidates, axis=0)
            imfs.append(imf)
            residual = residual - imf
            k += 1

            peaks = argrelextrema(residual, np.greater)[0]
            valleys = argrelextrema(residual, np.less)[0]

            if monotonic(residual, mod="monotonic") or len(peaks) + len(valleys) < 3:
                break

            epoch += 1

        IMFs = np.array(imfs)
        if IMFs.shape[0] == 0:
            IMFs = np.empty((0, N), dtype=np.float64)

        return DecompositionResult(
            IMFs,
            np.array(residual),
            {"epoch": epoch},
            CEEMDConfig(
                N_whitenoise=self.N_whitenoise,
                beta=self.beta,
                max_imf=max_imf,
                dead_line=self.dead_line,
                seed=effective_seed,
            ),
        )
