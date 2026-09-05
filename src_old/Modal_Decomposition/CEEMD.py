"""
Python version:  (must)
    3.10.11

Lib and Version:  (if None write None)
    numpy - 2.2.6
	time - 1.39.2
	tqdm - 4.67.3

Only accessed by:  (must)
    Only __init__.py

Description: (if None write None)
    Realize the CEEMD.

Modify:  (must)
    2026.3.25 - Create
    2026.4.2  - Finish the Optimization of the CEEMD.
    2026.5.1  - Fix the error of the decomposition.
    2026.7.8  - Reconstruct the method with abstract class.
"""

from typing import Union, Tuple, Optional
import numpy as np

from .Base import Decomposer, Name, Reference, DecompositionResult, Config, CEEMDConfig
from ._Registry import register_class, register_function
from .EMD import emd
from .Utils import monotonic, Check_ST_and_Transform
from warnings import warn


@register_class("CEEMD")
class CEEMD(Decomposer):
    name = "CEEMD"

    def __init__(self, N_whitenoise: int, beta: float, T: list | np.ndarray, max_imf: Optional[int] = None, dead_line: int = 10, dim: int = 1, RAISE: bool = True):
        """
        CEEMD: Complementary Ensemble Empirical Mode Decomposition

        :param S: Signal (1-dim)
        :param T: the time axis.
        :param N_whitenoise: the num of the added whitenoise.
        :param beta:
        :param max_imf: -1, None or other int | -1 means decompose completely, None means give a int auto, other int means the num of the IMFs
        :param dead_line: Sometime it'll be in unuseful cycle, when the average of the N's sequence with added whitenoise is empty([]). It'll be forced exit when the time of the cycle above the deadline.
        :return: IMFs (n_IMFs, N), Res (N,), Info(iter epoch), Config(N_whitenoise, beta)
        """

        self.N_whitenoise = N_whitenoise
        self.beta = beta
        self.T = T
        self.max_imf = max_imf
        self.dead_line = dead_line
        self.dim = dim
        self.RAISE = RAISE
        self.config: Config = None

        if beta <= 0:
            raise ValueError("The beta should > 0")

        if N_whitenoise <= 0 or not isinstance(N_whitenoise, int):
            raise TypeError("N_whitenoise must be int type or > 0")

    def decompose(self, S: list | np.ndarray, **kwargs) -> DecompositionResult:
        try:
            from scipy.signal import argrelextrema
        except ImportError:
            raise ModuleNotFoundError("Scipy module not available")

        uniform, DimSure, S, T, N = Check_ST_and_Transform(S, self.T, self.dim, self.RAISE)

        if self.max_imf is None:
            self.max_imf = int(np.log2(len(S))) + 2

        imfs = []
        residual = S.copy()
        k = 0
        dead_cycle = 0
        epoch = 0

        while True:
            if self.max_imf != -1 and k >= self.max_imf:
                break

            imf_candidates = []
            std_dev = np.std(residual)

            for i in range(self.N_whitenoise):
                white_noise = np.random.normal(0, std_dev * self.beta, N)
                S_plus = residual + white_noise
                S_minus = residual - white_noise

                imfs_plus, _, _ = emd(S_plus, T, max_imf=1)
                imfs_minus, _, _ = emd(S_minus, T, max_imf=1)

                if len(imfs_plus) > 0 and len(imfs_minus) > 0:
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

        self.config = CEEMDConfig(self.N_whitenoise, self.beta, self.max_imf, self.dead_line)
        return DecompositionResult \
        (
            np.array(imfs), np.array(residual),
            {
                "epoch": epoch,
                "dim_sure": DimSure,
                "uniform": uniform,
            },
            self.config
        )

@register_function("fast_CEEMD")
def fast_CEEMD(S: list | np.ndarray, T: list | np.ndarray = None, N_whitenoise=37, beta=0.2, max_imf: Optional[int] = None, dead_line: int = 10, dim: int = 1, RAISE: bool = True):
    ceemd = CEEMD(N_whitenoise=N_whitenoise, beta=beta, max_imf=max_imf, dead_line=dead_line, dim=dim, RAISE=RAISE)
    return ceemd.decompose(S)