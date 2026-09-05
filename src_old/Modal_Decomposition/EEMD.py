"""
Python version:  (must)
    3.10.11

Lib and Version:  (if None write None)
    numpy - 2.2.6

Only accessed by:  (must)
    Only __init__.py

Description: (if None write None)
    Realize the EEMD.

Modify:  (must)
    2026.3.25 - Create.
    2026.3.27 - Change the EEMD class' usage. EEMD(parallel=True) (default) -> EEMD(parallel=False). Now, it will not use parallel default.
    2026.4.2  - Finish the Optimization of the EEMD. Correct the logic.
    2026.5.1  - Fix: use PyEMD.EEMD to ensure standard algorithm.
    2026.7.8  - Reconstruct the EEMD method with abstract class.
"""

import numpy as np
from typing import Tuple, Optional

from .Base import Decomposer, Config, EEMDConfig, DecompositionResult
from .Utils import Check_ST_and_Transform
from ._Registry import register_function, register_class


@register_class("EEMD")
class EEMD(Decomposer):
    name = "EEMD"

    def __init__(self, trials: int = 100, noise_width: float = 0.05, max_imf: int = -1, T: list | np.ndarray = None, noise_seed: int = 42, parallel: bool = False, ext_EMD: Decomposer | object = None, dim: int = 1, RAISE: bool = True, **kwargs):
        self.trials = trials
        self.noise_width = noise_width
        self.max_imf = max_imf
        self.parallel = parallel
        self.T = T
        self.noise_seed = noise_seed
        self.kwargs = kwargs
        self.dim = dim
        self.RAISE = RAISE

        self.config: Config = None

        if ext_EMD is None:
            try:
                from PyEMD import EMD
            except ImportError:
                raise ModuleNotFoundError("PyEMD module not available")
            self.ext_EMD = EMD(**kwargs)
        else:
            self.ext_EMD = ext_EMD

    def decompose(self, S: list | np.ndarray, **kwargs) -> DecompositionResult:
        try:
            from PyEMD import EEMD as decomposer
        except ImportError:
            raise ModuleNotFoundError("PyEMD module not available")

        uniform, DimSure, S, T, N = Check_ST_and_Transform(S, self.T, self.dim, self.RAISE)

        decomposer = decomposer(trials=self.trials, noise_width=self.noise_width, ext_EMD=self.ext_EMD, parallel=self.parallel, **kwargs)
        decomposer.noise_seed(self.noise_seed)

        result = decomposer.eemd(S, T, max_imf=self.max_imf)

        IMFs = result[:-1, :]  # shape [n_imfs, N]
        Res = result[-1, :]  # shape [N,]

        Info = {"parallel": self.parallel}
        self.config = EEMDConfig(self.trials, self.noise_width, self.max_imf, self.noise_seed)

        return DecompositionResult(IMFs, Res, Info, self.config)

@register_function("fast_EEMD")
def fast_EEMD(S: list | np.ndarray, T: list | np.ndarray, trials: int = 100, noise_width: float = 0.05, max_imf: int = -1, noise_seed: int = 42, parallel: bool = False, **kwargs):
    decomposer = EEMD(T=T, trials=trials, noise_width=noise_width, max_imf=max_imf, noise_seed=noise_seed, parallel=parallel, **kwargs)
    return decomposer.decompose(S)