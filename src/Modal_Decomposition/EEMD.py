"""
Ensemble Empirical Mode Decomposition

Averages EMD results over white-noise perturbed copies of the signal,
implemented by PyEMD.

References
----------
10.1142/S1793536909000047
"""

from dataclasses import dataclass
from typing import ClassVar

from .Base import Config, Decomposer, DecompositionResult
from ._Registry import register_class
from .Utils import Check_Time_and_Signal, resolve_seed

__all__ = ["EEMD", "EEMDConfig"]


@dataclass(frozen=True, kw_only=True)
class EEMDConfig(Config):
    """
    Effective parameters of an EEMD run.

    Note: a custom ``ext_EMD`` spline_kind object is not snapshotted.
    """
    trials: int
    noise_width: float
    max_imf: int
    seed: int | None
    parallel: bool


@register_class("EEMD")
class EEMD(Decomposer):
    name: ClassVar[str] = "EEMD"

    def __init__(
        self,
        trials: int = 100,
        noise_width: float = 0.05,
        max_imf: int = -1,
        seed: int | None = None,
        parallel: bool = False,
        ext_EMD: object | None = None,
        **ext_emd_kwargs,
    ):
        """
        Parameters
        ----------
        trials : int
            Number of ensemble realizations.
        noise_width : float
            Amplitude of the added noise relative to the signal range.
        max_imf : int
            Maximum number of IMFs; -1 decomposes completely.
        seed : int | None
            Local random seed. Overridden by a global seed when both are set.
        parallel : bool
            Enable multiprocessing.
        ext_EMD : object, optional
            Custom EMD spline_kind; must expose ``emd(S, T, max_imf)``.
        **ext_emd_kwargs
            Additional keyword arguments forwarded to the internal PyEMD EMD
            object when ``ext_EMD`` is None.
        """
        self.trials = trials
        self.noise_width = noise_width
        self.max_imf = max_imf
        self.seed = seed
        self.parallel = parallel
        self.ext_emd_kwargs = ext_emd_kwargs

        if ext_EMD is None:
            from PyEMD import EMD as PyEMD_EMD

            self.ext_EMD = PyEMD_EMD(**ext_emd_kwargs)
        else:
            self.ext_EMD = ext_EMD

    def decompose(self, S, T=None) -> DecompositionResult:
        """
        Decompose the signal into IMFs and a residual.
        """
        from PyEMD import EEMD as PyEMD_EEMD

        S, T, _N = Check_Time_and_Signal(S, T, ndim={1}, method=self.name)

        effective_seed, _ = resolve_seed(self.seed, self.name)

        decomposer = PyEMD_EEMD(
            trials=self.trials,
            noise_width=self.noise_width,
            ext_EMD=self.ext_EMD,
            parallel=self.parallel,
        )
        decomposer.noise_seed(effective_seed)

        result = decomposer.eemd(S, T, max_imf=self.max_imf)

        IMFs = result[:-1, :]  # shape [n_imfs, N]
        Res = result[-1, :]  # shape [N,]

        return DecompositionResult(
            IMFs,
            Res,
            {},
            EEMDConfig(
                trials=self.trials,
                noise_width=self.noise_width,
                max_imf=self.max_imf,
                seed=effective_seed,
                parallel=self.parallel,
            ),
        )
