"""
Successive Variational Mode Decomposition

Extracts modes one after another by minimizing a variational criterion
against the residual spectrum.

References
----------
10.1016/j.sigpro.2020.107610
"""

import numpy as np
from dataclasses import dataclass
from typing import ClassVar, Literal

from .Base import Config, Decomposer, DecompositionResult
from ._Registry import register_class
from .Utils import Check_Time_and_Signal

__all__ = ["SVMD", "SVMDConfig"]


@dataclass(frozen=True, kw_only=True)
class SVMDConfig(Config):
    """
    Effective parameters of an SVMD run.
    """
    num_modes: int
    alpha: float
    tau: float
    tol: float
    max_iter: int
    backend: str


@register_class("SVMD")
class SVMD(Decomposer):
    name: ClassVar[str] = "SVMD"

    def __init__(
        self,
        num_modes: int = 3,
        alpha: float = 2000.0,
        tau: float = 0.0,
        tol: float = 1e-7,
        max_iter: int = 500,
        backend: Literal["numpy", "numba"] = "numpy",
    ):
        """
        Parameters
        ----------
        num_modes : int
            Number of modes.
        alpha : float
            Compactness constraint.
        tau : float
            Dual-ascent step.
        tol : float
            Convergence tolerance.
        max_iter : int
            Maximum number of iterations.
        backend : Literal["numpy", "numba"]
            Implementation backend. The numba backend is not yet
            implemented and raises RealizationError.
        """
        if backend not in ("numpy", "numba"):
            raise ValueError(f"backend must be 'numpy' or 'numba', got {backend!r}")

        self.num_modes = max(1, int(num_modes))
        self.alpha = float(alpha)
        self.tau = float(tau)
        self.tol = float(tol)
        self.max_iter = int(max_iter)
        self.backend = backend

        self._freqs = None
        self._signal_hat = None
        self._lambda_hat = None
        self._modes_hat = None

    def decompose(self, S, T=None) -> DecompositionResult:
        """
        Decompose the signal into modes and a residual.

        The time axis is validated but not used by the algorithm.
        """
        S, T, _N = Check_Time_and_Signal(S, T, ndim={1}, method=self.name)

        if self.backend == "numba":
            from ._SVMD_numba import numba_svmd

            modes, residual = numba_svmd(
                S,
                num_modes=self.num_modes,
                alpha=self.alpha,
                tau=self.tau,
                tol=self.tol,
                max_iter=self.max_iter,
            )

            return DecompositionResult(
                modes,
                residual,
                {},
                SVMDConfig(
                    num_modes=self.num_modes,
                    alpha=self.alpha,
                    tau=self.tau,
                    tol=self.tol,
                    max_iter=self.max_iter,
                    backend=self.backend,
                ),
            )

        from scipy.fft import fft, ifft, fftfreq

        signal = np.asarray(S, dtype=np.float64).ravel()
        N = signal.size
        if N < 8:
            raise ValueError("Signal length must be >= 8")

        K = self.num_modes
        eps = 1e-12

        if self._freqs is None or len(self._freqs) != N:
            self._freqs = fftfreq(N, d=1.0)

        self._signal_hat = fft(signal)

        if self._lambda_hat is None or len(self._lambda_hat) != N:
            self._lambda_hat = np.zeros(N, dtype=np.complex128)
        else:
            self._lambda_hat.fill(0.0)  # Reuse memory

        modes = np.zeros((K, N), dtype=np.float64)

        if self._modes_hat is None or len(self._modes_hat) != K or len(self._modes_hat[0]) != N:
            self._modes_hat = [np.zeros(N, dtype=np.complex128) for _ in range(K)]
        else:
            for mh in self._modes_hat:
                mh.fill(0.0)

        freqs_abs = np.abs(self._freqs)
        freq_range = freqs_abs.max() - freqs_abs.min()
        omega = freqs_abs.min() + (np.arange(0.1, 0.4, 0.4 / K)[:K] + 0.1) * freq_range

        denominators = []
        for k in range(K):
            diff = self._freqs - omega[k]
            denom = 1.0 + self.alpha * diff * diff
            denominators.append(denom)

        sum_modes_hat = np.zeros(N, dtype=np.complex128)
        recon_hat = np.zeros(N, dtype=np.complex128)

        prev_recon = np.zeros(N, dtype=np.float64)

        for iteration in range(self.max_iter):
            sum_modes_hat.fill(0.0)

            for k in range(K):
                numerator = self._signal_hat - (sum_modes_hat - self._modes_hat[k]) + 0.5 * self._lambda_hat

                self._modes_hat[k] = numerator / denominators[k]

                sum_modes_hat += self._modes_hat[k]

                power = np.abs(self._modes_hat[k])
                power_sq = power ** 2

                omega_num = np.sum(self._freqs * power_sq)
                omega_den = np.sum(power_sq) + eps

                omega[k] = omega_num / omega_den

                diff = self._freqs - omega[k]
                denominators[k] = 1.0 + self.alpha * diff * diff

            np.copyto(recon_hat, sum_modes_hat)

            self._lambda_hat += self.tau * (self._signal_hat - recon_hat)

            recon_time = np.real(ifft(recon_hat))

            error = np.linalg.norm(recon_time - prev_recon) / (np.linalg.norm(signal) + eps)

            if error < self.tol and iteration > 1:
                break

            np.copyto(prev_recon, recon_time)

        for k in range(K):
            modes[k] = np.real(ifft(self._modes_hat[k]))

        residual = signal - np.sum(modes, axis=0)

        return DecompositionResult(
            modes,
            residual,
            {},
            SVMDConfig(
                num_modes=self.num_modes,
                alpha=self.alpha,
                tau=self.tau,
                tol=self.tol,
                max_iter=self.max_iter,
                backend=self.backend,
            ),
        )
