"""
Improved Complete Ensemble Empirical Mode Decomposition with Adaptive Noise

Pre-decomposes a pool of white-noise realizations and estimates the local
means of noise-assisted copies of the residual.

References
----------
10.1007/s10470-021-01901-3
"""

import logging

import numpy as np
from dataclasses import dataclass
from typing import ClassVar

from .Base import Config, Decomposer, DecompositionResult
from ._Registry import register_class
from .EMD import EMD
from .Utils import Check_Time_and_Signal, is_monotonic, is_uniform, resolve_seed

__all__ = ["ICEEMDAN", "ICEEMDANConfig"]

logger = logging.getLogger(__name__)


@dataclass(frozen=True, kw_only=True)
class ICEEMDANConfig(Config):
    """
    Effective parameters of an ICEEMDAN run.
    """
    ensemble_size: int
    epsilon_0: float
    max_imfs: int
    spline_kind: str
    nbsym: int
    seed: int | None


@register_class("ICEEMDAN")
class ICEEMDAN(Decomposer):
    name: ClassVar[str] = "ICEEMDAN"

    def __init__(
        self,
        ensemble_size: int = 300,
        epsilon_0: float = 0.2,
        max_imfs: int | None = None,
        spline_kind: str = "cubic",
        nbsym: int = 2,
        seed: int | None = None,
    ):
        """
        Parameters
        ----------
        ensemble_size : int
            Number of white-noise realizations.
        epsilon_0 : float
            Base noise amplitude.
        max_imfs : int | None
            Maximum number of IMFs; None selects log2(N) + 5.
        spline_kind : str
            Interpolation kind for the internal EMD.
        nbsym : int
            Number of mirrored extrema for the internal EMD.
        seed : int | None
            Local random seed. Overridden by a global seed when both are set.
        """
        self.ensemble_size = ensemble_size
        self.epsilon_0 = epsilon_0
        self.max_imfs = max_imfs
        self.spline_kind = spline_kind
        self.nbsym = nbsym
        self.seed = seed

    def decompose(self, S, T=None) -> DecompositionResult:
        """
        Decompose the signal into IMFs and a residual.
        """
        S, T, _N = Check_Time_and_Signal(S, T, ndim={1}, method=self.name)

        S = np.asarray(S, dtype=np.float64)
        if S.ndim != 1:
            S = S.ravel()

        n_samples = len(S)
        if n_samples < 4:
            raise ValueError("Signal length must be at least 4")

        if not is_uniform(T):
            raise ValueError("Time axis must be uniformly sampled")

        max_imfs = self.max_imfs
        if max_imfs is None:
            max_imfs = int(np.log2(n_samples)) + 5
        elif max_imfs <= 0:
            raise ValueError(f"max_imfs must be positive, got {max_imfs}")

        if self.ensemble_size < 1:
            raise ValueError(f"ensemble_size must be >= 1, got {self.ensemble_size}")

        if self.epsilon_0 <= 0:
            raise ValueError(f"epsilon_0 must be > 0, got {self.epsilon_0}")

        effective_seed, _ = resolve_seed(self.seed, self.name)
        rng = np.random.default_rng(effective_seed)

        logger.debug("ICEEMDAN: N=%d, ensemble=%d, epsilon_0=%s, max_imfs=%d",
                     n_samples, self.ensemble_size, self.epsilon_0, max_imfs)

        # Generate fixed pool of white noise
        white_noise = rng.standard_normal((self.ensemble_size, n_samples))

        # Pre-decompose all noise realizations
        noise_imfs_list = []  # List of 2D arrays: (K, n_samples) for each noise
        valid_indices = []

        for i in range(self.ensemble_size):
            try:
                imfs = EMD(
                    spline_kind=self.spline_kind,
                    nbsym=self.nbsym,
                    max_imf=max_imfs + 5,  # Decompose 5 more IMFs than needed
                ).decompose(white_noise[i], T).IMFs

                if imfs.shape[0] > 0:
                    noise_imfs_list.append(imfs)
                    valid_indices.append(i)
            except Exception as e:
                logger.debug("ICEEMDAN noise %d decomposition failed: %s", i, e)
                continue

        # Valid noise count
        M = len(valid_indices)
        if M == 0:
            raise RuntimeError("All noise decompositions failed")

        logger.debug("ICEEMDAN valid noise realizations: %d/%d", M, self.ensemble_size)

        # Determine common number of IMFs across all noise realizations
        min_noise_imfs = min(imfs.shape[0] for imfs in noise_imfs_list)
        K = min(min_noise_imfs, max_imfs)  # Actual maximum number of IMFs to extract

        if K == 0:
            raise RuntimeError("No valid IMFs found in noise decompositions")

        # Compute noise IMF standard deviations (global std, as per paper)
        noise_std = np.zeros(K, dtype=np.float64)
        for k in range(K):
            # Stack all k-th IMFs from all noise realizations
            kth_imfs = np.stack([imfs[k] for imfs in noise_imfs_list])
            # Compute global standard deviation (across all realizations and samples)
            noise_std[k] = np.std(kth_imfs)

        residual = S.copy()
        imfs_list = []
        signal_energy = np.sum(S ** 2)
        if signal_energy < 1e-12:
            signal_energy = 1e-12  # Prevent division by zero

        for k in range(K):
            # Noise scaling factor: beta_k = epsilon_0 * std(r_{k-1}) / std(E_k(omega))
            beta = self.epsilon_0 * np.std(residual) / (noise_std[k] + 1e-12)
            beta = np.clip(beta, 1e-12, 1e12)  # Numerical stability

            # Pre-allocate array for local means
            local_means = np.zeros((M, n_samples), dtype=np.float64)
            valid_count = 0

            # Ensemble loop
            for i, noise_imfs in enumerate(noise_imfs_list):
                try:
                    # Noisy signal: r_{k-1} + beta_k * E_k(omega^(i))
                    noisy_signal = residual + beta * noise_imfs[k]

                    # Apply E operator: decompose and get first IMF
                    sig_imfs = EMD(
                        spline_kind=self.spline_kind,
                        nbsym=self.nbsym,
                        max_imf=1,  # Only need the first IMF
                    ).decompose(noisy_signal, T).IMFs

                    if sig_imfs.shape[0] > 0:
                        # Local mean: M(.) = S - E_1(.)
                        local_means[valid_count] = noisy_signal - sig_imfs[0, :]
                        valid_count += 1

                except Exception as e:
                    logger.debug("ICEEMDAN IMF %d noise %d failed: %s", k + 1, i, e)
                    continue

            if valid_count == 0:
                logger.debug("ICEEMDAN no valid realizations for IMF %d, stopping", k + 1)
                break

            # Compute ensemble average of local means
            r_k = np.mean(local_means[:valid_count], axis=0)

            # Extract IMF: IMF_k = r_{k-1} - r_k
            imf_k = residual - r_k

            # IMF validity
            imf_energy = np.sum(imf_k ** 2)
            if imf_energy < 1e-12 * signal_energy:
                logger.debug("ICEEMDAN IMF %d has negligible energy, stopping", k + 1)
                break

            imfs_list.append(imf_k)

            # Update residual: r_k
            residual = r_k

            # Stopping criteria (original paper conditions)
            if is_monotonic(residual):
                logger.debug("ICEEMDAN residual is monotonic, stopping")
                break

            if len(imfs_list) >= max_imfs:
                logger.debug("ICEEMDAN reached maximum IMF count, stopping")
                break

        if not imfs_list:
            logger.debug("ICEEMDAN no IMFs were extracted")
            return DecompositionResult(
                np.empty((0, n_samples), dtype=np.float64),
                residual,
                {},
                ICEEMDANConfig(
                    ensemble_size=self.ensemble_size,
                    epsilon_0=self.epsilon_0,
                    max_imfs=max_imfs,
                    spline_kind=self.spline_kind,
                    nbsym=self.nbsym,
                    seed=effective_seed,
                ),
            )

        imfs_array = np.vstack(imfs_list)

        return DecompositionResult(
            imfs_array,
            residual,
            {},
            ICEEMDANConfig(
                ensemble_size=self.ensemble_size,
                epsilon_0=self.epsilon_0,
                max_imfs=max_imfs,
                spline_kind=self.spline_kind,
                nbsym=self.nbsym,
                seed=effective_seed,
            ),
        )
