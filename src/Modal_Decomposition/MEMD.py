"""
Multivariate Empirical Mode Decomposition

Projects a multichannel signal onto direction vectors sampled on a
hypersphere and averages the per-direction envelopes.

References
----------
10.48550/arXiv.2206.00926
"""

import numpy as np
from dataclasses import dataclass
from typing import ClassVar

from .Base import Config, Decomposer, DecompositionResult
from ._Registry import register_class
from .Utils import Check_Time_and_Signal, monotonic

__all__ = ["MEMD", "MEMDConfig"]


@dataclass(frozen=True, kw_only=True)
class MEMDConfig(Config):
    """
    Effective parameters of a MEMD run.
    """
    d: int
    k: int
    max_imf: int
    sd_thresh: float
    max_iter: int
    spline_kind: str


@register_class("MEMD")
class MEMD(Decomposer):
    name: ClassVar[str] = "MEMD"

    def __init__(
        self,
        d: int | None = None,
        k: int | None = None,
        max_imf: int | None = None,
        sd_thresh: float = 0.2,
        max_iter: int = 10,
        spline_kind: str = "linear",
    ):
        """
        Parameters
        ----------
        d : int | None
            Number of channels; None infers it from the signal.
        k : int | None
            Number of directional vectors; None selects d * 128.
        max_imf : int | None
            Maximum number of IMFs; None selects log2(N).
        sd_thresh : float
            Sifting stopping threshold.
        max_iter : int
            Maximum sifting iterations per IMF.
        spline_kind : str
            Interpolation kind for envelope fitting.
        """
        self.d = d
        self.k = k
        self.max_imf = max_imf
        self.sd_thresh = sd_thresh
        self.max_iter = max_iter
        self.spline_kind = spline_kind

    def decompose(self, S, T=None) -> DecompositionResult:
        """
        Decompose the multichannel signal into IMFs and a residual.

        The time axis is validated but not used by the algorithm.
        """
        S, T, _N = Check_Time_and_Signal(S, T, ndim={1, 2}, method=self.name)

        S = np.asarray(S, dtype=np.float64)
        if S.ndim == 1:
            S = S.reshape(1, -1)
        d_infer, N = S.shape

        d = d_infer if self.d is None else self.d
        if d != d_infer:
            raise ValueError(f"Declared d={d} but signal has shape {S.shape}")

        k = d * 128 if self.k is None else self.k
        max_imf = int(np.log2(N)) if self.max_imf is None else self.max_imf

        T_axis = np.arange(N, dtype=np.float64)

        vectors = _generate_hammersley_points(k, d)  # shape: (d, k)

        imfs = []
        residue = S.copy()

        for imf_idx in range(max_imf):
            h = residue.copy()

            for iter_num in range(self.max_iter):
                h_old = h
                mean_envelope = _compute_local_mean(h, vectors, T_axis, spline_kind=self.spline_kind)

                h_new = h - mean_envelope
                sd = np.sum((h_old - h_new) ** 2) / np.sum(h_old ** 2)
                h = h_new

                if sd < self.sd_thresh or iter_num == self.max_iter - 1:
                    break

            imfs.append(h.copy())

            residue = residue - h

            if _should_stop(residue):
                break

        imfs_array = np.array(imfs)  # shape: (n_imfs, d, N)
        if imfs_array.shape[0] == 0:
            imfs_array = np.empty((0, d, N), dtype=np.float64)

        return DecompositionResult(
            imfs_array,
            residue,
            {},
            MEMDConfig(
                d=d,
                k=k,
                max_imf=max_imf,
                sd_thresh=self.sd_thresh,
                max_iter=self.max_iter,
                spline_kind=self.spline_kind,
            ),
        )


def _generate_hammersley_points(k, d):
    vectors = np.zeros((d, k))

    primes = _generate_primes(d - 1)

    for i in range(k):
        point = np.zeros(d)

        point[0] = _radical_inverse_vdc(i)

        for j in range(1, d):
            base = primes[j - 1] if j - 1 < len(primes) else primes[-1]
            point[j] = _radical_inverse(i, base)

        norm = np.linalg.norm(point)
        if norm > 0:
            point = point / norm

        vectors[:, i] = point

    return vectors


def _radical_inverse_vdc(index):
    """Van der Corput."""
    bits = index
    bits = (bits << 16) | (bits >> 16)
    bits = ((bits & 0x55555555) << 1) | ((bits & 0xAAAAAAAA) >> 1)
    bits = ((bits & 0x33333333) << 2) | ((bits & 0xCCCCCCCC) >> 2)
    bits = ((bits & 0x0F0F0F0F) << 4) | ((bits & 0xF0F0F0F0) >> 4)
    bits = ((bits & 0x00FF00FF) << 8) | ((bits & 0xFF00FF00) >> 8)
    return float(bits) / 2 ** 32


def _radical_inverse(index, base):
    result = 0.0
    f = 1.0 / base
    i = index
    while i > 0:
        result += f * (i % base)
        i = i // base
        f = f / base
    return result


def _generate_primes(n):
    if n <= 0:
        return []

    primes = []
    num = 2
    while len(primes) < n:
        is_prime = True
        for p in primes:
            if p * p > num:
                break
            if num % p == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(num)
        num += 1

    return primes


def _compute_local_mean(signal, vectors, T, spline_kind: str = "linear"):
    """
    signal: now (d, N)
    vectors: directional vectors (d, k)
    T: time axis
    """
    from scipy.signal import find_peaks
    from scipy.interpolate import interp1d

    d, N = signal.shape
    k = vectors.shape[1]

    current_projections = np.dot(signal.T, vectors)

    mean_envelope = np.zeros((d, N))

    for dir_idx in range(k):
        proj = current_projections[:, dir_idx]

        max_indices, _ = find_peaks(proj)
        min_indices, _ = find_peaks(-proj)

        if len(max_indices) == 0:
            max_indices = np.array([0, N - 1])
        else:
            if max_indices[0] != 0:
                max_indices = np.concatenate([[0], max_indices])
            if max_indices[-1] != N - 1:
                max_indices = np.concatenate([max_indices, [N - 1]])

        if len(min_indices) == 0:
            min_indices = np.array([0, N - 1])
        else:
            if min_indices[0] != 0:
                min_indices = np.concatenate([[0], min_indices])
            if min_indices[-1] != N - 1:
                min_indices = np.concatenate([min_indices, [N - 1]])

        for ch in range(d):
            max_values = signal[ch, max_indices]
            if len(max_indices) >= 4:
                try:
                    max_spline = interp1d(max_indices, max_values, kind=spline_kind, fill_value='extrapolate')
                    max_env = max_spline(T)
                except Exception:
                    max_env = np.interp(T, max_indices, max_values)
            else:
                max_env = np.interp(T, max_indices, max_values)

            min_values = signal[ch, min_indices]
            if len(min_indices) >= 4:
                try:
                    min_spline = interp1d(min_indices, min_values, kind=spline_kind, fill_value='extrapolate')
                    min_env = min_spline(T)
                except Exception:
                    min_env = np.interp(T, min_indices, min_values)
            else:
                min_env = np.interp(T, min_indices, min_values)

            mean_envelope[ch] += (max_env + min_env) / 2

    mean_envelope = mean_envelope / k

    return mean_envelope


def _should_stop(residue):
    from scipy.signal import find_peaks

    d, N = residue.shape

    for ch in range(d):
        if monotonic(residue[ch]):
            return True

        max_indices, _ = find_peaks(residue[ch])
        min_indices, _ = find_peaks(-residue[ch])
        if len(max_indices) + len(min_indices) <= 2:
            return True

    return False
