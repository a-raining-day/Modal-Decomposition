"""
Local Mean Decomposition

Extracts product functions by iteratively smoothing the local mean and
envelope estimate from extrema interpolation.

References
----------
10.1098/rsif.2005.0058
"""

import numpy as np
from dataclasses import dataclass
from typing import ClassVar

from .Base import Config, Decomposer, DecompositionResult
from ._Registry import register_class
from .Utils import Check_Time_and_Signal, is_monotonic

__all__ = ["LMD", "LMDConfig"]


@dataclass(frozen=True, kw_only=True)
class LMDConfig(Config):
    """
    Effective parameters of an LMD run.
    """
    max_pf: int
    max_iter: int
    eps: float
    eps_stable: float
    min_amp: float
    max_amp: float
    converge_mean: float
    smooth_window: int


@register_class("LMD")
class LMD(Decomposer):
    name: ClassVar[str] = "LMD"

    def __init__(
        self,
        max_pf: int | None = None,
        max_iter: int = 37,
        eps: float = 0.05,
        eps_stable: float = 1e-12,
        min_amp: float = 1e-12,
        max_amp: float = 1e12,
        converge_mean: float = 1e-3,
        smooth_window: int = 5,
    ):
        """
        Parameters
        ----------
        max_pf : int | None
            Maximum number of product functions. None selects log2(N), -1
            decomposes completely.
        max_iter : int
            Maximum sifting iterations per product function.
        eps : float
            Envelope convergence threshold.
        eps_stable : float
            Minimum energy of an accepted product function.
        min_amp : float
            Lower bound of the amplitude estimate.
        max_amp : float
            Upper bound of the amplitude estimate.
        converge_mean : float
            Mean convergence threshold.
        smooth_window : int
            Savitzky-Golay window length for the amplitude estimate.
        """
        self.max_pf = max_pf
        self.max_iter = max_iter
        self.eps = eps
        self.eps_stable = eps_stable
        self.min_amp = min_amp
        self.max_amp = max_amp
        self.converge_mean = converge_mean
        self.smooth_window = smooth_window

    def decompose(self, S, T=None) -> DecompositionResult:
        """
        Decompose the signal into product functions and a residual.

        The time axis is validated but not used by the algorithm.
        """
        from scipy.signal import argrelextrema, savgol_filter

        S, T, _N = Check_Time_and_Signal(S, T, ndim={1}, method=self.name)

        S = np.asarray(S, dtype=np.float64).ravel()
        n_samples = S.size

        if n_samples < 8:
            raise ValueError("LMD requires S length >= 8 (Smith 2005 standard)")

        t = np.arange(n_samples, dtype=np.float64)

        max_pf = self.max_pf
        if max_pf is None:
            max_pf = int(np.log2(n_samples))

        residue = S.copy()
        PFs: list[np.ndarray] = []

        while max_pf == -1 or len(PFs) < max_pf:
            h = residue.copy()
            a_total = np.ones(n_samples, dtype=np.float64)
            converged = False

            for _ in range(self.max_iter):
                max_loc = argrelextrema(h, np.greater)[0]
                min_loc = argrelextrema(h, np.less)[0]
                ext_idx = np.unique(np.concatenate([max_loc, min_loc]))

                if len(ext_idx) < 3:
                    break

                ext_idx, ext_vals = _mirror_extend_real(h, ext_idx, n_samples)
                ext_idx = np.sort(ext_idx)

                t_mid = (ext_idx[:-1] + ext_idx[1:]) / 2.0
                m_vals = (ext_vals[:-1] + ext_vals[1:]) / 2.0
                a_vals = np.abs(ext_vals[:-1] - ext_vals[1:]) / 2.0

                m_t = _safe_interpolate(t_mid, m_vals, t)
                a_t = _safe_interpolate(t_mid, a_vals, t)

                a_t = np.clip(a_t, self.min_amp, self.max_amp)
                a_t = savgol_filter(a_t, self.smooth_window, 2)

                if _check_convergence(m_t, a_t, self.eps, self.converge_mean):
                    converged = True
                    break

                s_new = (h - m_t) / a_t
                a_total = np.clip(a_total * a_t, self.min_amp, self.max_amp)
                h = s_new

            if not converged:
                break

            current_pf = np.clip(a_total * s_new, -self.max_amp, self.max_amp)

            if np.any(np.isnan(current_pf)) or np.any(np.isinf(current_pf)):
                break

            if np.sum(current_pf ** 2) < self.eps_stable:
                break

            PFs.append(current_pf)
            residue -= current_pf

            if (
                is_monotonic(residue) or
                np.sum(residue ** 2) < self.eps_stable or
                len(argrelextrema(residue, np.greater)[0]) + len(argrelextrema(residue, np.less)[0]) <= 2
            ):
                break

        IMFs = np.array(PFs, dtype=np.float64)
        if IMFs.shape[0] == 0:
            IMFs = np.empty((0, n_samples), dtype=np.float64)

        return DecompositionResult(
            IMFs,
            residue,
            {},
            LMDConfig(
                max_pf=max_pf,
                max_iter=self.max_iter,
                eps=self.eps,
                eps_stable=self.eps_stable,
                min_amp=self.min_amp,
                max_amp=self.max_amp,
                converge_mean=self.converge_mean,
                smooth_window=self.smooth_window,
            ),
        )


def _mirror_extend_real(signal: np.ndarray, ext_idx: np.ndarray, n_samples: int) -> tuple[np.ndarray, np.ndarray]:
    ext = ext_idx.copy()
    vals = signal[ext].copy()
    last_idx = n_samples - 1

    if ext[0] > 0:
        mirror_pos = 0
        mirror_val = 2 * signal[0] - vals[0]
        ext = np.insert(ext, 0, mirror_pos)
        vals = np.insert(vals, 0, mirror_val)

    if ext[-1] < last_idx:
        mirror_pos = last_idx
        mirror_val = 2 * signal[-1] - vals[-1]
        ext = np.append(ext, mirror_pos)
        vals = np.append(vals, mirror_val)

    return ext, vals


def _safe_interpolate(x: np.ndarray, y: np.ndarray, x_new: np.ndarray) -> np.ndarray:
    from scipy import interpolate

    try:
        interp = interpolate.CubicSpline(x, y, bc_type="natural")
        res = interp(x_new)
    except Exception:
        res = np.interp(x_new, x, y)

    left_mask = x_new < x[0]
    right_mask = x_new > x[-1]
    if np.any(left_mask):
        slope = (y[1] - y[0]) / (x[1] - x[0])
        res[left_mask] = y[0] + slope * (x_new[left_mask] - x[0])
    if np.any(right_mask):
        slope = (y[-1] - y[-2]) / (x[-1] - x[-2])
        res[right_mask] = y[-1] + slope * (x_new[right_mask] - x[-1])

    return res


def _check_convergence(m_t: np.ndarray, a_t: np.ndarray, eps: float, converge_mean) -> bool:
    mean_ok = np.max(np.abs(m_t)) < converge_mean
    envelope_ok = np.max(np.abs(a_t - 1.0)) < eps
    return mean_ok and envelope_ok
