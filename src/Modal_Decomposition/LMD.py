"""
Python version:  (must)
    3.10.11

Lib and Version:  (if None write None)
    numpy - 2.2.6
	scipy - 1.15.3

Only accessed by:  (must)
    Only __init__.py

Modify:  (must)
    2026.3.25

Description: (if None write None)
    Realize the LMD

Modify:
    2026.3.25 - Optimize the use of scipy
    2026.4.3  - Finish the Optimization of the LMD.
    2026.4.7  - LMD can support the decompose completely now.
    2026.4.9  - Change some hardcode parameters to args.
"""

import numpy as np
from .help_function import is_increasing
from typing import Union, Tuple


def lmd(
    S: Union[list, np.ndarray],
    max_pf: Union[int, None] = None,
    max_iter: int = 37,
    eps: float = 0.05,
    eps_stable: float = 1e-12,
    min_amp: float = 1e-12,
    max_amp: float = 1e12,
    converge_mean: float = 1e-3,
    smooth_window: int = 5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    LMD: Local mean decomposition

    :param S: Signal (1-dim)
    :param max_pf: the max num of the PFs. Default log2(N), -1 means decompose completely.
    :param max_iter: max iteration for each iter of decomposing the PFs. Default 37, according the paper.
    :param eps: Envelope convergence threshold
    :param eps_stable:
    :param min_amp:
    :param max_amp:
    :param converge_mean:
    :param smooth_window:

    Returns:
        PFs: (n_pf, N), Res: (N,)
    """
    from scipy.signal import argrelextrema, savgol_filter

    if S.ndim == 0:
        raise ValueError("The dim of the S must be 1-dim, not 0")

    elif S.ndim > 1:
        if 1 in S.shape:
            S = S.reshape(-1)

        else:
            raise ValueError(f"The dim of S must be 1-dim, not {S.ndim}")

    if not isinstance(S, np.ndarray):
        S = np.asarray(S, dtype=np.float64).ravel()
    n_samples = S.size

    if n_samples < 8:
        raise ValueError("LMD requires signal length ≥ 8 (Smith 2005 standard)")

    t = np.arange(n_samples, dtype=np.float64)
    max_pf = int(np.log2(n_samples)) if max_pf is None else max_pf
    residue = S.copy()
    PFs: list[np.ndarray] = []

    if max_pf != -1:
        for _ in range(max_pf):
            h = residue.copy()
            a_total = np.ones(n_samples, dtype=np.float64)
            converged = False

            for __ in range(max_iter):
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

                a_t = np.clip(a_t, min_amp, max_amp)
                a_t = savgol_filter(a_t, smooth_window, 2)

                if _check_convergence(m_t, a_t, eps, converge_mean):
                    converged = True
                    break

                s_new = (h - m_t) / a_t
                a_total = np.clip(a_total * a_t, min_amp, max_amp)
                h = s_new

            if not converged:
                break

            current_pf = np.clip(a_total * s_new, -max_amp, max_amp)

            if np.any(np.isnan(current_pf)) or np.any(np.isinf(current_pf)):
                break

            if np.sum(current_pf ** 2) < eps_stable:
                break

            PFs.append(current_pf)
            residue -= current_pf

            if (is_increasing(residue) or is_increasing(-residue) or
                    np.sum(residue ** 2) < eps_stable or
                    len(argrelextrema(residue, np.greater)[0]) + len(argrelextrema(residue, np.less)[0]) <= 2):
                break

    else:
        while True:
            h = residue.copy()
            a_total = np.ones(n_samples, dtype=np.float64)
            converged = False

            for __ in range(max_iter):
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

                a_t = np.clip(a_t, min_amp, max_amp)
                a_t = savgol_filter(a_t, smooth_window, 2)

                if _check_convergence(m_t, a_t, eps, converge_mean):
                    converged = True
                    break

                s_new = (h - m_t) / a_t
                a_total = np.clip(a_total * a_t, min_amp, max_amp)
                h = s_new

            if not converged:
                break

            current_pf = np.clip(a_total * s_new, -max_amp, max_amp)

            if np.any(np.isnan(current_pf)) or np.any(np.isinf(current_pf)):
                break

            if np.sum(current_pf ** 2) < eps_stable:
                break

            PFs.append(current_pf)
            residue -= current_pf

            if (is_increasing(residue) or is_increasing(-residue) or
                    np.sum(residue ** 2) < eps_stable or
                    len(argrelextrema(residue, np.greater)[0]) + len(argrelextrema(residue, np.less)[0]) <= 2):
                break

    del h, a_total, t
    return np.array(PFs, dtype=np.float64), residue

def _mirror_extend_real(signal: np.ndarray, ext_idx: np.ndarray, n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
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
    except:
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

def compute_envelope(signal: Union[list, np.ndarray]) -> np.ndarray:
    from scipy.linalg import hilbert

    signal = np.asarray(signal, dtype=np.float64).ravel()
    return np.abs(hilbert(signal))