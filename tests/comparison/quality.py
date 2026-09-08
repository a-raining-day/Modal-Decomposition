"""
Quality metrics for one decomposition result.

All metrics work on the component *stack* returned by the implementations
(rows = components, residue appended as last row when the backend reports
one), so the same code applies to all three workers.
"""

from __future__ import annotations

import numpy as np

__all__ = ["analyze"]


def recon_max_abs(S: np.ndarray, stack: np.ndarray) -> float:
    """Largest |reconstruction error|: max|S - sum(components)|."""
    err = S - np.sum(stack, axis=0)
    return float(np.max(np.abs(err))) if err.size else float("nan")


def orthogonality_index(stack: np.ndarray) -> float:
    """
    Huang's orthogonality index over all reported components:

        IO = sum_{i != j} <c_i, c_j> / sum_i ||c_i||^2

    IO ~ 0 means near-orthogonal components (no shared energy between IMFs).
    """
    rows = np.asarray(stack, dtype=np.float64)
    k, n = rows.shape
    if k == 0 or n == 0:
        return float("nan")
    denom = float(np.sum(rows * rows))
    if denom == 0:
        return float("nan")
    num = 0.0
    for i in range(k):
        for j in range(i + 1, k):
            num += 2.0 * float(np.dot(rows[i], rows[j]))
    return num / denom


def _best_corr_nrmse(mode: np.ndarray, rows: np.ndarray):
    """Best |correlation| (and its least-squares NRMSE) across all rows."""
    m = mode - mode.mean()
    nm = np.sqrt(np.dot(m, m))
    if nm == 0:
        return float("nan"), float("nan")
    best_corr, best_nrmse = -1.0, float("inf")
    for i in range(rows.shape[0]):
        r = rows[i]
        rc = r - r.mean()
        nr = np.sqrt(np.dot(rc, rc))
        if nr == 0:
            continue
        corr = abs(float(np.dot(rc, m)) / (nr * nm))
        if corr > best_corr:
            best_corr = corr
            scale = float(np.dot(r, mode)) / float(np.dot(r, r))
            best_nrmse = float(
                np.sqrt(np.sum((scale * r - mode) ** 2)) / np.sqrt(np.sum(mode**2))
            )
    return best_corr, best_nrmse


def analyze(S: np.ndarray, stack: np.ndarray, modes: dict) -> dict:
    """
    Return a serializable dict of quality metrics for ``stack``.

    ``modes`` maps ground-truth component names to arrays (see signals.py);
    pass ``{}`` when no ground truth exists.
    """
    rows = np.asarray(stack, dtype=np.float64)
    if rows.ndim == 1:
        rows = rows.reshape(1, -1)
    out: dict = {
        "n_rows": int(rows.shape[0]),
        "recon_max_abs_err": recon_max_abs(np.asarray(S, dtype=np.float64), rows),
        "orthogonality_index": orthogonality_index(rows),
    }
    recovery = {}
    for name, mode in modes.items():
        corr, nrmse = _best_corr_nrmse(np.asarray(mode, dtype=np.float64), rows)
        recovery[name] = {
            "best_abs_corr": None if np.isnan(corr) else float(corr),
            "best_nrmse": None if np.isnan(nrmse) else float(nrmse),
        }
    if recovery:
        out["mode_recovery"] = recovery
    return out
