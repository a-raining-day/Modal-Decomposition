"""
Reconstruction error thresholds per method.

Exact methods (residual + modes reconstruct the signal by construction)
use a tight tolerance. Methods whose reconstruction error is inherent to
the algorithm use documented loose tolerances.
"""

import numpy as np
import pytest

from Modal_Decomposition import Class

from _cases import CASES, METHODS

TIGHT_TOL = 1e-6

# Approximate methods: reconstruction error is inherent to the algorithm.
LOOSE_TOL = {
    "EEMD": 0.5,   # finite ensemble average of added noise
    "EWT": 0.5,    # boundary detection approximation
    "LMD": 0.05,   # sifting stops at convergence thresholds
    "VMD": 0.5,    # variational approximation, no residual
}


def _run(method, signal):
    params = CASES[method]
    if method == "MEMD":
        s2 = np.stack([signal, signal[::-1]])
        return getattr(Class, method)(**params).decompose(s2), s2
    return getattr(Class, method)(**params).decompose(signal), signal


@pytest.mark.parametrize("method", METHODS)
def test_reconstruction_within_tolerance(method, signal):
    result, s2 = _run(method, signal)
    recon = result.reconstruct()
    err = float(np.max(np.abs(recon - s2)))
    tol = LOOSE_TOL.get(method, TIGHT_TOL)
    assert err < tol, f"{method}: reconstruction error {err:.3e} >= {tol}"
