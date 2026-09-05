"""
Numba backend for SVMD.

Not yet implemented: raises RealizationError until the accelerated core is
fixed and re-integrated.
"""

from .Error import RealizationError

__all__ = ["numba_svmd"]


def numba_svmd(S, num_modes, alpha, tau, tol, max_iter):
    """
    Numba-accelerated SVMD core (placeholder).
    """
    raise RealizationError("SVMD numba backend is not implemented yet")
