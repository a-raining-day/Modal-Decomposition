"""
Signal and time-axis validation utilities.

Single validation entry point for the whole package. Validation failures
raise exceptions; successful validation is silent.
"""

import warnings

import numpy as np

__all__ = ["to_signal", "require_ndim", "is_uniform", "Check_Time_and_Signal"]


def to_signal(S) -> np.ndarray:
    """
    Convert an array-like to a float64 ndarray and squeeze singleton
    dimensions.

    Parameters
    ----------
    S : array-like
        Input signal.

    Returns
    -------
    np.ndarray
        Array with all singleton dimensions removed.

    Raises
    ------
    ValueError
        If the input is a scalar or becomes 0-d after squeeze.
    """
    S = np.asarray(S, dtype=np.float64)
    if S.ndim == 0:
        raise ValueError("Signal must have at least 1 dimension, got 0-d array")
    S = S.squeeze()
    if S.ndim == 0:
        raise ValueError("Signal must have at least 1 dimension after squeeze")
    return S


def require_ndim(S: np.ndarray, allowed, method: str = "") -> None:
    """
    Validate the number of dimensions of S.

    Parameters
    ----------
    S : np.ndarray
        Array to check.
    allowed : set[int]
        Accepted dimensionality values.
    method : str
        Method name used in the error message.

    Raises
    ------
    ValueError
        If S.ndim is not in ``allowed``.
    """
    if S.ndim not in allowed:
        prefix = f"{method}: " if method else ""
        raise ValueError(
            f"{prefix}expected input dimension in {sorted(allowed)}, got {S.ndim}"
        )


def is_uniform(T: np.ndarray) -> bool:
    """
    Check whether the time axis is uniformly sampled.

    Parameters
    ----------
    T : np.ndarray
        1-d time axis.

    Returns
    -------
    bool
        True if uniformly sampled or shorter than 2 samples.
    """
    if T.size < 2:
        return True
    diff = np.diff(T)
    return bool(np.allclose(diff, diff[0], rtol=1e-10, atol=1e-14))


def Check_Time_and_Signal(S, T=None, ndim=None, method: str = "") -> tuple[np.ndarray, np.ndarray, int]:
    """
    Validate the signal and its time axis and return them in canonical form.

    Parameters
    ----------
    S : array-like
        Signal. Singleton dimensions are squeezed.
    T : array-like, optional
        Time axis. If None, defaults to ``arange(N)``.
    ndim : set[int], optional
        Allowed dimensionality of S. No check when None.
    method : str
        Method name used in error messages.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, int]
        (S, T, N). S and T are float64 ndarrays; N is the last-axis length.

    Raises
    ------
    ValueError
        On wrong dimensionality, length mismatch, duplicate time points, or
        empty signal.
    """
    S = to_signal(S)
    if ndim is not None:
        require_ndim(S, ndim, method)

    N = S.shape[-1]
    if N == 0:
        raise ValueError("Signal must not be empty")

    if T is None:
        T = np.arange(N, dtype=np.float64)
    else:
        T = to_signal(T)
        if T.ndim != 1:
            raise ValueError("Time axis T must be 1-dimensional")
        if len(T) != N:
            raise ValueError(f"Length mismatch between T ({len(T)}) and S ({N})")

        diff = np.diff(T)
        if np.any(diff == 0):
            raise ValueError("Time axis T contains duplicate values")
        if np.any(diff < 0):
            warnings.warn(
                "T is not monotonically increasing; S and T are reordered by ascending T.",
                UserWarning,
                stacklevel=2,
            )
            idx = np.argsort(T)
            T = T[idx]
            S = S[..., idx]

    return S, T, N
