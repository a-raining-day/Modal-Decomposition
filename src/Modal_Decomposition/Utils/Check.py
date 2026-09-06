"""
Signal and time-axis validation utilities.

Single validation entry point for the whole package. Validation failures
raise exceptions; successful validation is silent.

``Check_Time_and_Signal`` is also the single choke point where data enters
the package: input-size detection and the disk-backed (``np.memmap``)
decision of the global memory policy (``Utils.Memory``) are applied here,
*before* any conversion, so very large inputs are never copied into RAM in
one shot.
"""

import os
import tempfile
import warnings

import numpy as np

from .Memory import should_use_memmap

__all__ = ["to_signal", "require_ndim", "is_uniform", "Check_Time_and_Signal", "detect_dtype"]

# Input-size arbitration constants (module-level so tests can patch them).
_LIST_MEM_THRESHOLD = 10_000_000  # elements: lists above this become memmaps
_F64_DISK_BYTES = 256 * 1024 * 1024  # float64 working copy above this goes to disk
_FILL_CHUNK_ELEMS = 8_388_608  # chunk size for streaming fills (8M elements)


def _temp_memmap(dtype, shape) -> np.memmap:
    """Create a writable temporary memmap. The backing file is left for the
    OS temp cleaner (it cannot be unlinked while mapped on Windows)."""
    fd, path = tempfile.mkstemp(prefix="md_memmap_", suffix=".dat")
    os.close(fd)
    return np.memmap(path, dtype=dtype, mode="w+", shape=shape)


def _fill_memmap(fp: np.memmap, source) -> None:
    """Stream ``source`` into ``fp`` chunk-wise so the peak memory stays at
    one chunk."""
    n = fp.shape[0]
    for start in range(0, n, _FILL_CHUNK_ELEMS):
        end = min(start + _FILL_CHUNK_ELEMS, n)
        fp[start:end] = source[start:end]
    fp.flush()


def _load_signal_input(S):
    """
    Resolve the raw signal input under the global memory policy.

    - ``.npy`` paths load as ``np.load(..., mmap_mode="r")`` when the policy
      says so, else fully in RAM.
    - Large in-RAM arrays are copied chunk-wise into a temporary
      disk-backed array when the policy triggers (note: the caller's original
      array remains in RAM; the copy is what keeps the pipeline's working
      memory bounded).
    - Very long sequences (list/tuple) are filled into a temporary memmap
      instead of being converted to a RAM array in one shot.
    """
    if isinstance(S, (str, os.PathLike)):
        path = os.fspath(S)
        if not path.endswith(".npy"):
            raise ValueError(f"only .npy files are supported, got {path!r}")
        if should_use_memmap(os.path.getsize(path)):
            return np.load(path, mmap_mode="r")
        return np.load(path)

    if isinstance(S, np.ndarray):
        if isinstance(S, np.memmap):
            return S  # already disk-backed
        if S.ndim >= 1 and S.size > 0 and should_use_memmap(S.nbytes):
            fp = _temp_memmap(S.dtype, S.shape)
            _fill_memmap(fp, S)
            return fp
        return S

    if isinstance(S, (list, tuple)) and len(S) > _LIST_MEM_THRESHOLD:
        fp = _temp_memmap(np.float64, (len(S),))
        _fill_memmap(fp, S)
        return fp

    return S


def _to_float64(S) -> np.ndarray:
    """
    Normalize ``S`` to float64 with bounded peak memory: small inputs convert
    in RAM as before, while large or already disk-backed non-float64 inputs
    are converted chunk-wise into a temporary float64 memmap.
    """
    if not isinstance(S, np.ndarray):
        return np.asarray(S, dtype=np.float64)

    if S.dtype == np.float64:
        return S  # no copy; memmaps stay disk-backed

    projected = S.nbytes * (8 // max(S.dtype.itemsize, 1))
    if S.ndim >= 1 and S.size > 0 and (isinstance(S, np.memmap) or projected >= _F64_DISK_BYTES):
        fp = _temp_memmap(np.float64, S.shape)
        _fill_memmap(fp, S)
        return fp

    return np.asarray(S, dtype=np.float64)


def detect_dtype(S) -> str:
    """
    Detect the numeric kind of an array-like.

    Parameters
    ----------
    S : array-like
        Input array.

    Returns
    -------
    str
        ``"bool"``, ``"int"`` or ``"float"``.

    Raises
    ------
    ValueError
        For non-numeric dtypes (complex, string, object, datetime, ...).
    """
    arr = np.asarray(S)

    if arr.dtype.kind == "b":
        return "bool"

    if arr.dtype.kind in "iu":
        return "int"

    if arr.dtype.kind == "f":
        return "float"

    raise ValueError(f"unsupported non-numeric dtype {arr.dtype}")


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
    if isinstance(S, np.memmap) and S.dtype == np.float64:
        S = S  # keep disk-backed inputs on disk (np.asarray would strip it)
    else:
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
    S : array-like or str/os.PathLike
        Signal, or the path of a ``.npy`` file. Input-size detection and the
        global memory policy are applied here: oversized inputs are served
        from a ``np.memmap`` instead of being copied into RAM. Singleton
        dimensions are squeezed.
    T : array-like, optional
        Time axis. If None, defaults to ``arange(N)``.
    ndim : set[int], optional
        Allowed dimensionality of S. No check when None.
    method : str
        Method name used in error messages.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, int]
        (S, T, N). S and T are float64 arrays (possibly memmap-backed);
        N is the last-axis length.

    Raises
    ------
    ValueError
        On wrong dimensionality, length mismatch, duplicate time points, an
        unsupported file suffix, or empty signal.
    """
    S = _load_signal_input(S)
    S = _to_float64(S)
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
