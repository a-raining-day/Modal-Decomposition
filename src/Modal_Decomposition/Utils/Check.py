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
from typing import Optional

import numpy as np

from ..Base.ConstDefine import SIZE
from .Chunk import chunked_fill, default_chunk_size, iter_chunks
from .Memory import should_use_memmap

__all__ = ["to_signal", "require_ndim", "is_uniform", "Check_Time_and_Signal", "detect_dtype"]

# Input-size arbitration constants (module-level so tests can patch them;
# 大小统一取自 Base.ConstDefine.SIZE)。
_LIST_MEM_THRESHOLD = 10 * SIZE["1MB"]  # elements: lists above this become memmaps
_F64_DISK_BYTES = 256 * SIZE["1MB"]     # float64 working copy above this goes to disk
_FILL_CHUNK_ELEMS = 8 * SIZE["1MB"]     # chunk size for streaming fills (8M elements)


def _temp_memmap(dtype, shape) -> np.memmap:
    """Create a writable temporary memmap. The backing file is left for the
    OS temp cleaner (it cannot be unlinked while mapped on Windows)."""
    fd, path = tempfile.mkstemp(prefix="md_memmap_", suffix=".dat")
    os.close(fd)
    return np.memmap(path, dtype=dtype, mode="w+", shape=shape)


def _fill_memmap(fp: np.memmap, source) -> None:
    """Stream ``source`` into ``fp`` chunk-wise so the peak memory stays at
    one chunk (delegates to ``Utils.Chunk.chunked_fill``; chunk granularity
    comes from ``Chunk.default_chunk_size`` under the global memory policy)."""
    src_nbytes = getattr(source, "nbytes", None)
    nbytes = max(fp.nbytes, src_nbytes if src_nbytes is not None else fp.nbytes)
    chunk_size = default_chunk_size(nbytes, base=_FILL_CHUNK_ELEMS)
    chunked_fill(fp, source, chunk_size)


def _load_signal_input(S, target_dtype=None):
    """
    Resolve the raw signal input under the global memory policy.

    - ``.npy`` paths load as ``np.load(..., mmap_mode="r")`` when the policy
      says so, else fully in RAM.
    - Large in-RAM arrays are copied chunk-wise into a temporary
      disk-backed array when the policy triggers (note: the caller's original
      array remains in RAM; the copy is what keeps the pipeline's working
      memory bounded). When ``target_dtype`` is given, the conversion happens
      **during** that fill — a single pass, a single temp file (instead of
      copy + second conversion pass).
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
            out_dtype = target_dtype if target_dtype is not None else S.dtype
            fp = _temp_memmap(out_dtype, S.shape)
            _fill_memmap(fp, S)  # dtype 转换在填充时一并完成 (单趟)
            return fp
        return S

    if isinstance(S, (list, tuple)) and len(S) > _LIST_MEM_THRESHOLD:
        out_dtype = target_dtype if target_dtype is not None else np.float64
        fp = _temp_memmap(out_dtype, (len(S),))
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


def _to_keep_dtype(S) -> np.ndarray:
    """
    Normalize ``S`` **without** dtype conversion: array-ify, reject
    non-numeric / 0-d, squeeze singleton dims, keep the original dtype.
    """
    if not isinstance(S, np.ndarray):
        S = np.asarray(S)
    if S.dtype.kind not in "biuf":
        raise ValueError(f"unsupported non-numeric dtype {S.dtype}")
    if S.ndim == 0:
        raise ValueError("Signal must have at least 1 dimension, got 0-d array")
    S = S.squeeze()
    if S.ndim == 0:
        raise ValueError("Signal must have at least 1 dimension after squeeze")
    return S


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


def Check_Time_and_Signal(
    S, T=None, ndim=None, method: str = "", default_T: bool = True,
    dtype: Optional[str] = None,
) -> tuple[np.ndarray, Optional[np.ndarray], int]:
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
        Time axis. If None, defaults to ``arange(N)`` unless ``default_T``
        is False.
    ndim : set[int], optional
        Allowed dimensionality of S. No check when None.
    method : str
        Method name used in error messages.
    default_T : bool
        When True (default) and ``T`` is None, build ``arange(N, float64)``.
        When False, leave ``T`` as None: backends that rebuild their own
        timeline (e.g. the PyEMD sifters, which ignore a passed time axis
        under the default extrema detection) then avoid a one-signal-sized
        transient allocation. Callers must only set this when their algorithm
        does not use the returned ``T``.
    dtype : {"float64", None}, optional
        Target dtype of the returned signal. Default ``None``: **keep the
        input's dtype** (bool/int/float 按原样, 校验为实数数值即可) —— 不再
        默认转 float64 以避免 f16/f32 输入的内存翻倍。传 ``"float64"`` 时
        才做统一 float64 转换 (大输入按内存策略走分块磁盘转换)。``T`` 遵循
        同一策略 (缺省生成的 ``arange(N)`` 保持 int64)。

    Returns
    -------
    tuple[np.ndarray, Optional[np.ndarray], int]
        (S, T, N). S 默认保留输入 dtype (可能 memmap 支撑); 显式
        ``dtype="float64"`` 时为 float64; T 同理 (或 None 当 ``T`` 未给且
        ``default_T`` 为 False); N 为末轴长度。

    Raises
    ------
    ValueError
        On wrong dimensionality, length mismatch, duplicate time points, an
        unsupported file suffix, empty signal, or an invalid ``dtype`` value.
    """
    if dtype not in (None, "float64"):
        raise ValueError(f"dtype must be 'float64' or None, got {dtype!r}")

    # 单趟: 内存策略落盘时直接按目标 dtype 写入 (见 _load_signal_input)。
    S = _load_signal_input(S, target_dtype=dtype)

    if dtype == "float64":
        S = _to_float64(S)
        S = to_signal(S)
    else:
        S = _to_keep_dtype(S)

    if ndim is not None:
        require_ndim(S, ndim, method)

    N = S.shape[-1]
    if N == 0:
        raise ValueError("Signal must not be empty")

    if T is None:
        if default_T:
            T = np.arange(N)
        # else: keep T as None (see default_T above).
    else:
        T = to_signal(T) if dtype == "float64" else _to_keep_dtype(T)
        if T.ndim != 1:
            raise ValueError("Time axis T must be 1-dimensional")
        if len(T) != N:
            raise ValueError(f"Length mismatch between T ({len(T)}) and S ({N})")

        # 分块检查重复与降序 (粒度经 Chunk.default_chunk_size 自适应;
        # 避免整条 np.diff 的一次性内存占用)。
        needs_sort = False
        prev_last = None
        t_chunk_size = default_chunk_size(T.nbytes, base=_FILL_CHUNK_ELEMS)
        for chunk in iter_chunks(T, t_chunk_size):
            d = np.diff(chunk)
            if np.any(d == 0):
                raise ValueError("Time axis T contains duplicate values")
            if not needs_sort and np.any(d < 0):
                needs_sort = True
            if prev_last is not None and chunk[0] == prev_last:
                raise ValueError("Time axis T contains duplicate values")
            prev_last = chunk[-1]

        if needs_sort:
            warnings.warn(
                "T is not monotonically increasing; S and T are reordered by ascending T.",
                UserWarning,
                stacklevel=2,
            )
            idx = np.argsort(T)
            T = T[idx]
            S = S[..., idx]

    return S, T, N
