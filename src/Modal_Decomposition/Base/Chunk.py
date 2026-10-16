"""
Chunked processing utilities (统一分块工具)

该模块对同属同级子模块仅能使用 Memory
不同级模块可使用 Base
"""

import os
import tempfile
import uuid
from enum import Enum
from warnings import warn
from typing import Callable, Iterator, Literal, Optional

import numpy as np

from ..Define import SIZE, DEFAULT_NUMPY_TYPE
from .Memory import memory_budget

__all__ = ["Chunk", "CHUNK_MOD", "CHUNK_MOD_ARGS"]

# 流式填充的默认块大小 (元素数) = 8M 元素。
DEFAULT_FILL_CHUNK_SIZE = 8 * SIZE["1MB"]

CHUNK_MOD_ARGS = Literal["memory", "exo-memory"]
class CHUNK_MOD(Enum):
    Memory = "memory"
    ExoMemory = "exo-memory"


def validate_chunk_size(chunk_size: int) -> int:
    """校验块大小: 必须为正整数 (拒绝 bool 与非整数)。"""
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int):
        raise TypeError(f"chunk_size must be a positive integer, got {chunk_size!r}")

    if chunk_size < 1:
        raise ValueError(f"chunk_size must be positive, got {chunk_size!r}")

    return chunk_size

class Chunk:
    """
    Class facade over the chunking functions.

    Parameters
    ----------
    chunk_size : int
        Positive chunk length in elements.
    mod : Literal["memory", "exo-memory", "None"]
        Splitting mode: ``"memory"`` (default) yields in-RAM views;
        ``"exo-memory"`` yields temporary disk-backed memmaps (deleted after
        use); ``"None"`` performs no splitting (the whole array is yielded as
        a single chunk).
    memmap_pth : str, optional
        Alias of ``temp_dir`` (exo-memory mode only); defaults to the OS temp
        directory.
    memmap_type : np.dtype, optional
        Alias of ``dtype`` (exo-memory disk dtype); defaults to the input
        dtype.

    Examples
    --------
    >>> from Modal_Decomposition.Utils.Chunk import Chunk
    >>> import numpy as np
    >>> S = np.arange(10)
    >>> [c.sum() for c in Chunk(4).chunk(S)]      # memory mode
    [6, 22, 30]
    >>> Chunk().chunk(S, chunk_size=4, mod="memory").__next__().shape
    (10,)
    """
    @staticmethod
    def chunk_size(nbytes: int, *, chunk_ratio: float = 0.75, extras_bytes: int = 0, **kwargs) -> int:
        """
        If `nbytes` <= `DEFAULT_FILL_CHUNK_SIZE`, use `DEFAULT_FILL_CHUNK_SIZE`.
        Else, if `nbytes` <= `budget` * `chunk_ratio`, use `nbytes`.
        Else, use `buget` * `chunk_ratio`.

        Starts from ``base`` (default 8M elements) so the working set fits the global memory policy — i.e. a **granular, policy-aware** chunk size for
        ``chunked_fill`` / ``iter_chunks`` callers (used by ``Utils.Check`` for memmap fills and now-axis validation).

        Parameters
        ----------
        nbytes : int
            Input size in bytes.
        chunk_ratio : float
            The ratio of the remaining available memory.
        extras_bytes : int
            The extras bytes

        Returns
        -------
        int
            Adaptive chunk size (never below :data:`MIN_CHUNK_ELEMS` unless
            ``base`` is smaller).
        """
        if isinstance(nbytes, (np.ndarray, np.memmap)):
            try:
                nbytes = nbytes.nbytes
                warn("the input not int! get nbtyes from its attribute")
            except Exception:
                raise ValueError("the input not the nbtyes! error happened when try to get .nbtyes from input!")

        if not isinstance(nbytes, int):
            raise ValueError("nbytes must be int! not {}".format(type(nbytes)))

        if not isinstance(chunk_ratio, float):
            raise ValueError("chunk_ratio must be float! not {}".format(type(chunk_ratio)))
        else:
            if not (0 < chunk_ratio <= 1):
                raise ValueError("chunk ratio must in (0, 1], not {}".format(chunk_ratio))

        if not isinstance(extras_bytes, int):
            raise TypeError("extras bytes must be int! not {}".format(type(extras_bytes)))
        else:
            if not (extras_bytes >= 0):
                raise ValueError("extras bytes must be >= 0! not {}".format(extras_bytes))

        if nbytes <= DEFAULT_FILL_CHUNK_SIZE:
            return DEFAULT_FILL_CHUNK_SIZE

        mb = memory_budget(extras_bytes=extras_bytes, force=kwargs.get("force", False))
        limit = mb["budget"]
        limit_chunk_memory = int(limit * chunk_ratio)

        if nbytes <= limit_chunk_memory:
            return nbytes

        else:
            return limit_chunk_memory

    @classmethod
    def chunk(cls, S: np.ndarray, *, chunk_size: int = None, mod: CHUNK_MOD_ARGS | CHUNK_MOD = "memory", axis: int=0, dtype=None, temp_dir: Optional[str]=None, **kwargs) -> Iterator[np.ndarray]:
        """
        Split ``S`` according to ``mod`` — always an iterator (in ``"None"``
        mode a single-chunk iterator over the whole array), so consumers need
        no per-mode branching.
        """
        if not isinstance(S, (np.ndarray, np.memmap)):
            try:
                S = np.asarray(S, copy=False, dtype=DEFAULT_NUMPY_TYPE)
            except Exception:
                raise

        if S.dtype.kind not in "uif":
            raise ValueError("the kind of dtype of input must be int or float! not {}".format(S.dtype.kind))

        if S.ndim == 0:
            if S.size != 0:
                yield S
                return
            else:
                raise ValueError("chunking requires at least 1 dimension, got 0-d array")

        chunk_size = validate_chunk_size(chunk_size if chunk_size is not None else cls.chunk_size(nbytes=S.nbytes, chunk_ratio=kwargs.get("chunk_ratio", 0.75), extras_bytes=kwargs.get("extras_bytes", 0)))

        try:
            mod = CHUNK_MOD(mod)
        except ValueError:
            raise ValueError(f"mod must be one of ['memory', 'exo-memory'], got {mod!r}")

        if isinstance(axis, bool) or not isinstance(axis, int):
            raise TypeError(f"axis must be an int, got {axis!r}")
        if not (0 <= axis < S.ndim):
            raise ValueError(f"axis {axis} out of bounds for array of ndim {S.ndim}")

        match mod:
            case CHUNK_MOD.Memory:
                yield from cls._iter_chunks(S, chunk_size, axis=axis)

            case CHUNK_MOD.ExoMemory:
                yield from cls._exo_chunks(S, chunk_size, temp_dir=temp_dir, dtype=dtype, axis=axis)

    @staticmethod
    def _iter_chunks(S: np.ndarray, chunk_size: int, axis: int = 0) -> Iterator[np.ndarray]:
        length = S.shape[axis]
        if length == 0:
            raise ValueError("chunking for goal dim requires at least 1, got 0")

        head: tuple = (slice(None),) * axis  # 前 axis 维全取
        for start in range(0, length, chunk_size):
            idx = head + (slice(start, start + chunk_size),)
            yield S[idx]

    @staticmethod
    def _exo_chunks(
            S: np.ndarray,
            chunk_size: int,
            axis: int = 0,
            temp_dir: Optional[str] = None,
            dtype=None,
    ) -> Iterator[np.memmap]:
        """
        Yield disk-backed chunks of ``S``.

        Each chunk is written to a temporary ``.dat`` memmap in ``temp_dir``
        (default: the OS temp directory). Files stay mapped for the whole
        iteration and are closed + removed (best-effort; Windows may keep a file
        while external views still map it) when the generator is exhausted or
        closed.

        Warning
        -------
        A yielded ``np.memmap`` is only guaranteed valid **inside its iteration
        step**: consume (or copy) it before requesting the next chunk; do not
        keep references past the end of the iteration.

        Parameters
        ----------
        S : array-like
            Input array (splitting happens on axis 0).
        chunk_size : int
            Positive chunk length.
        temp_dir : str, optional
            Directory for the temporary files (created when missing).
        dtype : np.dtype, optional
            Disk dtype; defaults to ``S.dtype``.

        Yields
        ------
        np.memmap
            One disk-backed chunk per step.
        """
        length = S.shape[axis]
        if length == 0:
            raise ValueError("chunking for goal dim requires at least 1, got 0")

        head: tuple = (slice(None),) * axis

        temp_dir = temp_dir if temp_dir is not None else tempfile.gettempdir()
        os.makedirs(temp_dir, exist_ok=True)

        prefix = uuid.uuid4().hex

        dtype = dtype if dtype is not None else S.dtype

        kept: list = []  # (memmap, path) —— 迭代结束后统一关闭句柄并删除文件

        try:
            for i in range(0, length, chunk_size):
                idx = head + (slice(i, i + chunk_size),)

                path = os.path.join(temp_dir, f"md_chunk_{prefix}_{i // chunk_size}.dat")

                mm = np.memmap(
                    path,
                    dtype=dtype,
                    mode="w+",
                    shape=S[idx].shape,
                )
                mm[:] = S[idx]
                mm.flush()

                kept.append((mm, path))

                yield mm
        finally:
            for mm, path in kept:
                try:
                    mm._mmap.close()
                except Exception:
                    pass
                try:
                    os.remove(path)
                except OSError:
                    pass

    @staticmethod
    def chunked_map(
        func: Callable,
        S,
        chunk_size: Optional[int] = None,
        out: Optional[np.ndarray] = None,
        dtype=None,
    ) -> np.ndarray:
        """
        Apply ``func`` chunk-by-chunk along axis 0, returning one array.

        The peak working memory is one chunk plus the output array. ``fn`` must
        map a chunk to an array of the same per-element shape (e.g. elementwise
        transforms); the result is written into a preallocated output of shape
        ``S.shape``.

        Parameters
        ----------
        func : callable
            Per-chunk function: ``func(chunk) -> array with chunk.shape``.
        S : array-like
            Input array.
        chunk_size : int, optional
            Chunk length; defaults to :data:`DEFAULT_FILL_CHUNK_SIZE`.
        out : np.ndarray, optional
            Preallocated destination (same shape as ``S``; a ``np.memmap`` keeps
            the output disk-backed). When None, an in-RAM array is allocated
            with ``dtype`` (or the dtype of the first chunk result).
        dtype : np.dtype, optional
            Output dtype used when ``out`` is None.

        Returns
        -------
        np.ndarray
            The concatenated result (``out`` itself when provided).

        Notes
        -----
        For an empty input, ``out`` (or a newly allocated empty array) is
        returned untouched.
        """
        chunk_size = validate_chunk_size(
            DEFAULT_FILL_CHUNK_SIZE if chunk_size is None else chunk_size
        )
        S = np.asarray(S)
        if S.ndim == 0:
            raise ValueError("chunked_map requires at least 1 dimension, got 0-d array")

        length = S.shape[0]

        if out is not None:
            if out.shape != S.shape:
                raise ValueError(
                    f"out shape {out.shape} must equal input shape {S.shape}"
                )
            for start in range(0, length, chunk_size):
                end = min(start + chunk_size, length)
                out[start:end] = func(S[start:end])
            if hasattr(out, "flush"):
                out.flush()
            return out

        if length == 0:
            return np.empty(S.shape, dtype=dtype if dtype is not None else S.dtype)

        first_end = min(chunk_size, length)
        first = np.asarray(func(S[:first_end]))
        out = np.empty(S.shape, dtype=dtype if dtype is not None else first.dtype)
        out[:first_end] = first

        for start in range(first_end, length, chunk_size):
            end = min(start + chunk_size, length)
            out[start:end] = func(S[start:end])
        return out

    @staticmethod
    def chunked_fill(
        dst: np.ndarray | np.memmap,
        src: np.ndarray | np.memmap,
        chunk_size: int = DEFAULT_FILL_CHUNK_SIZE,
    ) -> np.ndarray:
        """
        Stream ``src`` into ``dst`` chunk-wise (peak memory = one chunk).

        ``dst`` may be a plain ndarray or a ``np.memmap`` (the typical use case:
        filling a disk-backed working copy from a large in-RAM / on-disk source
        without materializing a second full copy).

        Parameters
        ----------
        dst : np.ndarray
            Destination (preallocated, shape-compatible with ``src``).
        src : array-like
            Source; must broadcast-assign onto ``dst[start:end]`` slices.
        chunk_size : int
            Chunk length in elements; defaults to
            :data:`DEFAULT_FILL_CHUNK_SIZE`.

        Returns
        -------
        np.ndarray
            ``dst`` (now filled).
        """
        if not isinstance(dst, (np.ndarray, np.memmap)):
            raise TypeError("dst must be an ndarray or a memmap")

        if not isinstance(src, (np.ndarray, np.memmap)):
            raise TypeError("src must be an ndarray or a memmap")

        chunk_size = validate_chunk_size(chunk_size)
        length = dst.shape[0]

        for start in range(0, length, chunk_size):
            end = min(start + chunk_size, length)
            dst[start:end] = src[start:end]

        if hasattr(dst, "flush"):
            dst.flush()

        return dst

    @classmethod
    def fill_memmap(cls, fp: np.memmap, source: np.ndarray) -> None:
        """Stream ``source`` into ``fp`` chunk-wise so the peak memory stays at
        one chunk (delegates to ``Utils.Chunk.chunked_fill``; chunk granularity
        comes from ``Chunk.default_chunk_size`` under the global memory policy)."""
        src_nbytes = getattr(source, "nbytes", 0)
        nbytes = max(fp.nbytes, src_nbytes)

        chunk_size = cls.chunk_size(nbytes)
        cls.chunked_fill(fp, source, chunk_size)



if __name__ == "__main__":
    arr = np.arange(1000)

    for chunk in Chunk.chunk(arr, chunk_size=Chunk.chunk_size(arr)):
        print(chunk)