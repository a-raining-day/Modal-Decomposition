"""
Chunked processing utilities (统一分块工具)。

背景
----
分块逻辑原本散落在两处: ``Check._fill_memmap`` (流式填充 memmap) 与
``Monotonicity`` (分块单调检测 + ``_adapt_chunk_size`` 自适应块大小)。
本模块把它们收敛为通用 API, 供全库复用 (见 docs/ChunkReport.md 效用报告):

* ``iter_chunks`` / ``chunks``   —— 沿第 0 维的内存视图分块迭代器 (零拷贝);
* ``chunked_map``               —— 逐块映射, 可预分配输出 (含 memmap), 峰值
                                  内存 = 一块 + 输出;
* ``chunked_fill``              —— 逐块流式填充 (dst 为 ndarray / memmap,
                                  峰值内存 = 一块);
* ``exo_chunks``                —— 外存临时 memmap 分块 (迭代期间保持映射,
                                  迭代结束后关闭句柄并删除文件)。
* ``adapt_chunk_size``          —— 按全局内存策略 (Utils.Memory) 收缩块大小,
                                  工作集 = ``nbytes + extra * chunk_elems``;
* ``default_chunk_size``        —— 流式处理的"粒度选择"入口: 以默认块为起点
                                  自适应收缩 (Check 的填充与时间轴校验使用);
* ``Chunk``                     —— 上述能力的类式门面 (向后兼容旧参数名)。

约定: 本模块自身不接触 ``Base.Cache`` —— 缓存注册统一由
``Utils.get_chunk()`` 完成 (见 Utils/__init__.py)。
"""

import os
import tempfile
import uuid
from typing import Callable, Iterator, Literal, Optional, Union

import numpy as np

from ..Base.ConstDefine import SIZE
from .Memory import get_available_memory, get_memory_policy

__all__ = [
    "iter_chunks",
    "chunks",
    "chunked_map",
    "chunked_fill",
    "exo_chunks",
    "adapt_chunk_size",
    "default_chunk_size",
    "Chunk",
    "MIN_CHUNK_ELEMS",
    "ADAPT_MIN_BYTES",
    "DEFAULT_FILL_CHUNK_ELEMS",
]

# --------------------------------------------------------------------------- #
# 常量 (大小统一取自 Base.ConstDefine.SIZE)
# --------------------------------------------------------------------------- #
#: 自适应收缩的块大小下限 (元素数) = 4KB 元素。
MIN_CHUNK_ELEMS = 4 * SIZE["1KB"]
#: 输入低于该字节数时不咨询全局内存策略 (直接沿用调用方块大小)。
ADAPT_MIN_BYTES = 64 * SIZE["1MB"]
#: 流式填充的默认块大小 (元素数) = 8M 元素。
DEFAULT_FILL_CHUNK_ELEMS = 8 * SIZE["1MB"]

#: Chunk 类支持的模式。
_MODES = ("memory", "exo-memory", "None")


def _as_array(S) -> np.ndarray:
    """把输入规范为 ndarray (保持 memmap 原样, 避免不必要的拷贝)。"""
    if isinstance(S, np.ndarray):
        return S
    return np.asarray(S)


def _validate_chunk_size(chunk_size) -> int:
    """校验块大小: 必须为正整数 (拒绝 bool 与非整数)。"""
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int):
        raise TypeError(
            f"chunk_size must be a positive integer, got {chunk_size!r}"
        )
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be positive, got {chunk_size!r}")
    return chunk_size


# --------------------------------------------------------------------------- #
# 函数级 API
# --------------------------------------------------------------------------- #
def iter_chunks(S, chunk_size: int) -> Iterator[np.ndarray]:
    """
    Yield zero-copy views of ``S`` split along axis 0.

    Parameters
    ----------
    S : array-like
        Input array (any dimensionality >= 1; splitting happens on axis 0).
        ndarray / memmap inputs are viewed in place (no copy).
    chunk_size : int
        Positive chunk length in elements. The last chunk may be shorter.

    Yields
    ------
    np.ndarray
        Views ``S[i:i+chunk_size]`` — mutating a chunk mutates ``S``.

    Raises
    ------
    TypeError
        On a non-integer ``chunk_size``.
    ValueError
        On a non-positive ``chunk_size`` or a 0-d input.
    """
    chunk_size = _validate_chunk_size(chunk_size)
    S = _as_array(S)
    if S.ndim == 0:
        raise ValueError("chunking requires at least 1 dimension, got 0-d array")

    length = S.shape[0]
    for start in range(0, length, chunk_size):
        yield S[start:start + chunk_size]


def chunks(S, chunk_size: int) -> Iterator[np.ndarray]:
    """Alias of :func:`iter_chunks`."""
    return iter_chunks(S, chunk_size)


def chunked_map(
    fn: Callable,
    S,
    chunk_size: Optional[int] = None,
    out: Optional[np.ndarray] = None,
    dtype=None,
) -> np.ndarray:
    """
    Apply ``fn`` chunk-by-chunk along axis 0, returning one array.

    The peak working memory is one chunk plus the output array. ``fn`` must
    map a chunk to an array of the same per-element shape (e.g. elementwise
    transforms); the result is written into a preallocated output of shape
    ``S.shape``.

    Parameters
    ----------
    fn : callable
        Per-chunk function: ``fn(chunk) -> array with chunk.shape``.
    S : array-like
        Input array.
    chunk_size : int, optional
        Chunk length; defaults to :data:`DEFAULT_FILL_CHUNK_ELEMS`.
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
    chunk_size = _validate_chunk_size(
        DEFAULT_FILL_CHUNK_ELEMS if chunk_size is None else chunk_size
    )
    S = _as_array(S)
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
            out[start:end] = fn(S[start:end])
        if hasattr(out, "flush"):
            out.flush()
        return out

    if length == 0:
        return np.empty(S.shape, dtype=dtype if dtype is not None else S.dtype)

    first_end = min(chunk_size, length)
    first = np.asarray(fn(S[:first_end]))
    out = np.empty(S.shape, dtype=dtype if dtype is not None else first.dtype)
    out[:first_end] = first

    for start in range(first_end, length, chunk_size):
        end = min(start + chunk_size, length)
        out[start:end] = fn(S[start:end])
    return out


def chunked_fill(
    dst: np.ndarray,
    src,
    chunk_size: int = DEFAULT_FILL_CHUNK_ELEMS,
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
        :data:`DEFAULT_FILL_CHUNK_ELEMS`.

    Returns
    -------
    np.ndarray
        ``dst`` (now filled).
    """
    chunk_size = _validate_chunk_size(chunk_size)
    length = dst.shape[0]
    for start in range(0, length, chunk_size):
        end = min(start + chunk_size, length)
        dst[start:end] = src[start:end]
    if hasattr(dst, "flush"):
        dst.flush()
    return dst


def exo_chunks(
    S,
    chunk_size: int,
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
    chunk_size = _validate_chunk_size(chunk_size)
    S = _as_array(S)
    if S.ndim == 0:
        raise ValueError("chunking requires at least 1 dimension, got 0-d array")

    length = S.shape[0]
    if length == 0:
        return

    temp_dir = temp_dir if temp_dir is not None else tempfile.gettempdir()
    os.makedirs(temp_dir, exist_ok=True)
    prefix = uuid.uuid4().hex
    dtype = dtype if dtype is not None else S.dtype

    kept: list = []  # (memmap, path) —— 迭代结束后统一关闭句柄并删除文件

    try:
        for i in range(0, length, chunk_size):
            path = os.path.join(
                temp_dir, f"md_chunk_{prefix}_{i // chunk_size}.dat"
            )
            mm = np.memmap(
                path,
                dtype=dtype,
                mode="w+",
                shape=S[i:i + chunk_size].shape,
            )
            mm[:] = S[i:i + chunk_size]
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


def adapt_chunk_size(chunk_size: int, nbytes: int, extra_per_elem: int = 3) -> int:
    """
    Shrink ``chunk_size`` so the working set fits the global memory policy.

    Working set = ``nbytes + extra_per_elem * chunk_elements`` (the default
    ``3`` matches the boolean-mask temporaries of the monotonicity checks).
    Inputs below :data:`ADAPT_MIN_BYTES` keep the caller's chunk size
    untouched; when the input alone already exceeds the policy limit, the
    chunk size is returned unchanged (smaller chunks cannot help).

    Parameters
    ----------
    chunk_size : int
        Caller's chunk length.
    nbytes : int
        Input size in bytes.
    extra_per_elem : int
        Bytes of working memory per chunk element (default 3).

    Returns
    -------
    int
        Possibly shrunk chunk size, never below
        :data:`MIN_CHUNK_ELEMS` (unless the caller's size is smaller).
    """
    if nbytes < ADAPT_MIN_BYTES:
        return chunk_size

    policy = get_memory_policy()

    if policy["use_ratio_strategy"]:
        available = get_available_memory()
        if available is None:
            return chunk_size  # psutil unavailable: cannot judge the ratio
        limit = policy["memmap_ratio_limit"] * available
    else:
        limit = policy["absolute_memmap_limit_bytes"]

    budget = int(limit) - int(nbytes)
    if budget <= 0:
        # 输入本身已超策略上限: 缩小块无济于事, 沿用调用方块大小。
        return chunk_size

    cap = max(MIN_CHUNK_ELEMS, budget // max(extra_per_elem, 1))
    return max(1, min(chunk_size, cap))


def default_chunk_size(
    nbytes: int,
    base: int = DEFAULT_FILL_CHUNK_ELEMS,
    extra_per_elem: int = 3,
) -> int:
    """
    Pick the chunk size for streaming over an array of ``nbytes`` bytes.

    Starts from ``base`` (default 8M elements) and shrinks it via
    :func:`adapt_chunk_size` so the working set fits the global memory
    policy — i.e. a **granular, policy-aware** chunk size for
    ``chunked_fill`` / ``iter_chunks`` callers (used by ``Utils.Check``
    for memmap fills and time-axis validation).

    Parameters
    ----------
    nbytes : int
        Input size in bytes.
    base : int
        Preferred chunk length in elements (default 8M).
    extra_per_elem : int
        Bytes of working memory per chunk element (default 3).

    Returns
    -------
    int
        Adaptive chunk size (never below :data:`MIN_CHUNK_ELEMS` unless
        ``base`` is smaller).
    """
    return adapt_chunk_size(
        _validate_chunk_size(base), nbytes, extra_per_elem
    )


# --------------------------------------------------------------------------- #
# 类式门面
# --------------------------------------------------------------------------- #
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
    >>> Chunk(4, mod="None").chunk(S).__next__().shape
    (10,)
    """

    def __init__(
        self,
        chunk_size: int,
        mod: Literal["memory", "exo-memory", "None"] = "memory",
        memmap_pth: Optional[str] = None,
        memmap_type=None,
        **kwargs,
    ):
        if mod not in _MODES:
            raise ValueError(f"mod must be one of {_MODES}, got {mod!r}")
        self.mod = mod
        self.chunk_size = _validate_chunk_size(chunk_size)
        self.temp_dir = memmap_pth
        self.dtype = memmap_type
        self.kwargs = kwargs

    # -- 迭代 ------------------------------------------------------------- #
    def chunk(
        self, S: np.ndarray
    ) -> Iterator[Union[np.ndarray, np.memmap]]:
        """
        Split ``S`` according to ``mod`` — always an iterator (in ``"None"``
        mode a single-chunk iterator over the whole array), so consumers need
        no per-mode branching.
        """
        if self.mod == "memory":
            return iter_chunks(S, self.chunk_size)
        if self.mod == "exo-memory":
            return exo_chunks(
                S, self.chunk_size, temp_dir=self.temp_dir, dtype=self.dtype
            )
        # mod == "None": 不分块, 单块迭代 (保持统一迭代器语义)
        S = _as_array(S)
        if S.ndim == 0:
            raise ValueError("chunking requires at least 1 dimension, got 0-d array")
        return iter((S,))

    def __call__(self, S):
        """Alias of :meth:`chunk`."""
        return self.chunk(S)

    # -- 便捷 ------------------------------------------------------------- #
    def map(self, fn: Callable, S, out=None, dtype=None) -> np.ndarray:
        """Apply ``fn`` chunk-by-chunk (see :func:`chunked_map`)."""
        return chunked_map(
            fn, S, chunk_size=self.chunk_size, out=out, dtype=dtype
        )

    def fill(self, dst: np.ndarray, src) -> np.ndarray:
        """Stream ``src`` into ``dst`` chunk-wise (see :func:`chunked_fill`)."""
        return chunked_fill(dst, src, self.chunk_size)

    def __repr__(self) -> str:
        return (
            f"<Chunk mod={self.mod!r} chunk_size={self.chunk_size} "
            f"temp_dir={self.temp_dir!r}>"
        )
