"""
分块处理工具：按指定轴把数组切成若干块，支持内存视图与磁盘 memmap 两种模式。

依赖方向：同一层（``Utils``）内的子模块只能经 ``Base.Memory`` 取内存预算；
跨层模块才可直接使用 ``Base``。
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
    分块工具的门面：所有入口都是 ``@staticmethod`` / ``@classmethod``，不需要实例。

    ``chunk`` 的两种模式：``"memory"``（默认）产出内存视图；``"exo-memory"`` 产出
    临时磁盘 memmap，迭代结束即关闭句柄并尽力删除。

    Examples
    --------
    >>> import numpy as np
    >>> from Modal_Decomposition.Base.Chunk import Chunk
    >>> S = np.arange(10)
    >>> [int(c.sum()) for c in Chunk.chunk(S, chunk_size=4)]
    [6, 22, 17]
    >>> next(Chunk.chunk(S, chunk_size=4)).shape
    (4,)
    """
    @staticmethod
    def chunk_size(nbytes: int, *, chunk_ratio: float = 0.75, extras_bytes: int = 0, **kwargs) -> int:
        """
        自适应块大小（元素数）：小输入固定用 :data:`DEFAULT_FILL_CHUNK_SIZE`，大输入
        取 ``min(nbytes, budget * chunk_ratio)``，让单块工作集落在全局内存预算内。

        入参 ``nbytes`` 以字节计，返回值以元素计（与 :data:`DEFAULT_FILL_CHUNK_SIZE`
        同单位）；传入 ``ndarray`` / ``np.memmap`` 时取其实例的 ``.nbytes`` 并告警。

        Parameters
        ----------
        nbytes : int
            输入体积（字节）。
        chunk_ratio : float
            允许占用的可用内存比例，取值 ``(0, 1]``。
        extras_bytes : int
            预算中需预留的额外字节数，不能为负。

        Returns
        -------
        int
            块大小（元素数）。
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
        按 ``mod`` 切分 ``S``，始终返回迭代器，调用方无需按模式分支。

        ``axis`` 指定切分轴；``mod`` 只接受 ``CHUNK_MOD`` 的成员或值
        （``"memory"`` / ``"exo-memory"``）；``chunk_size=None`` 时按输入体积调用
        ``Chunk.chunk_size`` 自适应。
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
        """沿 ``axis`` 产出内存视图分块，不复制数据。"""
        length = S.shape[axis]
        if length == 0:
            raise ValueError("chunking for goal dim requires at least 1, got 0")

        head: tuple = (slice(None),) * axis  # 前 axis 维全取，只在 axis 上切片
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
        产出 ``S`` 的磁盘分块。

        每块写入 ``temp_dir``（默认系统临时目录）下的临时 ``.dat`` memmap；迭代期间
        保持映射，生成器耗尽或被关闭时统一关闭句柄并尽力删除文件（Windows 上外部
        视图仍持有映射时可能删不掉）。

        Warning
        -------
        产出的 ``np.memmap`` 只保证在本次迭代步内有效：取下一块之前消费或拷贝它，
        不要越过迭代末尾继续持有引用。

        Parameters
        ----------
        S : array-like
            输入数组。
        chunk_size : int
            块长度（元素数），必须为正。
        axis : int
            切分轴，默认 0。
        temp_dir : str, optional
            临时文件目录，不存在时创建。
        dtype : np.dtype, optional
            磁盘 dtype，默认取 ``S.dtype``。

        Yields
        ------
        np.memmap
            每步一个磁盘分块。
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
        沿 0 轴逐块应用 ``func``，返回一个数组。

        峰值内存为「一块 + 输出数组」。``func`` 必须把一块映射成逐元素形状相同的
        数组（逐元素变换），结果写入按 ``S.shape`` 预分配的输出。

        Parameters
        ----------
        func : callable
            逐块函数：``func(chunk) -> 与 chunk.shape 相同的数组``。
        S : array-like
            输入数组。
        chunk_size : int, optional
            块长度，默认 :data:`DEFAULT_FILL_CHUNK_SIZE`。
        out : np.ndarray, optional
            预分配的目标数组，形状须与 ``S`` 相同；传 ``np.memmap`` 时输出落在磁盘。
            为 None 时按 ``dtype``（或第一块结果的 dtype）新建内存数组。
        dtype : np.dtype, optional
            ``out`` 为 None 时新建输出的 dtype。

        Returns
        -------
        np.ndarray
            拼接后的结果（传了 ``out`` 则返回 ``out`` 本身）。

        Notes
        -----
        输入为空时不调用 ``func``，直接返回 ``out``（未提供时返回新建的空数组）。
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
        逐块把 ``src`` 拷进 ``dst``，峰值内存为一块。

        ``dst`` 可以是普通 ndarray 或 ``np.memmap``；典型用法是从大输入填充磁盘
        工作副本，而不额外物化一份全量拷贝。

        Parameters
        ----------
        dst : np.ndarray
            目标数组，已预分配且形状与 ``src`` 兼容。
        src : array-like
            源数组，必须能广播赋值到 ``dst[start:end]`` 切片。
        chunk_size : int
            块长度（元素数），默认 :data:`DEFAULT_FILL_CHUNK_SIZE`。

        Returns
        -------
        np.ndarray
            填好的 ``dst``。
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
        """逐块把 ``source`` 流式写入 ``fp``，峰值内存为一块。

        块大小取 ``Chunk.chunk_size``（受全局内存预算约束），实际写入交给
        ``Chunk.chunked_fill``。"""
        src_nbytes = getattr(source, "nbytes", 0)
        nbytes = max(fp.nbytes, src_nbytes)

        chunk_size = cls.chunk_size(nbytes)
        cls.chunked_fill(fp, source, chunk_size)
