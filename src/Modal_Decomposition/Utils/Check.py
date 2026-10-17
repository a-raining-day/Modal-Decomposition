"""
本模块仅 Check_Time_and_Signal 主要负责对上级使用
Check 主要用于内部

因此, 本模块必须自实现, 可延迟导入同属同级子模块, 但是延迟导入仍然可能导致循环导入, 使用时必须谨慎!
Base 为不同级模块, 可使用
"""

import os
import warnings
from enum import Enum
from typing import Optional, Any, Type, Literal

import numpy as np

from ..Define import DEFAULT_NUMPY_TYPE, SIZE, SAFE_MODS
from ..Base import CHUNK_MOD, Chunk, should_use_memmap, temp_memmap


__all__ = ["Check", "Check_Time_and_Signal"]


LIST_MEM_THRESHOLD = 10 * SIZE["1MB"]  # elements: list/tuple above this become memmaps


class Check:
    """
    针对已载入数组 (np.ndarray | np.memmap) 的原子检测与规范化。

    约定：
      is_* / has_*   只读谓词, 返回 bool, 不抛异常
      require_*      只读校验, 失败抛 ValueError, 成功静默
      to_*           唯一允许改数据的操作, 名字即行为

    不做 IO, 不触发全局内存策略, 不 emit warning。
    """

    # 返回各类型
    @classmethod
    def return_dtype_kind(cls, S: np.ndarray | np.memmap) -> str:
        if not cls.is_numeric(S):
            raise TypeError("the type of input must be np.ndarray, not {}".format(type(S)))

        match S.dtype.kind:
            case 'f':
                return "float"

            case 'i':
                return "int"

            case 'u':
                return "uint"

            case 'b':
                return "bool"

            case 'c':
                return "complex"

            case 'm':
                return "timedelta"

            case 'M':
                return "datatime"

            case 'O':
                return "object"

            case 'S':
                return "string"

            case 'T':
                return "string"

            case 'U':
                return "unicode"

            case 'V':
                return "void"

            case _:
                raise ValueError("the unknown kind of S.dtype -> {}".format(S.dtype.kind))

    # 报错
    @staticmethod
    def not_numpy_arr(x: Any, method: str="", raise_error: bool=True) -> bool:
        prefix = f"{method}: " if method else ""
        if not isinstance(x, np.ndarray):
            if raise_error:
                raise TypeError("{}The type of input must be np.ndarray, not {}".format(prefix, type(x)))
            else:
                return False
        return True

    @staticmethod
    def not_numpy_memmap(x: Any, method: str="", raise_error: bool=True) -> bool:
        prefix = f"{method}: " if method else ""
        if not isinstance(x, np.memmap):
            if raise_error:
                raise TypeError("{}The type of input must be np.ndarray, not {}".format(prefix, type(x)))
            else:
                return False
        return True

    # 只读
    @classmethod
    def is_numeric(cls, S) -> bool:
        """dtype 为 bool / int / uint / float 之一。"""
        return S.dtype.kind in "biuf"

    @classmethod
    def is_real(cls, S) -> bool:
        """dtype 为 int / uint / float 之一 (不含 bool)。"""
        return S.dtype.kind in "iuf"

    @classmethod
    def is_0d(cls, S: np.ndarray) -> bool:
        """
        与 is_1d, is_2d 不同, is_0d 是指标量, 如 np.array(42)
        :param S:
        :return:
        """
        if S.ndim == 0 and S.size != 0:
            return True
        else:
            return False

    @classmethod
    def is_1d(cls, S: np.ndarray) -> bool:
        return S.ndim == 1

    @classmethod
    def is_2d(cls, S: np.ndarray) -> bool:
        return S.ndim == 2

    @classmethod
    def is_empty(cls, S) -> bool:
        return S.size == 0

    @classmethod
    def is_duplicates(cls, S: np.ndarray, axis: int=0, temp_dir: Optional[str]=None) -> bool:
        """是否存在相邻重复值。分块遍历, 含跨块边界"""
        if cls.is_0d(S):
            return True

        mod = CHUNK_MOD.ExoMemory if should_use_memmap(S.nbytes) else CHUNK_MOD.Memory

        prev_last = None
        chunk_size = Chunk.chunk_size(S.nbytes)

        FIRST: tuple = (slice(None),) * axis + (slice(0, 1, 1),)
        LAST: tuple  = (slice(None),) * axis + (slice(-1, -2, -1),)

        for chunk in Chunk.chunk(S, chunk_size=chunk_size, mod=mod, axis=axis, temp_dir=temp_dir):
            if prev_last is not None and chunk[FIRST] == prev_last:
                return True

            if np.any(np.diff(chunk) == 0):
                return True

            prev_last = chunk[LAST]

        return False

    @classmethod
    def is_uniform(cls, S: np.ndarray, *, rtol: float = 1e-10, atol: float = 1e-14, temp_dir: Optional[str]=None) -> bool:
        """是否等间距。分块遍历, 含跨块边界差"""

        if cls.is_0d(S):
            return True

        if not cls.is_1d(S):
            S = S.squeeze()
            if not cls.is_1d(S):
                raise ValueError("the dim of the input must be 1-dim!")

        mod = CHUNK_MOD.ExoMemory if should_use_memmap(S.nbytes) else CHUNK_MOD.Memory

        d0 = None
        prev_last = None
        chunk_size = Chunk.chunk_size(S.nbytes)
        for chunk in Chunk.chunk(S, mod=mod, chunk_size=chunk_size, temp_dir=temp_dir):
            if prev_last is not None:
                d = chunk[0] - prev_last
                if d0 is None:
                    d0 = d
                elif not np.isclose(d, d0, rtol=rtol, atol=atol):
                    return False
            if chunk.size >= 2:
                diffs = np.diff(chunk)
                if d0 is None:
                    d0 = diffs[0]
                if not np.allclose(diffs, d0, rtol=rtol, atol=atol):
                    return False
            prev_last = chunk[-1]
        return True

    @classmethod
    def is_infinite(cls, S: np.ndarray | np.memmap, chunk_size: int = None) -> bool:
        """`True` means finite, `False` means infinite."""
        if should_use_memmap(S.nbytes):
            chunk_size = Chunk.chunk_size(S.nbytes) if chunk_size is None else chunk_size
            for chunk in Chunk.chunk(S, chunk_size=chunk_size):
                if not np.all(np.isfinite(chunk)):
                    return False

            return True

        else:
            return np.all(np.isfinite(S))

    # 校验 (抛错)
    @staticmethod
    def require_numeric(S, method: str = "", *, allow_bool: bool = True) -> None:
        prefix = f"{method}: " if method else ""
        if not isinstance(S, np.ndarray):
            raise ValueError(f"{prefix}expected np.ndarray, got {type(S).__name__}")
        allowed = "biuf" if allow_bool else "iuf"
        if S.dtype.kind not in allowed:
            raise ValueError(f"{prefix}unsupported non-numeric dtype {S.dtype}")

    @staticmethod
    def require_ndim(S, allowed, method: str = "") -> None:
        prefix = f"{method}: " if method else ""
        if S.ndim not in allowed:
            raise ValueError(
                f"{prefix}expected input dimension in {sorted(allowed)}, got {S.ndim}"
            )

    @staticmethod
    def require_nonempty(S, method: str = "") -> None:
        prefix = f"{method}: " if method else ""
        if S.size == 0:
            raise ValueError(f"{prefix}array must not be empty")

    @staticmethod
    def require_1d(S, method: str = "") -> None:
        prefix = f"{method}: " if method else ""
        if S.ndim != 1:
            raise ValueError(f"{prefix}expected 1-d array, got {S.ndim}-d")

    @staticmethod
    def require_length_match(T, S, method: str = "") -> None:
        prefix = f"{method}: " if method else ""
        if T.shape[0] != S.shape[-1]:
            raise ValueError(
                f"{prefix}length mismatch between S ({T.shape[0]}) and S ({S.shape[-1]})"
            )

    # 检测输入合规性
    @staticmethod
    def validate_safe(safe: str | SAFE_MODS) -> SAFE_MODS:
        try:
            return SAFE_MODS(safe)
        except ValueError:
            raise ValueError(
                "the parameter of `safe` can only be ['safe', 'suspicious', 'unknown']"
            )

    @staticmethod
    def validate_mod(mod: str | Type[Enum], MOD: Type[Enum]):
        try:
            return MOD(mod)
        except ValueError:
            raise ValueError(
                f"the parameter of `mod` can only be {MOD._member_names_}"
            )

    #  规范化 (改数据)
    @classmethod
    def to_array(cls, S, *, dtype: type | np.dtype = None, safe: str | SAFE_MODS = "unknown", ndim: int | set | tuple | None = 1, need_squeeze: bool = True) -> np.ndarray:
        """
        规范化任意 array-like 为 ndarray, 可选转 dtype。

        处理顺序 (顺序关键, 保证 memmap 不被物化):
          1. 非 ndarray → np.asarray
          2. 拒绝非数值 dtype (kind ∉ "biuf")
          3. 拒绝 0-d
          4. 若 dtype 给定且与当前不同:
               - memmap 走 temp_memmap + Chunk.fill_memmap 分块写 (不物化)
               - 普通 ndarray 走 astype(copy=False)
          5. squeeze 单例维; 若变 0-d 抛错

        Parameters
        ----------
        S : array-like
            输入。
        dtype : np.dtype | str | None
            目标 dtype。None 时保留原 dtype, 仅做规范化。

        Returns
        -------
        np.ndarray
            规范化后的数组。若输入是 memmap 且未触发 squeeze 变形,
            返回值仍可能由磁盘支撑; 否则为普通 ndarray。

        Raises
        ------
        ValueError
            非数值 dtype、0-d 输入、或 squeeze 后变 0-d。
        """
        safe = cls.validate_safe(safe)
        if safe == SAFE_MODS.Safe:
            if dtype is not None:
                S = cls.to_dtype(S, dtype)

            return S
        else:
            # 1. array-ify
            if not isinstance(S, np.ndarray):
                S = np.asarray(S, dtype=DEFAULT_NUMPY_TYPE if dtype is None else dtype)

            # 2. dtype 合法性
            if not cls.is_numeric(S):
                raise ValueError(f"unsupported non-numeric dtype {S.dtype}")

            # 3. dtype 转换 —— 必须在 squeeze 之前
            #    squeeze 会返回普通 ndarray 视图 (不再是 np.memmap), 之后再 astype 就会绕过 fill_memmap, 把整条数组拉回 RAM
            if dtype is not None:
                S = cls.to_dtype(S, dtype)

            if safe == SAFE_MODS.Suspicious:
                return S

            else:
                # 4. 维度检查 —— ndim 允许 int | 维数集合 | None(不检查)
                #    Check_Time_and_Signal 的契约是 set[int], 库内分解方法均以
                #    ndim={1} / ndim={1, 2} 调用, 这里统一收口
                if ndim is None:
                    pass
                elif isinstance(ndim, int) and not isinstance(ndim, bool):
                    if not (ndim >= 1):
                        raise ValueError("ndim must be >= 1")
                    ndim = (ndim,)
                elif isinstance(ndim, (set, frozenset, list, tuple)):
                    if not ndim:
                        raise ValueError("ndim must not be empty")
                    if any((not isinstance(d, int)) or isinstance(d, bool) or d < 1 for d in ndim):
                        raise ValueError("ndim must contain ints >= 1")
                    ndim = tuple(ndim)
                else:
                    raise TypeError("ndim must be an int, a collection of ints, or None")

                if S.ndim == 0:
                    if S.size != 0:
                        S = S.reshape(1,)

                    else:
                        raise ValueError("Signal must have at least 1 dimension, got 0-d array")

                # 5. 有限性检查
                if Check.return_dtype_kind(S) == "float" and not Check.is_infinite(S):
                    raise ValueError("NaN or Inf found")

                if S.ndim == 1:
                    if ndim is None or 1 in ndim:
                        return S
                    else:
                        raise ValueError(f"the input is 1-dim! not {ndim}")

                if not isinstance(need_squeeze, bool):
                    raise ValueError("need_squeeze must be bool")

                if need_squeeze:
                    S = S.squeeze()
                    if S.ndim == 0:
                        raise ValueError("Signal must have at least 1 dimension after squeeze")

                if ndim is not None and S.ndim not in ndim:
                    raise ValueError(f"the ndim of input -> {S.ndim} != {ndim}")

        return S

    @staticmethod
    def to_dtype(S: np.ndarray | np.memmap, dtype: np.dtype) -> np.ndarray | np.memmap:
        """
        转 dtype。memmap 走 temp_memmap + Chunk.fill_memmap 分块写,
        绝不 .astype —— 那会把整条数组拉回 RAM, 破坏全局内存策略。
        """
        try:
            dtype = np.dtype(dtype)
        except TypeError:
            raise TypeError("dtype must be np.dtype or float, int...")

        if S.dtype == dtype:
            return S

        if isinstance(S, np.memmap):
            fp = temp_memmap(S.shape, dtype)
            if fp is None:
                raise BaseException("it's failed to create memmap")

            Chunk.fill_memmap(fp, S)
            return fp

        return S.astype(dtype, copy=False)


def Check_Time_and_Signal(
    S, T=None,
    ndim=None,
    method: str = "",
    default_T: bool = True,
    dtype: np.dtype = None,
    axis: int = 0,
    need_T: bool = True,
    chunk_size: int = None,
    temp_dir: Optional[str] = None
) -> tuple[np.ndarray, Optional[np.ndarray], int] | tuple[np.ndarray, int]:
    """
    Validate the signal and its now axis and return them in canonical form.

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
        When True (default) and ``S`` is None, build ``arange(N, float64)``.
        When False, leave ``S`` as None: backends that rebuild their own
        timeline (e.g. the PyEMD sifters, which ignore a passed now axis
        under the default extrema detection) then avoid a one-signal-sized
        transient allocation. Callers must only set this when their algorithm
        does not use the returned ``S``.
    dtype : {"float64", None}, optional
        Target dtype of the returned signal. Default ``None``: **keep the
        input's dtype** (bool/int/float 按原样, 校验为实数数值即可) —— 不再
        默认转 float64 以避免 f16/f32 输入的内存翻倍。传 ``"float64"`` 时
        才做统一 float64 转换 (大输入按内存策略走分块磁盘转换)。``S`` 遵循
        同一策略 (缺省生成的 ``arange(N)`` 保持 int64)。
    axis : int, optional
        the checking axis for the signal at least 2-dim.
    chunk_size : int, optional
        the size of chunk
    temp_dir : str, optional
        the temporary dir of the tempfiles.

    Returns
    -------
    tuple[np.ndarray, Optional[np.ndarray], int] | tuple[np.ndarray, int]
        ``need_T=True`` (默认): ``(S, T, N)``。``S`` 默认保留输入 dtype (可能
        memmap 支撑); 显式 ``dtype="float64"`` 时为 float64; ``T`` 为时间轴,
        未给且 ``default_T=False`` 时为 None; ``N`` 为末轴长度。
        ``need_T=False``: ``(S, N)`` —— 不构造 ``T``, 因而不产生
        ``arange(N)`` 这一份与信号等长的临时数组。算法不使用时间轴时应当选它。

    Raises
    ------
    ValueError
        On wrong dimensionality, length mismatch, duplicate now points, an
        unsupported file suffix, empty signal, or an invalid ``dtype`` value.
    """
    from .Monotonicity import monotonic

    S: list | tuple | np.ndarray | np.memmap = _load(S, target_dtype=dtype)  # load
    S: np.ndarray | np.memmap = Check.to_array(S=S, dtype=dtype, safe="unknown", ndim=ndim)  # transform the (list, tuple, set) to np.ndarray or np.memmap

    N = S.shape[-1]
    if Check.is_empty(S):
        raise ValueError("Signal must not be empty")

    if T is None:
        if default_T and need_T:
            T = np.arange(N, dtype=S.dtype)
        # else: keep S as None (see default_T above).
    else:
        T = Check.to_array(S=T, dtype=S.dtype, ndim=1)
        Check.require_1d(T, method=method)
        Check.require_length_match(T, S, method=method)

        # 分块检查重复与降序 (检的是时间轴 T, 不是信号 S)
        duplicate = Check.is_duplicates(T, axis=axis, temp_dir=temp_dir)
        if duplicate:
            raise ValueError("Time axis contains duplicate values")

        if not monotonic(T, strict=True, chunk_size=chunk_size if chunk_size is not None else Chunk.chunk_size(T.nbytes), mod="increasing", safe="safe"):
            warnings.warn(
                "S is not monotonically increasing with T; S and T are reordered by ascending T.",
                UserWarning,
                stacklevel=2,
            )

            # TODO: 这部分面对 np.memmap 能做好吗?
            idx = np.argsort(T)
            T = T[idx]
            S = S[..., idx]

    if need_T:
        return S, T, N
    else:
        return S, N


def _load(S, target_dtype=None) -> np.ndarray | np.memmap | list | tuple:
    """
    统一外部输入 -> np.ndarray | np.memmap。

    这是整个包唯一涉及文件 IO 和全局内存策略的地方。
      - np.memmap       : 原样返回
      - 过大的 ndarray  : 分块拷入临时 memmap（target_dtype 在填充时一并完成）
      - .npy 路径       : 按策略 mmap_mode="r" 或全量读入
      - 过长的 list/tuple: 分块填入临时 memmap
      - 其他            : 原样返回，交给 Check 报错
    """
    if isinstance(S, np.memmap):
        return S

    if isinstance(S, np.ndarray):
        if S.ndim >= 1 and S.size > 0 and should_use_memmap(S.nbytes):
            out_dtype = target_dtype if target_dtype is not None else S.dtype
            # 形参顺序是 (shape, dtype) —— 写反了会静默建不出 memmap
            fp = temp_memmap(S.shape, out_dtype)
            if fp is None:
                return S            # 建不起来就退回内存, 不把 None 交给 fill_memmap
            Chunk.fill_memmap(fp, S)
            return fp
        return S

    if isinstance(S, (str, os.PathLike)):
        path = os.fspath(S)
        if not path.endswith(".npy"):
            raise ValueError(f"only .npy files are supported, got {path!r}")
        if should_use_memmap(os.path.getsize(path)):
            return np.load(path, mmap_mode="r")
        return np.load(path)

    if isinstance(S, (list, tuple)) and len(S) > LIST_MEM_THRESHOLD:
        out_dtype = target_dtype if target_dtype is not None else DEFAULT_NUMPY_TYPE
        fp = temp_memmap((len(S),), out_dtype)      # 形参顺序 (shape, dtype)
        if fp is None:
            return S
        Chunk.fill_memmap(fp, S)
        return fp

    return S