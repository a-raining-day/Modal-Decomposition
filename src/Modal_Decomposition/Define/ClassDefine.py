"""
Core contracts: the decomposition result type and the decomposer base class.
"""


import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar

from .ConfigDefine import Config
from .TextDefine import Name, Reference

__all__ = ["DecompositionResult", "Decomposer"]


@dataclass(frozen=True)
class DecompositionResult:
    """
    Unified return type of every decomposition method.

    Attributes
    ----------
    IMFs : np.ndarray
        Decomposed modes. Shape (K, N) for univariate methods and
        (K, d, N) for multivariate methods.
    Res : np.ndarray | None
        Residual. None for methods without a residual concept.
    info : dict[str, Any]
        Method-specific diagnostics. Always a dict.
    config : Config
        Effective parameter snapshot of this run. Never None.
    """

    IMFs: np.ndarray
    Res: np.ndarray | None
    info: dict[str, Any] = field(default_factory=dict)
    config: Config = field(default_factory=Config)

    @property
    def n_imfs(self) -> int:
        """
        Number of decomposed modes.
        """
        return int(self.IMFs.shape[0])

    @property
    def shape(self) -> tuple:
        """
        Shape of the IMFs array.
        """
        return tuple(self.IMFs.shape)


def check_output_memmap(output_memmap) -> bool:
    """
    校验 ``decompose`` 的 ``output_memmap`` 形参。

    只接受真正的 ``bool``（``0``/``1`` 这类整数不放过），否则 ``TypeError``。
    返回规范化后的布尔值，便于调用方直接使用。
    """
    if type(output_memmap) is not bool:
        raise TypeError(
            f"output_memmap must be bool, got {type(output_memmap).__name__}"
        )
    return output_memmap


def finalize_output(result: "DecompositionResult", output_memmap, persist: bool = False) -> "DecompositionResult":
    """
    按需把结果里的数组落到临时 memmap 上。

    ``output_memmap=False`` 时原样返回（零开销）。为 ``True`` 时新建磁盘后备的
    ``np.memmap`` 并把数据拷进去：结果不再常驻内存，适合大数组或需要长时间保留
    结果的场景。转成 memmap 的范围是：
    * ``IMFs`` 与 ``Res``；
    * ``info`` 里**所有** ``np.ndarray``（含嵌套 dict / list / tuple 内的），
      这样诊断用的谱、历史轨迹等同样不会撑爆内存。

    ``persist=True`` 时后备文件不登记进 ``Base.Memory`` 的清理表，由调用方持有
    （见 :func:`Base.Memory.temp_memmap`）。

    0 维数组、标量、字符串、``None`` 保持原样；平台不支持 memmap 时
    ``Base.Memory.temp_memmap`` 返回 ``None``，此时该数组也保持原 ndarray。
    """
    if not check_output_memmap(output_memmap):
        return result
    if type(persist) is not bool:
        raise TypeError(f"persist must be bool, got {type(persist).__name__}")

    from ..Base.Memory import temp_memmap

    def _to_memmap(obj):
        if isinstance(obj, np.ndarray):
            if obj.ndim == 0:          # 0 维无法映射, 保留原样
                return obj
            mm = temp_memmap(obj.shape, dtype=obj.dtype, persist=persist)
            if mm is None:
                return obj
            mm[...] = obj
            return mm
        if isinstance(obj, dict):
            return {k: _to_memmap(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_to_memmap(v) for v in obj]
        if isinstance(obj, tuple):
            return tuple(_to_memmap(v) for v in obj)
        return obj

    return DecompositionResult(
        IMFs=_to_memmap(result.IMFs),
        Res=_to_memmap(result.Res),
        info=_to_memmap(result.info),
        config=result.config,
    )


class Decomposer(ABC):
    """
    Base class of all decomposition methods.

    Subclasses must set the ``name`` class attribute to their registry key
    and implement ``decompose``.
    """

    name: ClassVar[str]

    @abstractmethod
    def decompose(self, S) -> DecompositionResult:
        """
        Decompose the signal.

        Parameters
        ----------
        S : array-like
            Signal.
        T : array-like, optional
            Time axis. If None, a default index axis is used.

        Returns
        -------
        DecompositionResult
            Unified decomposition result.
        """

    @abstractmethod
    def param_check(self) -> None:
        """
        Check the parameters of the init. If the init method has problem, it will raise or warning.
        :return:
        """

    @abstractmethod
    def config_generate(self) -> Config:
        """
        Generate the Config of this decomposition method.
        :return:
        """

    @property
    def full_name(self) -> str:
        """
        Full descriptive name of the method from the Name table.
        """
        try:
            return Name[self.name]
        except KeyError:
            raise ValueError(f"No full name defined for method {self.name!r}")

    @property
    def reference(self) -> str:
        """
        Bibliographic reference (DOI) from the Reference table.
        """
        try:
            return Reference[self.name]
        except KeyError:
            raise ValueError(f"No reference defined for method {self.name!r}")
