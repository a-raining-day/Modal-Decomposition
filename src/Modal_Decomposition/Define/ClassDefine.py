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
