"""
Core contracts: the decomposition result type and the decomposer base class.
"""

import copy
import inspect
import warnings
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar

from .ConfigDefine import Config
from .TextDefine import Name, Reference
from ..Error import PreError

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

    def reconstruct(self) -> np.ndarray:
        """
        Reconstruct the signal as the sum of all modes plus the residual.

        A None residual contributes zero.
        """
        recon = np.sum(self.IMFs, axis=0)
        if self.Res is not None:
            recon = recon + self.Res
        return recon


class Decomposer(ABC):
    """
    Base class of all decomposition methods.

    Subclasses must set the ``name`` class attribute to their registry key
    and implement ``decompose``.
    """

    name: ClassVar[str]

    def __init__(self):
        self.__snapshot = None
        self.__config_construct_success = False

    @abstractmethod
    def decompose(self, S, T=None) -> DecompositionResult:
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

    def config_construct_start(self) -> None:
        internal = {"_Decomposer__snapshot", "_Decomposer__config_construct_success"}

        self.__snapshot = set(vars(self)) - internal
        self.__config_construct_success = True

    def config_construct_end(self) -> dict:
        if (not self.__config_construct_success) or (self.__snapshot is None):
            raise PreError(
                "the pre-config isn't constructed yet, use `config_construct_start` at the beginning of __init__"
            )

        missing = [k for k in self.__snapshot if not hasattr(self, k)]
        if missing:
            raise PreError(
                f"the following config keys were recorded at __init__, but are not set as attributes on self: {missing}"
            )

        return {k: getattr(self, k) for k in self.__snapshot}

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
