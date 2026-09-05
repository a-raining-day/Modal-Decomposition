"""
Python version:  (must)
    3.10

Lib and Version:  (if None write None)
    typing - None
	abc    - None

Only accessed by:  (must)
    All

Description: (if None write None)
    The defination of the abstrct class of the decomposition method.

Modify:  (must)
    2026.7.7 - Create, and create the abstract class.
"""

import numpy as np
from typing import Tuple, Protocol, Dict, Any
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from .ConstDefine import Name, Reference
from .ConfigDefine import *

__all__ = ['DecompositionResult', 'Decomposer']

@dataclass(frozen=True)
class DecompositionResult:
    IMFs: np.ndarray
    Res: np.ndarray
    info: dict[str, Any] = field(default_factory=dict)
    config: 'Config' = field(default_factory=lambda: Config())  # 允许空配置

class Decomposer(ABC):
    name: str

    @abstractmethod
    def decompose(self, S: list | np.ndarray, **kwargs) -> DecompositionResult:
        """
        Return IMFs, Res, and Info(other information, such as: energy, convergence)
        :param S: Signal
        :return: The struct of return's information
        """
        pass

    @property
    def name(self) -> str:
        full_name = Name.get(self.__class__.name, None)
        if full_name is None:
            raise ValueError("The full of this method is not exist!")
        return full_name

    @property
    def refs(self) -> str:
        reference = Reference.get(self.__class__.name, None)
        if reference is None:
            raise ValueError("The reference of this method is not exist!")
        return reference