"""
Global Seed

应当隶属于全局
"""

import threading
import contextlib
from warnings import warn
from typing import Generator

__all__ = ["Seed", "Global_Seed"]


class Seed:
    """
    全局仅该唯一类可以设置全局种子，该种子对全局覆盖且唯一
    但是提供临时上下文设置种子
    """
    GLOBAL_SEED = 42

    _lock = threading.Lock()
    _locked = False
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance

    @property
    def status(self) -> bool:
        with self._lock:
            return self._locked

    @staticmethod
    def validate(seed: int) -> int:
        if not isinstance(seed, int):
            raise TypeError(f"seed must be int, got {type(seed)}")

        try:
            value = seed.__index__()
        except TypeError:
            raise TypeError(f"seed must be an int or None, got {type(seed).__name__}")

        if value < 0:
            raise ValueError(f"seed must be non-negative, got {value}")

        return value

    @classmethod
    def set(cls, seed: int, lock: bool=False, verbose: bool=True) -> int:
        """
        Set the seed.
        :param seed: int of your seed.
        :param lock: if True, the global seed will be locked, cannot be modified.
        :param verbose: if True, warn won't remain silent.
        :return:
        """
        with cls._lock:
            seed = cls.validate(seed)

            if cls._locked:
                raise ValueError("GLOBAL_SEED is locked and cannot be modified.")

            if cls.GLOBAL_SEED is not None and verbose:
                warn("Overriding existing GLOBAL_SEED.")

            cls.GLOBAL_SEED = seed
            if lock:
                cls._locked = True

            return cls.GLOBAL_SEED

    @classmethod
    def get(cls) -> int:
        with cls._lock:
            return cls.GLOBAL_SEED

    @property
    def seed(self):
        return self.get()

    @staticmethod
    @contextlib.contextmanager
    def temporary_seed(seed: int) -> Generator[None, None, None]:
        """
        在多线程并发时，该窗口会带来隐式的复现性破坏。虽然实现了，但是不建议使用
        :param seed:
        :return:
        """
        old_seed = Seed().GLOBAL_SEED

        seed = Seed().validate(seed)
        Seed().GLOBAL_SEED = seed
        try:
            yield
        finally:
            Seed().GLOBAL_SEED = old_seed

Global_Seed = Seed.get()
