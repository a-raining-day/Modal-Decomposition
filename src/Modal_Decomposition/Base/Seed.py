"""
全局种子：进程内唯一的 ``Seed`` 单例负责读写全局种子，并提供临时覆盖的上下文管理器。
"""

import threading
import contextlib
from warnings import warn
from typing import Generator

__all__ = ["Seed", "Global_Seed"]


class Seed:
    """
    全局种子的唯一持有者：``GLOBAL_SEED`` 是类属性，全进程共享，单例只保证
    ``Seed()`` 拿到同一实例。

    临时覆盖走 ``temporary_seed``；``set(lock=True)`` 之后类级修改会抛 ``ValueError``。
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
        """全局种子是否已被锁定（锁定时 ``set`` 抛 ``ValueError``）。"""
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
        设置全局种子并返回设置后的值。

        :param seed: 非负整数。
        :param lock: 为 True 时同时锁定，之后 ``set`` 抛 ``ValueError``。
        :param verbose: 为 True 时覆盖已有种子会发 ``UserWarning``。
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
        """返回当前的全局种子。"""
        with cls._lock:
            return cls.GLOBAL_SEED

    @property
    def seed(self):
        """当前全局种子（与 ``Seed.get()`` 同值）。"""
        return self.get()

    @staticmethod
    @contextlib.contextmanager
    def temporary_seed(seed: int) -> Generator[None, None, None]:
        """
        在 ``with`` 块内临时覆盖全局种子，退出时（含异常路径）恢复原值。

        多线程并发时该窗口会带来隐式的复现性破坏，不建议在并发路径上使用。

        必须写类属性 ``Seed.GLOBAL_SEED``；写 ``Seed().GLOBAL_SEED = seed`` 只会给
        单例实例加一个遮蔽属性，类方法 ``Seed.get()`` 读的仍是类属性，整个上下文
        管理器会退化成 no-op。
        """
        old_seed = Seed.GLOBAL_SEED

        seed = Seed().validate(seed)
        Seed.GLOBAL_SEED = seed
        try:
            yield
        finally:
            Seed.GLOBAL_SEED = old_seed

# 导入时的种子快照；之后 ``Seed.set`` 不会更新这个变量。
Global_Seed = Seed.get()
