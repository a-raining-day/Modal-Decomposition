"""
Shared configuration for the decomposition memory tests.

Restores the package default memory policy after every test, since the
measured runs temporarily switch the global policy (memmap forcing).

注: 旧的 ``Utils.set_memmap_ratio`` 已随重构移除, 现行接口是
``Base.MemoryPolicy.set_memory_reserve``。
"""

import pytest

from Modal_Decomposition.Base import MemoryPolicy

_DEFAULT_RESERVE = 0.25


@pytest.fixture(autouse=True)
def _reset_memory_policy():
    yield
    MemoryPolicy.set_memory_reserve(_DEFAULT_RESERVE)
