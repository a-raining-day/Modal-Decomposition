"""
Shared configuration for the decomposition memory tests.

Restores the package default memory policy after every test, since the
measured runs temporarily switch the global policy (memmap forcing).
"""

import pytest

from Modal_Decomposition.Utils import set_memmap_ratio


@pytest.fixture(autouse=True)
def _reset_memory_policy():
    yield
    set_memmap_ratio(0.6)
