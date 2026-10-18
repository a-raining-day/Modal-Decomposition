"""
Shared pytest configuration for the Modal-Decomposition test suite.

把仓库的 ``src/`` 加入 ``sys.path`` (包未安装时也能直接 ``pytest tests/``),
并在每个用例结束后重置进程级全局种子。
"""

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np
import pytest

import Modal_Decomposition

try:
    from _cases import OPTIONAL_DEPENDENCIES
except ImportError:                      # pytest 未把 tests/ 加入 sys.path 时
    from tests._cases import OPTIONAL_DEPENDENCIES


def pytest_collection_modifyitems(config, items):
    """
    可选依赖缺失时自动跳过相关方法的全部用例 (用例 id 里带 ``[METHOD]``)。
    """
    import importlib.util

    missing = {
        method: dep
        for method, dep in OPTIONAL_DEPENDENCIES.items()
        if importlib.util.find_spec(dep) is None
    }
    if not missing:
        return
    for item in items:
        for method, dep in missing.items():
            if f"[{method}]" in item.nodeid:
                item.add_marker(pytest.mark.skip(reason=f"可选依赖 {dep!r} 未安装"))
                break


@pytest.fixture(scope="session")
def signal():
    """Two-tone signal with light noise, length 256."""
    rng = np.random.default_rng(0)
    t = np.linspace(0, 1, 256, endpoint=False)
    return (
        np.sin(2 * np.pi * 5 * t)
        + 0.5 * np.sin(2 * np.pi * 20 * t)
        + 0.1 * rng.standard_normal(256)
    )


@pytest.fixture(scope="session")
def multichannel_signal(signal):
    """Two-channel signal for MEMD."""
    return np.stack([signal, signal[::-1]], axis=0)


@pytest.fixture(autouse=True)
def _reset_global_seed():
    """把进程级全局种子恢复成默认值, 避免用例之间互相污染。"""
    yield
    setter = getattr(Modal_Decomposition, "set_seed", None)
    if setter is not None:
        setter(42)
        return
    try:
        from Modal_Decomposition.Base import Seed
        Seed().GLOBAL_SEED = 42
    except (ImportError, AttributeError):
        return
