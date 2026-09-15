"""
Shared pytest configuration for the Modal-Decomposition test suite.

Adds src_new to sys.path so tests import the rebuilt package, and provides
shared fixtures.
"""

import sys
from pathlib import Path

SRC_NEW = Path(__file__).resolve().parents[1] / "src_new"
if str(SRC_NEW) not in sys.path:
    sys.path.insert(0, str(SRC_NEW))

import numpy as np
import pytest

import src.Modal_Decomposition as Modal_Decomposition
# import Modal_Decomposition

from _cases import OPTIONAL_DEPENDENCIES


def pytest_collection_modifyitems(config, items):
    """
    可选依赖缺失时, 自动跳过相关方法的全部用例。

    ``EWTpy`` 需要可选第三方包 ``ewtpy``; 未安装时 (例如只装了
    ``pip install Modal-Decomposition`` 而未装 extras) 不该让整个测试套件失败,
    故在此集中打 skip, 而不是逐个测试文件写 importorskip。
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
    """Keep the process-level seed isolated between tests."""
    yield
    Modal_Decomposition.set_seed(None)
