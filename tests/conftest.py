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
