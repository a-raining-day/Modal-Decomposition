"""
Boundary tests for the merged single-loop implementations of FMD and LMD.
"""

import numpy as np
import pytest

from Modal_Decomposition import Class
from Modal_Decomposition.Base import DecompositionResult


# --- FMD ---

def test_fmd_complete_decomposition_terminates_and_reconstructs(signal):
    r = Class.FMD(K=-1, max_iter=3, num_hand=3, seed=0).decompose(signal)
    assert isinstance(r, DecompositionResult)
    assert r.IMFs.ndim == 2
    assert np.max(np.abs(r.reconstruct() - signal)) < 1e-9


def test_fmd_fixed_k_equals_k_minus_one_prefix(signal):
    full = Class.FMD(K=-1, max_iter=3, num_hand=3, seed=0).decompose(signal)
    fixed = Class.FMD(K=3, max_iter=3, num_hand=3, seed=0).decompose(signal)
    # the first K modes of the complete run equal the fixed-K run
    assert np.allclose(full.IMFs[:3], fixed.IMFs)


def test_fmd_empty_result_on_very_short_signal():
    r = Class.FMD(K=2).decompose(np.zeros(5))
    assert r.IMFs.shape == (0, 5)


def test_fmd_constant_signal_short_circuit():
    r = Class.FMD(K=2).decompose(np.ones(20))
    assert r.IMFs.shape == (0, 20)


# --- LMD ---

def test_lmd_complete_decomposition_terminates(signal):
    r = Class.LMD(max_pf=-1, max_iter=10).decompose(signal)
    assert isinstance(r, DecompositionResult)
    assert r.IMFs.ndim == 2


def test_lmd_fixed_count_equals_complete_prefix(signal):
    full = Class.LMD(max_pf=-1, max_iter=10).decompose(signal)
    fixed = Class.LMD(max_pf=2, max_iter=10).decompose(signal)
    assert np.allclose(full.IMFs[:2], fixed.IMFs)


def test_lmd_short_signal_raises():
    with pytest.raises(ValueError, match="length"):
        Class.LMD().decompose(np.ones(7))


def test_lmd_pure_sinusoid_produces_empty_imfs_and_residual(signal):
    t = np.linspace(0, 2 * np.pi, 256, endpoint=False)
    s = np.sin(t)
    r = Class.LMD(max_pf=2).decompose(s)
    assert r.IMFs.ndim == 2
    assert r.IMFs.shape == (0, 256)
    assert np.max(np.abs(r.reconstruct() - s)) < 1e-9
