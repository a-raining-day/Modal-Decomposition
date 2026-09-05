"""
Unified return contract tests: every method returns DecompositionResult
with the agreed field semantics.
"""

import numpy as np
import pytest

from Modal_Decomposition import Class
from Modal_Decomposition.Base import DecompositionResult
from Modal_Decomposition.Error import RealizationError

from _cases import CASES, METHODS


def _run(method, signal):
    params = CASES[method]
    if method == "MEMD":
        s2 = np.stack([signal, signal[::-1]])
        return getattr(Class, method)(**params).decompose(s2), s2, 3
    return getattr(Class, method)(**params).decompose(signal), signal, 2


@pytest.mark.parametrize("method", METHODS)
def test_result_type(method, signal):
    result, _, _ = _run(method, signal)
    assert isinstance(result, DecompositionResult)


@pytest.mark.parametrize("method", METHODS)
def test_imfs_shape(method, signal):
    result, s2, ndim = _run(method, signal)
    assert result.IMFs.ndim == ndim
    assert result.IMFs.shape[-1] == s2.shape[-1]
    if ndim == 3:
        assert result.IMFs.shape[1] == 2


@pytest.mark.parametrize("method", METHODS)
def test_res_field_semantics(method, signal):
    result, s2, _ = _run(method, signal)
    if result.Res is None:
        assert method in ("SSA", "VMD"), f"{method}: unexpected None Res"
    else:
        assert isinstance(result.Res, np.ndarray)
        assert result.Res.shape == tuple(s2.shape)


@pytest.mark.parametrize("method", METHODS)
def test_info_is_dict(method, signal):
    result, _, _ = _run(method, signal)
    assert isinstance(result.info, dict)


@pytest.mark.parametrize("method", METHODS)
def test_config_snapshot(method, signal):
    result, _, _ = _run(method, signal)
    assert result.config is not None
    data = result.config.to_dict()
    assert isinstance(data, dict)
    if method == "SSA":
        assert data["window_size"] == signal.size // 3
    if method in ("CEEMD", "CEEMDAN", "EEMD", "FMD", "ICEEMDAN"):
        assert data["seed"] == 0


@pytest.mark.parametrize("method", METHODS)
def test_derived_properties(method, signal):
    result, s2, _ = _run(method, signal)
    assert result.n_imfs == result.IMFs.shape[0]
    assert result.shape == tuple(result.IMFs.shape)
    assert result.reconstruct().shape == tuple(s2.shape)


def test_svmd_backend_validation():
    with pytest.raises(ValueError):
        Class.SVMD(backend="x")


def test_svmd_numba_backend_not_implemented(signal):
    with pytest.raises(RealizationError):
        Class.SVMD(backend="numba").decompose(signal)
