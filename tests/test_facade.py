"""
Facade equivalence and docstring composition tests.
"""

import numpy as np
import pytest

from Modal_Decomposition import Class, Function

from _cases import CASES, METHODS


@pytest.mark.parametrize("method", METHODS)
def test_facade_matches_class(method, signal):
    params = CASES[method]
    if method == "MEMD":
        s2 = np.stack([signal, signal[::-1]])
        r_facade = Function.MEMD(s2, **params)
        r_class = Class.MEMD(**params).decompose(s2)
    else:
        r_facade = getattr(Function, method)(signal, **params)
        r_class = getattr(Class, method)(**params).decompose(signal)

    assert np.allclose(r_facade.IMFs, r_class.IMFs)
    if r_class.Res is not None:
        assert np.allclose(r_facade.Res, r_class.Res)
    else:
        assert r_facade.Res is None
    assert r_facade.config.to_dict() == r_class.config.to_dict()


def test_facade_docstring_with_seed():
    doc = Function.FMD.__doc__
    assert "Filtered Mode Decomposition" in doc
    assert "Parameters" in doc
    assert "References" in doc
    assert "10.1109/TIE.2022.3156156" in doc
    assert "Random seed" in doc


def test_facade_docstring_without_seed():
    doc = Function.EMD.__doc__
    assert "Empirical Mode Decomposition" in doc
    assert "Parameters" in doc
    assert "References" in doc
    assert "Random seed" not in doc


def test_facade_docstring_contains_config_type():
    assert "EMDConfig" in Function.EMD.__doc__
