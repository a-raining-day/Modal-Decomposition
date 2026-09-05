"""
Registry and metadata consistency tests.
"""

import pytest

from Modal_Decomposition import Class, Function
from Modal_Decomposition.Base import Decomposer, Name, Reference
from Modal_Decomposition._Registry import register_class

from _cases import METHODS


def test_registry_keys_match_method_list():
    assert set(Class.__dict__) == set(METHODS)


def test_registry_keys_match_metadata_tables():
    assert set(Class.__dict__) == set(Name)
    assert set(Class.__dict__) == set(Reference)


def test_function_namespace_matches_registry():
    assert set(Function.__dict__) == set(Class.__dict__)


def test_all_entries_are_decomposer_subclasses():
    for m in METHODS:
        cls = getattr(Class, m)
        assert issubclass(cls, Decomposer), f"{m}: not a Decomposer subclass"
        assert cls.name == m, f"{m}: name attribute mismatch"


def test_full_name_and_reference_lookup():
    for m in METHODS:
        cls = getattr(Class, m)
        instance = cls()
        assert instance.full_name == Name[m]
        assert instance.reference == Reference[m]


def test_register_class_rejects_duplicate():
    with pytest.raises(ValueError):
        @register_class("CEEMD")
        class _Duplicate:
            name = "CEEMD"


def test_register_class_rejects_name_mismatch():
    with pytest.raises(ValueError):
        @register_class("NOPE")
        class _Mismatch:
            name = "OTHER"


def test_register_class_rejects_non_class():
    with pytest.raises(TypeError):
        register_class("FUNC")(lambda: None)


def test_call_alias_equals_decompose(signal):
    result_call = Class.EMD()(signal)
    result_direct = Class.EMD().decompose(signal)
    import numpy as np

    assert np.allclose(result_call.IMFs, result_direct.IMFs)
    assert np.allclose(result_call.Res, result_direct.Res)
