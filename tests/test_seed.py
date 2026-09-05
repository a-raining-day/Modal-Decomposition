"""
Two-level seed resolution tests.
"""

import numpy as np
import pytest

from Modal_Decomposition import Class
from Modal_Decomposition.Utils import get_seed, resolve_seed, set_seed


def test_priority_no_seeds():
    set_seed(None)
    assert resolve_seed(None, "T") == (None, False)


def test_priority_local_only():
    set_seed(None)
    assert resolve_seed(3, "T") == (3, False)


def test_priority_global_only():
    set_seed(7)
    assert resolve_seed(None, "T") == (7, False)


def test_priority_global_overrides_local():
    set_seed(7)
    with pytest.warns(UserWarning, match="overrides local seed 3"):
        assert resolve_seed(3, "T") == (7, True)


def test_warning_message_contains_both_seeds_and_method():
    set_seed(11)
    with pytest.warns(UserWarning, match="global seed 11 overrides local seed 5 for method CEEMD"):
        resolve_seed(5, "CEEMD")


def test_validation_negative():
    with pytest.raises(ValueError):
        set_seed(-1)


def test_validation_bool():
    with pytest.raises(TypeError):
        set_seed(True)


def test_validation_float():
    with pytest.raises(TypeError):
        set_seed(3.5)


def test_validation_string():
    with pytest.raises(TypeError):
        set_seed("abc")


def test_numpy_integer_accepted():
    set_seed(np.int64(4))
    assert get_seed() == 4


def test_get_set_roundtrip():
    set_seed(None)
    assert get_seed() is None
    set_seed(9)
    assert get_seed() == 9


def test_reproducibility_with_local_seed(signal):
    set_seed(None)
    r1 = Class.CEEMD(N_whitenoise=4, seed=9).decompose(signal).IMFs
    r2 = Class.CEEMD(N_whitenoise=4, seed=9).decompose(signal).IMFs
    assert np.allclose(r1, r2)


def test_global_seed_controls_rng_method(signal):
    set_seed(5)
    a = Class.FMD(K=1, max_iter=3).decompose(signal).IMFs
    set_seed(5)
    b = Class.FMD(K=1, max_iter=3).decompose(signal).IMFs
    assert np.allclose(a, b)


def test_global_seed_overrides_local_in_decompose(signal):
    set_seed(5)
    with pytest.warns(UserWarning, match="overrides local seed 3 for method CEEMD"):
        a = Class.CEEMD(N_whitenoise=4, seed=3).decompose(signal).IMFs
    set_seed(5)
    b = Class.CEEMD(N_whitenoise=4, seed=None).decompose(signal).IMFs
    assert np.allclose(a, b)
