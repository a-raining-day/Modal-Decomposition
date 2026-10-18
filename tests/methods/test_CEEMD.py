"""
``Modal_Decomposition.CEEMD`` 的统一契约测试。

覆盖: 注册 / 分解与返回契约 / 重构恒等式 / 门面等价 / 退化输入 /
输入不变异 / 确定性 / 参数校验。
"""

import numpy as np
import pytest

import Modal_Decomposition as MD
from Modal_Decomposition import Class, Function
from Modal_Decomposition.Define import DecompositionResult

METHOD = "CEEMD"
PARAMS = {"num_imf": 3, "M": 5}
HAS_RES = True
RECON_TOL = 1e-9


def _sig(n: int = 256, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, n, endpoint=False)
    return np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 20 * t) + 0.05 * rng.standard_normal(n)


def _run(seed: int = 42):
    MD.set_seed(seed, verbose=False)
    return getattr(Class, METHOD)(**PARAMS).decompose(_sig())


def test_registered_in_class_and_function():
    assert getattr(Class, METHOD) is not None
    assert getattr(Function, METHOD) is not None
    assert Function.__dict__[METHOD].__qualname__ == f"Function.{METHOD}"


def test_returns_decomposition_result():
    r = _run()
    assert isinstance(r, DecompositionResult)
    assert isinstance(r.IMFs, np.ndarray)
    assert r.IMFs.ndim == 2
    assert r.IMFs.shape[1] == 256
    assert r.IMFs.shape[0] >= 1
    assert isinstance(r.info, dict)
    assert r.config is not None
    assert isinstance(r.config.to_dict(), dict)


def test_config_snapshot_is_json_safe():
    import json
    r = _run()
    json.dumps(r.config.to_dict())


def test_reconstruction_identity():
    if not HAS_RES:
        pytest.skip("method has no residual concept (Res is None)")
    S = _sig()
    MD.set_seed(42, verbose=False)
    r = getattr(Class, METHOD)(**PARAMS).decompose(S)
    assert r.Res is not None
    err = float(np.max(np.abs(np.asarray(r.IMFs).sum(axis=0) + np.asarray(r.Res) - S)))
    assert err < RECON_TOL, f"reconstruction error {err:.3e}"


def test_facade_matches_class():
    S = _sig()
    MD.set_seed(42, verbose=False)
    a = getattr(Class, METHOD)(**PARAMS).decompose(S)
    MD.set_seed(42, verbose=False)
    b = getattr(Function, METHOD)(S, **PARAMS)
    assert np.array_equal(np.asarray(a.IMFs), np.asarray(b.IMFs))
    if a.Res is not None:
        assert np.array_equal(np.asarray(a.Res), np.asarray(b.Res))


@pytest.mark.parametrize("label,S", [
    ("constant", np.full(128, 3.0)),
    ("zeros", np.zeros(128)),
    ("ramp", np.linspace(0.0, 1.0, 128)),
])
def test_degenerate_inputs(label, S):
    MD.set_seed(42, verbose=False)
    r = getattr(Class, METHOD)(**PARAMS).decompose(S)
    assert isinstance(r, DecompositionResult)
    assert np.asarray(r.IMFs).ndim == 2
    assert np.asarray(r.IMFs).shape[1] == S.size


def test_input_not_mutated():
    S = _sig()
    S0 = S.copy()
    MD.set_seed(42, verbose=False)
    getattr(Class, METHOD)(**PARAMS).decompose(S)
    assert np.array_equal(S, S0)


def test_deterministic_repeat():
    a = np.asarray(_run().IMFs)
    b = np.asarray(_run().IMFs)
    assert np.array_equal(a, b)


def test_invalid_parameter_rejected():
    with pytest.raises((TypeError, ValueError)):
        getattr(Class, METHOD)(__definitely_not_a_param__=1)
