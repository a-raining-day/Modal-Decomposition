"""
``output_memmap`` 契约测试。

覆盖 15 个方法的 ``decompose`` 与 ``f_*`` 门面：
* 默认返回 ``np.ndarray``；
* ``output_memmap=True`` 时 ``IMFs`` / ``Res`` / ``info`` 里的数组都是 ``np.memmap``；
* 非法类型（``1`` / ``0`` / ``None`` / 字符串 / 浮点）抛 ``TypeError``；
* 两条路径的数值逐位一致。
"""

import numpy as np
import pytest

import Modal_Decomposition as MD
from tests._cases import CASES, METHODS

S = (np.sin(2 * np.pi * 5 * np.linspace(0, 1, 512, endpoint=False))
     + 0.4 * np.sin(2 * np.pi * 31 * np.linspace(0, 1, 512, endpoint=False)))


def _arrays_in(obj):
    """递归收集 info 里的 ndarray / memmap。"""
    out = []
    if isinstance(obj, np.ndarray):
        out.append(obj)
    elif isinstance(obj, dict):
        for v in obj.values():
            out.extend(_arrays_in(v))
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            out.extend(_arrays_in(v))
    return out


@pytest.mark.parametrize("name", METHODS)
def test_decompose_default_is_ndarray(name):
    """默认不返回 memmap。"""
    r = getattr(MD.Class, name)(**CASES[name]).decompose(S)
    assert isinstance(r.IMFs, np.ndarray)
    assert not isinstance(r.IMFs, np.memmap)


@pytest.mark.parametrize("name", METHODS)
def test_decompose_output_memmap(name):
    """`output_memmap=True` 时 IMFs / Res / info 数组全部是 memmap。"""
    r = getattr(MD.Class, name)(**CASES[name]).decompose(S, output_memmap=True)
    assert isinstance(r.IMFs, np.memmap)
    if r.Res is not None:
        assert isinstance(r.Res, np.memmap)
    for arr in _arrays_in(r.info):
        if arr.ndim == 0:
            continue
        assert isinstance(arr, np.memmap), f"{name}: info 里的 0 维以外数组应为 memmap"


@pytest.mark.parametrize("name", METHODS)
def test_memmap_values_match_ndarray(name):
    """两条路径数值一致。

    EEMD 族内部要加噪，所以两次调用之间必须复位种子，否则 RNG 状态推进会得到
    不同的噪声实现（那是预期行为，与本测试无关）。
    """
    kw = CASES[name]
    MD.set_seed(42, verbose=False)
    a = getattr(MD.Class, name)(**kw).decompose(S)
    MD.set_seed(42, verbose=False)
    b = getattr(MD.Class, name)(**kw).decompose(S, output_memmap=True)
    assert np.array_equal(np.asarray(a.IMFs), np.asarray(b.IMFs))
    if a.Res is None:
        assert b.Res is None
    else:
        assert np.array_equal(np.asarray(a.Res), np.asarray(b.Res))


@pytest.mark.parametrize("name", METHODS)
def test_facade_output_memmap(name):
    """门面函数同样支持 `output_memmap`。"""
    kw = CASES[name]
    r = getattr(MD.Function, name)(S, output_memmap=True, **kw)
    assert isinstance(r.IMFs, np.memmap)
    r2 = getattr(MD.Function, name)(S, **kw)
    assert not isinstance(r2.IMFs, np.memmap)


@pytest.mark.parametrize("name", METHODS)
@pytest.mark.parametrize("bad", [1, 0, None, "yes", 1.0])
def test_output_memmap_type_check(name, bad):
    """只接受真正的 bool。"""
    with pytest.raises(TypeError, match="output_memmap must be bool"):
        getattr(MD.Class, name)(**CASES[name]).decompose(S, output_memmap=bad)


def test_vmd_info_diagnostics_are_memmap():
    """VMD 的诊断量（omega/history/u_hat）也随 output_memmap 落盘。"""
    r = MD.Class.VMD(num_imf=2, n=100).decompose(S, output_memmap=True)
    for key in ("omega", "omega_hz", "omega_history", "u_hat"):
        val = r.info.get(key)
        if isinstance(val, np.ndarray) and val.ndim > 0:
            assert isinstance(val, np.memmap), f"VMD.info[{key}] 应为 memmap"


def test_signature_has_output_memmap_last():
    """`decompose` 的最后一个形参是 `output_memmap`，默认 False。"""
    import inspect
    for name in METHODS:
        params = list(inspect.signature(getattr(MD.Class, name).decompose).parameters.values())
        assert params[-1].name == "output_memmap", name
        assert params[-1].default is False, name
