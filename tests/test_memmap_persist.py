"""
``temp_memmap(persist=...)`` 与 ``finalize_output(..., persist=...)`` 契约测试。

``persist=False``（默认）：后备文件登记进 ``Base.Memory._MM_FILES``，
:func:`Base.Memory.cleanup_memmaps` 或解释器退出时删除。

``persist=True``：不登记，所有权交给调用方，清理函数不会碰它。
"""

import os

import numpy as np
import pytest

from Modal_Decomposition.Base import cleanup_memmaps, temp_memmap
from Modal_Decomposition.Base.Memory import _MM_FILES
from Modal_Decomposition.Define import Config, DecompositionResult, finalize_output


def test_persist_false_is_registered_and_cleaned():
    """默认登记进清理表，并在 cleanup 后消失。"""
    mm = temp_memmap((16,))
    if mm is None:
        pytest.skip("平台不支持 memmap")
    path = mm.filename
    assert path in _MM_FILES
    cleanup_memmaps(force=True)
    assert not os.path.exists(path)


def test_persist_true_is_not_registered():
    """`persist=True` 不登记，且 cleanup 不会删它。"""
    mm = temp_memmap((16,), persist=True)
    if mm is None:
        pytest.skip("平台不支持 memmap")
    path = mm.filename
    try:
        assert path not in _MM_FILES
        assert os.path.exists(path)
        cleanup_memmaps(force=True)
        assert os.path.exists(path)
    finally:
        del mm
        try:
            os.remove(path)
        except OSError:
            pass


@pytest.mark.parametrize("bad", [1, 0, None, "yes", 1.0, b"true"])
def test_persist_type_check(bad):
    """`persist` 只接受真正的 bool。"""
    with pytest.raises(TypeError, match="persist"):
        temp_memmap((4,), persist=bad)


def test_finalize_output_persist_true_keeps_files():
    """`finalize_output(persist=True)` 的 IMFs / Res / info 都不登记。"""
    r = DecompositionResult(np.ones((2, 8)), np.zeros(8), {"d": np.arange(4.0)}, Config())
    m = finalize_output(r, True, persist=True)
    paths = [m.IMFs.filename, m.Res.filename, m.info["d"].filename]
    try:
        assert all(p not in _MM_FILES for p in paths)
        cleanup_memmaps(force=True)
        assert all(os.path.exists(p) for p in paths)
        assert np.array_equal(np.asarray(m.info["d"]), np.arange(4.0))
    finally:
        for p in paths:
            try:
                os.remove(p)
            except OSError:
                pass


def test_finalize_output_persist_type_check():
    """`finalize_output` 的 `persist` 同样做类型检测。"""
    r = DecompositionResult(np.ones((2, 4)), None, {}, Config())
    with pytest.raises(TypeError, match="persist"):
        finalize_output(r, True, persist=1)
