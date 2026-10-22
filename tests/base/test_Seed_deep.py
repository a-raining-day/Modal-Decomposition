"""``Base.Seed`` 深度测试：单例、校验、锁、临时覆盖。

锁是**进程级且不可解除**的（没有 unlock API），所以"锁定后 set 抛 ValueError"这一条
放在子进程里验证，避免污染整个 pytest 会话。
"""

from __future__ import annotations

import importlib
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

_TESTS = Path(__file__).resolve().parents[1]
if str(_TESTS) not in sys.path:
    sys.path.insert(0, str(_TESTS))

_Seed = importlib.import_module("Modal_Decomposition.Base.Seed")
Seed = _Seed.Seed
_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def _restore_seed():
    saved = Seed.GLOBAL_SEED
    saved_locked = Seed._locked
    try:
        yield
    finally:
        Seed.GLOBAL_SEED = saved
        Seed._locked = saved_locked


def test_singleton():
    assert Seed() is Seed()
    assert Seed().GLOBAL_SEED == Seed.GLOBAL_SEED


def test_default_and_getters():
    assert Seed.get() == Seed.GLOBAL_SEED
    assert Seed().seed == Seed.get()
    assert isinstance(Seed.get(), int)


def test_set_returns_value_and_warns(recwarn):
    assert Seed.set(7) == 7
    assert Seed.get() == 7
    warned = [w for w in recwarn if "Overriding" in str(w.message)]
    assert warned, "覆盖已有种子应告警"


def test_set_verbose_false_silent(recwarn):
    Seed.set(8, verbose=False)
    assert Seed.get() == 8
    assert not [w for w in recwarn if "Overriding" in str(w.message)]


@pytest.mark.parametrize("bad,exc", [
    (-1, ValueError),
    ("42", TypeError),
    (3.5, TypeError),
    (None, TypeError),
])
def test_set_invalid(bad, exc):
    with pytest.raises(exc):
        Seed.set(bad)


def test_validate():
    assert Seed.validate(0) == 0
    assert Seed.validate(2 ** 40) == 2 ** 40
    with pytest.raises(ValueError):
        Seed.validate(-1)


def test_numpy_integer_characterisation():
    """特征化：``validate`` 先 ``isinstance(seed, int)``，所以 ``np.int64`` 被拒。

    （``bool`` 是 ``int`` 子类，会被接受并转成 0/1。）
    """
    with pytest.raises(TypeError):
        Seed.validate(np.int64(5))
    assert Seed.validate(True) == 1


def test_temporary_seed_restores():
    Seed.GLOBAL_SEED = 42
    with Seed.temporary_seed(11):
        assert Seed.get() == 11
    assert Seed.get() == 42


def test_temporary_seed_restores_on_exception():
    Seed.GLOBAL_SEED = 42
    with pytest.raises(RuntimeError):
        with Seed.temporary_seed(12):
            assert Seed.get() == 12
            raise RuntimeError("boom")
    assert Seed.get() == 42


def test_temporary_seed_validates():
    with pytest.raises(ValueError):
        with Seed.temporary_seed(-5):
            pass


def test_global_seed_is_import_snapshot():
    """``Global_Seed`` 是导入时快照，``Seed.set`` 之后不会跟着变。"""
    Seed.set(99, verbose=False)
    assert Seed.get() == 99
    assert _Seed.Global_Seed != 99, "Global_Seed 是快照，不是活引用"


def test_documented_pitfall_instance_attribute_shadow():
    """docstring 里的坑：写实例属性不会改变类属性，``Seed.get()`` 仍读类属性。"""
    Seed.GLOBAL_SEED = 42
    inst = Seed()
    inst.GLOBAL_SEED = 7
    try:
        assert Seed.get() == 42
        assert inst.GLOBAL_SEED == 7
    finally:
        del inst.GLOBAL_SEED


def test_lock_is_process_global():
    """锁定 → 后续 ``set`` 抛 ValueError（在子进程里测，避免污染本会话）。"""
    code = textwrap.dedent(
        """
        import sys
        sys.path.insert(0, r"SRC")
        from Modal_Decomposition.Base.Seed import Seed
        print("status_before", Seed().status)
        Seed.set(5, lock=True, verbose=False)
        print("status_after", Seed().status)
        print("seed", Seed.get())
        try:
            Seed.set(6, verbose=False)
        except ValueError as e:
            print("locked_error", str(e)[:40])
        """
    ).replace("SRC", str(_ROOT / "src"))
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, encoding="utf-8")
    assert out.returncode == 0, out.stderr
    assert "status_before False" in out.stdout
    assert "status_after True" in out.stdout
    assert "seed 5" in out.stdout
    assert "locked_error GLOBAL_SEED is locked" in out.stdout


def test_thread_safety_smoke():
    import threading

    Seed.GLOBAL_SEED = 42
    errs = []

    def worker(i):
        try:
            for k in range(200):
                Seed.set(i * 1000 + k, verbose=False)
                assert isinstance(Seed.get(), int)
        except Exception as exc:                    # noqa: BLE001
            errs.append(exc)

    ts = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert not errs, errs


def test_public_surface():
    import Modal_Decomposition.Base as BasePkg

    assert BasePkg.Seed is Seed
    assert BasePkg.Global_Seed == _Seed.Global_Seed
    assert set(_Seed.__all__) == {"Seed", "Global_Seed"}
