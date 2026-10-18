"""
``Modal_Decomposition.Base.Seed`` 的模块级契约测试。

覆盖: ``__all__`` 与真实导出 / 默认种子 / ``set``-``get`` 往返与返回值 /
状态恢复（``try``/``finally``）/ 包级 ``set_seed`` 与 ``numpy`` 全局 RNG 联动 /
``lock`` 语义 / ``Global_Seed`` 的导入期快照语义 / 入参校验。

注意: 包命名空间里 ``Modal_Decomposition.Base.Seed`` 这个名字已被**类**接管
（子模块 import 后被 ``from .Seed import Seed`` 覆盖），因此
``import Modal_Decomposition.Base.Seed as m`` 拿到的是类而不是模块，
必须用 ``importlib.import_module`` 取模块对象。
"""

import importlib
import warnings

import numpy as np
import pytest

import Modal_Decomposition as MD
from Modal_Decomposition.Base import Global_Seed, Seed

# 取真正承载 ``__all__`` 的子模块对象（不是被同名类遮蔽的包属性）。
seed_mod = importlib.import_module("Modal_Decomposition.Base.Seed")

DEFAULT_SEED = 42
OVERRIDE_MSG = "Overriding existing GLOBAL_SEED"

# ``tests/conftest.py`` 的 autouse fixture 在每个用例结束后用
# ``set_seed(42)``（默认 ``verbose=True``）复位种子，会额外发出一条
# "Overriding existing GLOBAL_SEED."；只在本模块静音，避免重复噪声。
# 模块内需要观察告警的用例自行用 ``warnings.catch_warnings`` 打开。
pytestmark = pytest.mark.filterwarnings(f"ignore:{OVERRIDE_MSG}")


@pytest.fixture()
def restore_seed():
    """备份并在用例结束后恢复全局种子与 lock 状态，避免污染后续用例。"""
    old_seed = Seed.get()
    old_locked = Seed._locked          # 无公开解锁 API，只能直接恢复私有标记
    try:
        yield
    finally:
        Seed._locked = old_locked
        Seed.set(old_seed, verbose=False)


# --------------------------------------------------------------------------- #
# 1. 导出契约
# --------------------------------------------------------------------------- #
def test_dunder_all_matches_actual_exports():
    assert seed_mod.__all__ == ["Seed", "Global_Seed"]
    assert len(set(seed_mod.__all__)) == len(seed_mod.__all__)   # 无重复
    for name in seed_mod.__all__:
        assert hasattr(seed_mod, name), f"__all__ 里的 {name!r} 不存在"

    # 导出的对象身份: 类就是包里那个类，快照就是模块级那个常量
    assert seed_mod.Seed is Seed
    assert isinstance(Seed, type)
    assert seed_mod.Global_Seed is Global_Seed
    assert seed_mod.Global_Seed == DEFAULT_SEED


# --------------------------------------------------------------------------- #
# 2. 默认值
# --------------------------------------------------------------------------- #
def test_default_seed_is_42():
    # conftest 的 autouse fixture 每个用例后都复位到 42，故此处即模块默认态
    assert Seed.GLOBAL_SEED == DEFAULT_SEED
    assert Seed.get() == DEFAULT_SEED
    assert type(Seed.get()) is int
    assert MD.get_seed() == DEFAULT_SEED


# --------------------------------------------------------------------------- #
# 3. set / get 往返（含返回值）
# --------------------------------------------------------------------------- #
def test_set_then_get_roundtrip(restore_seed):
    assert Seed.set(123, verbose=False) == 123
    assert Seed.get() == 123
    assert Seed.GLOBAL_SEED == 123
    assert MD.get_seed() == 123


# --------------------------------------------------------------------------- #
# 4. 恢复性: try/finally 保证不污染后续用例
# --------------------------------------------------------------------------- #
def test_try_finally_restores_previous_seed():
    old = Seed.get()
    try:
        Seed.set(2024, verbose=False)
        assert Seed.get() == 2024
    finally:
        Seed.set(old, verbose=False)

    assert Seed.get() == old
    assert Seed.GLOBAL_SEED == old


# --------------------------------------------------------------------------- #
# 5. 包级 set_seed 与 numpy 全局 RNG 联动
# --------------------------------------------------------------------------- #
def test_package_set_seed_also_reseeds_numpy(restore_seed):
    assert MD.set_seed(7, verbose=False) == 7
    assert MD.get_seed() == 7
    assert Seed.get() == 7                 # 包级入口确实写入了 Seed
    assert Seed.GLOBAL_SEED == 7

    first = np.random.random()             # numpy 全局 RNG 已被 seed
    MD.set_seed(7, verbose=False)
    second = np.random.random()
    assert first == second                 # 同一个种子 → 同一个数


def test_package_set_seed_is_int_seeded(restore_seed):
    # numpy 侧的可复现性: 连续两次同种子重置后，整段随机序列逐位相同
    MD.set_seed(11, verbose=False)
    a = np.random.random(8)
    MD.set_seed(11, verbose=False)
    b = np.random.random(8)
    assert np.array_equal(a, b)


# --------------------------------------------------------------------------- #
# 6. lock 语义: lock=True 后的 set 抛 ValueError（实测行为）
# --------------------------------------------------------------------------- #
def test_lock_blocks_further_set(restore_seed):
    assert Seed.set(999, lock=True, verbose=False) == 999
    assert Seed.get() == 999
    assert Seed().status is True

    with pytest.raises(ValueError, match="locked"):
        Seed.set(555, verbose=False)

    # 失败的写入不改变已有值；读取在锁住后仍然可用
    assert Seed.get() == 999
    assert Seed.GLOBAL_SEED == 999
    assert Seed().status is True


def test_lock_default_is_unlocked(restore_seed):
    assert Seed().status is False
    # 默认 lock=False: 连续两次写入都成功
    assert Seed.set(5, verbose=False) == 5
    assert Seed.set(6, verbose=False) == 6
    assert Seed().status is False


# --------------------------------------------------------------------------- #
# 7. Global_Seed 是导入期常量快照
# --------------------------------------------------------------------------- #
def test_global_seed_is_import_time_snapshot(restore_seed):
    snapshot = Global_Seed
    assert snapshot == DEFAULT_SEED

    Seed.set(123, verbose=False)
    assert Seed.GLOBAL_SEED == 123         # 类属性跟着变
    assert Global_Seed == snapshot         # 模块级快照不联动
    assert seed_mod.Global_Seed is Global_Seed
    assert Global_Seed is not Seed.GLOBAL_SEED


# --------------------------------------------------------------------------- #
# 其它公开契约: 单例 / 属性 / 入参校验 / verbose
# --------------------------------------------------------------------------- #
def test_seed_is_singleton(restore_seed):
    assert Seed() is Seed()
    assert Seed().seed == Seed.get()


@pytest.mark.parametrize("value", [-1, -42])
def test_negative_seed_rejected(value):
    with pytest.raises(ValueError):
        Seed.validate(value)
    with pytest.raises(ValueError):
        Seed.set(value, verbose=False)


@pytest.mark.parametrize("value", [1.0, "x", None, np.int64(5)])
def test_non_int_seed_rejected(value):
    with pytest.raises(TypeError):
        Seed.validate(value)
    with pytest.raises(TypeError):
        Seed.set(value, verbose=False)


def test_verbose_controls_override_warning(restore_seed):
    Seed.set(1, verbose=False)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        Seed.set(2)                        # 覆盖默认 verbose=True → 告警
    assert [w for w in caught if OVERRIDE_MSG in str(w.message)]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        Seed.set(3, verbose=False)         # 静默覆盖 → 无告警
    assert not [w for w in caught if OVERRIDE_MSG in str(w.message)]
    assert Seed.get() == 3
