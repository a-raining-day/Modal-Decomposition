"""
``Modal_Decomposition.Base.CPU`` 模块测试 (一模块一测试)。

实测契约 (由脚本逐一确认, 非猜测):

* ``Base.CPU`` **是模块**, 不是类: 模块里没有 ``CPU`` 这个类,
  ``Modal_Decomposition.Base.CPU`` / ``Modal_Decomposition.CPU`` 解析到的都是本模块。
* ``__all__ == ["cpu_num", "allocable_cpu_num", "total_cpu", "set_allocable"]``。
* ``cpu_num`` / ``allocable_cpu_num`` 是模块级 ``int`` 常量 (导入时求值一次)。
* ``total_cpu`` 是**函数** (``() -> int``), 不是常量; 结果缓存在 ``cpu_num`` 里。
* ``set_allocable(value)``: ``int`` 为绝对核数, ``float`` 为比例, ``-1`` 是
  "全部" 哨兵值 (**不抛异常**, 返回 ``cpu_num``)。
* 非法输入: ``0`` / 越界 ``int`` / 越界比例 -> ``ValueError``;
  ``str`` / ``None`` / ``bool`` -> ``TypeError``。

所有会改写模块级全局 ``allocable_cpu_num`` 的用例都通过 ``allocable_guard``
(``try/finally``) 恢复测试前的值, 避免污染同进程里的其它用例。
"""

import importlib

import pytest

import Modal_Decomposition
import Modal_Decomposition.Base as Base
import Modal_Decomposition.Base.CPU as CPU_MOD
from Modal_Decomposition.Base.CPU import (
    allocable_cpu_num,
    cpu_num,
    set_allocable,
    total_cpu,
)

# 模块的公开 API (实测 __all__)。
EXPECTED_ALL = ["cpu_num", "allocable_cpu_num", "total_cpu", "set_allocable"]


@pytest.fixture
def allocable_guard():
    """快照 ``allocable_cpu_num``, 用例结束后无条件恢复 (try/finally)。"""
    saved = CPU_MOD.allocable_cpu_num
    try:
        yield saved
    finally:
        CPU_MOD.set_allocable(saved)


# --------------------------------------------------------------------------- #
# 1. 可导入 / __all__ 与实际导出一致
# --------------------------------------------------------------------------- #
def test_module_importable_and_all_matches_exports():
    assert CPU_MOD.__name__ == "Modal_Decomposition.Base.CPU"
    assert CPU_MOD.__all__ == EXPECTED_ALL, f"__all__ 变了: {CPU_MOD.__all__!r}"
    # __all__ 里每一项都必须真的存在且不是私有名。
    for name in CPU_MOD.__all__:
        assert hasattr(CPU_MOD, name), f"__all__ 声明了不存在的 {name!r}"
        assert not name.startswith("_")
    # 反过来: 模块里自己定义的函数 / 模块级 int 常量都必须在 __all__ 里。
    defined_here = {
        name
        for name, obj in vars(CPU_MOD).items()
        if not name.startswith("_") and getattr(obj, "__module__", None) == CPU_MOD.__name__
    }
    assert defined_here <= set(EXPECTED_ALL), f"未被 __all__ 覆盖的函数: {defined_here - set(EXPECTED_ALL)}"
    own_ints = {
        name
        for name, obj in vars(CPU_MOD).items()
        if not name.startswith("_") and isinstance(obj, int) and not isinstance(obj, bool)
    }
    assert own_ints <= set(EXPECTED_ALL), f"未被 __all__ 覆盖的 int 常量: {own_ints - set(EXPECTED_ALL)}"


def test_imported_helpers_are_not_advertised_as_api():
    # os / Define 里的常量是内部依赖, 不属于本模块的公开 API。
    assert "os" not in CPU_MOD.__all__
    assert "PYTHON_VERSION" not in CPU_MOD.__all__
    assert "DEFAULT_ALLOCABLE_CPU_RATIO" not in CPU_MOD.__all__


def test_star_import_yields_exactly_all():
    ns = {}
    exec("from Modal_Decomposition.Base.CPU import *", ns)  # noqa: S102 - 验证 __all__
    exported = {k for k in ns if not k.startswith("__")}
    assert exported == set(CPU_MOD.__all__)


# --------------------------------------------------------------------------- #
# 2. 常量与 total_cpu 的取值契约
# --------------------------------------------------------------------------- #
def test_cpu_num_is_positive_int_constant():
    assert isinstance(cpu_num, int) and not isinstance(cpu_num, bool)
    assert cpu_num >= 1


def test_total_cpu_is_callable_not_constant():
    # 背景资料说 total_cpu 是 int 常量, 实测是函数 () -> int; 按真实行为断言。
    assert callable(total_cpu), "total_cpu 应当是函数"
    assert not isinstance(total_cpu, int)
    value = total_cpu()
    assert isinstance(value, int) and not isinstance(value, bool)
    assert value >= 1
    # 结果缓存在模块级 cpu_num 上, 两次调用一致。
    assert value == cpu_num
    assert total_cpu() == value


def test_allocable_cpu_num_within_bounds():
    assert isinstance(allocable_cpu_num, int) and not isinstance(allocable_cpu_num, bool)
    assert 1 <= allocable_cpu_num <= cpu_num


def test_default_ratio_is_a_valid_ratio():
    from Modal_Decomposition.Define import DEFAULT_ALLOCABLE_CPU_RATIO as RATIO

    assert isinstance(RATIO, float)
    assert 0.0 < RATIO <= 1.0
    # 该比例应用后必然落在 [1, cpu_num] 内。
    assert 1 <= max(1, int(cpu_num * RATIO)) <= cpu_num


# --------------------------------------------------------------------------- #
# 3. set_allocable: 正常输入
# --------------------------------------------------------------------------- #
def test_set_allocable_one_returns_one(allocable_guard):
    assert set_allocable(1) == 1
    assert CPU_MOD.allocable_cpu_num == 1


def test_set_allocable_half_ratio_returns_int_in_range(allocable_guard):
    result = set_allocable(0.5)
    assert isinstance(result, int) and not isinstance(result, bool)
    assert 1 <= result <= cpu_num
    assert result == max(1, int(cpu_num * 0.5)), "float 走 floor(cpu_num * ratio), 且下限 1"
    assert CPU_MOD.allocable_cpu_num == result


def test_set_allocable_float_below_one_cpu_clamps_to_one(allocable_guard):
    assert set_allocable(0.01) == 1
    assert set_allocable(1.0) == cpu_num


def test_set_allocable_minus_one_means_all_not_an_error(allocable_guard):
    # 实测: -1 是 docstring 里写明的 "all" 哨兵值, 返回 cpu_num, 不抛异常。
    assert set_allocable(-1) == cpu_num
    assert CPU_MOD.allocable_cpu_num == cpu_num


def test_set_allocable_rebinds_module_global():
    stale = allocable_cpu_num              # 导入时 from-import 得到的副本 (int 快照)
    saved = CPU_MOD.allocable_cpu_num      # 当前模块属性
    try:
        CPU_MOD.set_allocable(1)
        # 模块属性已重绑定; 但 from-import 出去的副本不会跟着变。
        assert CPU_MOD.allocable_cpu_num == 1
        assert allocable_cpu_num == stale, "from-import 的 int 是快照, 不应被重绑定"
    finally:
        CPU_MOD.set_allocable(saved)


# --------------------------------------------------------------------------- #
# 4. set_allocable: 非法输入 (实测的异常类型)
# --------------------------------------------------------------------------- #
def test_set_allocable_zero_raises_value_error(allocable_guard):
    with pytest.raises(ValueError):
        set_allocable(0)


def test_set_allocable_string_raises_type_error(allocable_guard):
    with pytest.raises(TypeError):
        set_allocable("x")


@pytest.mark.parametrize(
    "value, exc",
    [
        (-2, ValueError),            # 负数里只有 -1 是哨兵
        (cpu_num + 1, ValueError),   # 超过物理核数
        (1.5, ValueError),           # 比例越界
        (0.0, ValueError),           # 0 比例
        (-0.5, ValueError),          # 负比例
        (None, TypeError),
        (True, TypeError),           # bool 是 int 子类, 被显式拒绝
        ([1], TypeError),
    ],
    ids=["minus_2", "above_total", "ratio_gt_1", "ratio_zero", "ratio_negative",
         "none", "bool", "list"],
)
def test_set_allocable_rejects_invalid_values(allocable_guard, value, exc):
    with pytest.raises(exc):
        set_allocable(value)
    # 抛错时不应留下半改的全局状态。
    assert CPU_MOD.allocable_cpu_num == allocable_guard


# --------------------------------------------------------------------------- #
# 5. 恢复性: try/finally 把 allocable_cpu_num 还原
# --------------------------------------------------------------------------- #
def test_allocable_restored_by_explicit_try_finally():
    saved = CPU_MOD.allocable_cpu_num
    try:
        assert set_allocable(1) == 1
        assert CPU_MOD.allocable_cpu_num == 1
    finally:
        CPU_MOD.set_allocable(saved)
    assert CPU_MOD.allocable_cpu_num == saved
    assert 1 <= CPU_MOD.allocable_cpu_num <= CPU_MOD.cpu_num


def test_restore_after_exception_inside_try_finally():
    saved = CPU_MOD.allocable_cpu_num
    try:
        set_allocable(2)
        with pytest.raises(RuntimeError):
            raise RuntimeError("boom")
    finally:
        CPU_MOD.set_allocable(saved)
    assert CPU_MOD.allocable_cpu_num == saved
    assert 1 <= CPU_MOD.allocable_cpu_num <= CPU_MOD.cpu_num


# --------------------------------------------------------------------------- #
# 6. 没有 CPU 类: 用 "类/模块属性存在" 做烟测 (dir() 实测无类)
# --------------------------------------------------------------------------- #
def test_module_exposes_no_class_the_name_is_the_module():
    classes = [
        name
        for name, obj in vars(CPU_MOD).items()
        if isinstance(obj, type) and getattr(obj, "__module__", None) == CPU_MOD.__name__
    ]
    assert classes == [], f"实测模块里没有类, 出现了 {classes}"

    # 所以 Base.CPU / Modal_Decomposition.CPU 都是这个模块对象本身。
    resolved = importlib.import_module("Modal_Decomposition.Base.CPU")
    assert Base.CPU is resolved
    assert Modal_Decomposition.CPU is resolved
    assert isinstance(Base.CPU, type(CPU_MOD))
    # 烟测: 通过包属性访问与模块直连访问拿到的是同一份全局状态。
    assert Base.CPU.cpu_num == CPU_MOD.cpu_num
    assert Base.allocable_cpu_num == CPU_MOD.allocable_cpu_num
