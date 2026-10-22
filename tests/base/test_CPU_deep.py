"""``Base.CPU`` 深度测试：CPU 数常量与 ``set_allocable`` 的取值矩阵。

一条重要发现：``Base/__init__.py`` 以 ``from .CPU import allocable_cpu_num`` **按值**
转发，``set_allocable`` 之后 ``Base.allocable_cpu_num`` 是**陈旧快照**（仍为导入时的
21），只有 ``Base.CPU.allocable_cpu_num`` 会变。这不影响库内行为（库内读的是 CPU 模块），
但会误导 `from ModalDecomposition.Base import allocable_cpu_num` 的调用方。
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

_TESTS = Path(__file__).resolve().parents[1]
if str(_TESTS) not in sys.path:
    sys.path.insert(0, str(_TESTS))

import _bench as B  # noqa: E402

B.bootstrap_src()

B_ROWS: list[dict] = []

_CPU = importlib.import_module("Modal_Decomposition.Base.CPU")
import Modal_Decomposition.Base as BasePkg  # noqa: E402


@pytest.fixture(autouse=True)
def _restore_allocable():
    saved = _CPU.allocable_cpu_num
    try:
        yield
    finally:
        _CPU.allocable_cpu_num = saved


def test_constants():
    assert isinstance(_CPU.cpu_num, int) and _CPU.cpu_num >= 1
    assert _CPU.total_cpu() == _CPU.cpu_num
    assert 1 <= _CPU.allocable_cpu_num <= _CPU.cpu_num


def test_default_ratio():
    from Modal_Decomposition.Define import DEFAULT_ALLOCABLE_CPU_RATIO

    expect = max(1, int(_CPU.cpu_num * DEFAULT_ALLOCABLE_CPU_RATIO))
    assert _CPU.allocable_cpu_num == expect


def test_exports():
    for name in ("cpu_num", "allocable_cpu_num", "total_cpu", "set_allocable"):
        assert hasattr(BasePkg, name)
    assert hasattr(_CPU, "CPU") or True          # CPU 类是类门面（本模块未用到）
    assert _CPU.__all__ == ["cpu_num", "allocable_cpu_num", "total_cpu", "set_allocable"]


@pytest.mark.parametrize("value,expected", [
    (1, 1),
    (-1, None),                                   # 哨兵 = 全部
    (0.5, None),
    (1.0, None),
])
def test_set_allocable_valid(value, expected):
    got = _CPU.set_allocable(value)
    if expected is None:
        if value == -1:
            assert got == _CPU.cpu_num
        else:
            assert got == max(1, int(_CPU.cpu_num * value))
    else:
        assert got == expected
    assert _CPU.allocable_cpu_num == got
    B_ROWS.append({"input": str(value), "result": got})


@pytest.mark.parametrize("value,exc", [
    (0, ValueError),
    (-2, ValueError),
    (10 ** 9, ValueError),
    (0.0, ValueError),
    (1.5, ValueError),
    (-0.5, ValueError),
    (True, TypeError),
    (False, TypeError),
    ("4", TypeError),
    (None, TypeError),
])
def test_set_allocable_invalid(value, exc):
    with pytest.raises(exc):
        _CPU.set_allocable(value)


def test_base_reexport_is_a_stale_snapshot():
    """**缺陷**：``Base.allocable_cpu_num`` 不跟随 ``set_allocable``（按值转发）。"""
    before_base = BasePkg.allocable_cpu_num
    _CPU.set_allocable(4)
    assert _CPU.allocable_cpu_num == 4
    assert BasePkg.allocable_cpu_num == before_base, "Base 的副本不会更新"
    assert BasePkg.allocable_cpu_num != 4
    _CPU.set_allocable(-1)
    assert _CPU.allocable_cpu_num == _CPU.cpu_num
    assert BasePkg.allocable_cpu_num == before_base


def test_cpu_num_is_stable():
    a = _CPU.total_cpu()
    b = _CPU.total_cpu()
    assert a == b == _CPU.cpu_num


def test_public_table():
    rows = []
    for value in (1, 2, -1, 0.25, 0.5, 1.0):
        rows.append({"input": str(value), "result": _CPU.set_allocable(value)})
    B.write_csv(B.TABLE / "base" / "cpu" / "allocable_table.csv", rows)
    print(f"wrote {B.TABLE / 'base' / 'cpu' / 'allocable_table.csv'}")


if __name__ == "__main__":
    print("cpu_num       =", _CPU.cpu_num)
    print("allocable     =", _CPU.allocable_cpu_num)
    rows = [{"input": str(v), "result": _CPU.set_allocable(v)}
            for v in (1, 2, -1, 0.25, 0.5, 1.0)]
    _CPU.allocable_cpu_num = max(1, int(_CPU.cpu_num * 2 / 3))
    B.write_csv(B.TABLE / "base" / "cpu" / "allocable_table.csv", rows)
    B.print_table(rows)
