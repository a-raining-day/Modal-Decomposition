"""
``Base.Cache`` 单模块测试：进程级导入缓存（``Cache`` / ``cache``）。

本文件与 ``docs/test/Cache_Engine_Report.md`` 的契约清单一一对应：

1. 可导入且 ``__all__`` 与实际导出一致；
2. 公开可调用对象（``Cache.import_module``）签名合理，能接受一个模块名；
3. 缓存语义：返回真实模块对象，两次调用 ``is`` 同一对象（不重复 import）；
4. 缺失模块返回 ``None`` 且**不抛异常**（当前实现的真实行为，先用脚本核实）；
5. 唯一名注册：``cache`` 与 ``Cache`` 共用同一张注册表（``Cache`` 是别名）；
6. 副作用可控：新导入的模块确实进入缓存表，且用例结束后恢复进程级全局状态。

约定：``Modal_Decomposition.Base.Cache`` 这个**模块**在包命名空间里被同名类
``Cache`` 遮蔽，因此这里用 ``importlib.import_module`` 取真正的模块对象来做
``__all__`` 断言。
"""

from __future__ import annotations

import importlib
import inspect
import json
import math
import statistics

import pytest

from Modal_Decomposition.Base import Cache, cache

# 真正的模块对象（Base.Cache 这个名字在包命名空间里指向类，不是模块）
_MODULE = importlib.import_module("Modal_Decomposition.Base.Cache")


@pytest.fixture(autouse=True)
def _isolated_cache_registry():
    """
    注册表隔离：快照 -> 清空 -> 运行用例 -> 精确恢复。

    缓存是**进程级共享**的（类属性），所以每个用例都必须从干净状态出发，
    结束后再把别人的条目原样放回去，避免污染同一次 pytest 会话里的其他文件。
    """
    saved = (
        dict(Cache._entries),
        dict(Cache._descriptions),
        dict(Cache._exists),
    )
    with Cache._lock:
        Cache._entries.clear()
        Cache._descriptions.clear()
        Cache._exists.clear()
    try:
        yield
    finally:
        with Cache._lock:
            Cache._entries.clear()
            Cache._entries.update(saved[0])
            Cache._descriptions.clear()
            Cache._descriptions.update(saved[1])
            Cache._exists.clear()
            Cache._exists.update(saved[2])


# --------------------------------------------------------------------------- #
# 契约 1：可导入且 __all__ 与实际导出一致
# --------------------------------------------------------------------------- #
def test_module_exports_cache_and_cache_alias():
    """契约 1：``Cache`` 与 ``cache`` 都可从 ``Modal_Decomposition.Base`` 导入，且与模块 ``__all__`` 一致。"""
    assert _MODULE.__all__ == ["Cache", "cache"]
    # __all__ 里写的东西必须真的存在
    for name in _MODULE.__all__:
        assert hasattr(_MODULE, name), f"__all__ 声明了 {name!r} 但没有导出"

    assert Cache is not None and cache is not None
    # 包级导出必须就是模块里那两个对象（不是另建的副本）
    import Modal_Decomposition.Base as Base

    assert Base.Cache is _MODULE.Cache is Cache
    assert Base.cache is _MODULE.cache is cache

    # typing 之类只是模块内部的实现细节，不应混入 __all__
    for leaked in ("Any", "Dict", "List", "Optional", "Tuple"):
        assert leaked not in _MODULE.__all__


# --------------------------------------------------------------------------- #
# 契约 2：签名合理
# --------------------------------------------------------------------------- #
def test_import_module_signature_accepts_module_name():
    """契约 2：``Cache.import_module`` 存在，首参 ``name`` 为必填模块名，其余参数可选并有默认值。"""
    assert callable(Cache.import_module)

    sig = inspect.signature(Cache.import_module)
    params = list(sig.parameters.values())

    assert params, "import_module 应至少有一个 name 参数"
    assert params[0].name == "name"
    assert params[0].default is inspect.Parameter.empty, "name 必须是必填的模块名"

    # 除 name 之外全部可选：没有必须由调用方补齐的隐藏参数
    for p in params[1:]:
        assert p.default is not inspect.Parameter.empty, (
            f"{p.name} 应有默认值（不要隐藏必填参数）"
        )
    assert "description" in sig.parameters and "verbose" in sig.parameters

    # 能真正接受一个模块名并返回模块对象
    assert inspect.ismodule(Cache.import_module("json"))


# --------------------------------------------------------------------------- #
# 契约 3：缓存语义
# --------------------------------------------------------------------------- #
def test_import_module_returns_the_real_module():
    """契约 3：``Cache.import_module("json")`` 返回的就是 ``json`` 模块本体。"""
    assert Cache.import_module("json") is json
    assert Cache.get("json") is json, "注册表里存的也必须是同一个对象"


def test_two_calls_return_the_same_object():
    """契约 3：两次调用返回同一对象（第二次直接命中缓存，不重新 import）。"""
    first = Cache.import_module("json")
    second = Cache.import_module("json")
    assert second is first


def test_cached_call_does_not_import_again(monkeypatch):
    """契约 3：缓存命中时不再触发 importlib 导入（用计数器证明“只导入一次”）。"""
    calls = []
    real_import = importlib.import_module

    def counting_import(name, *args, **kwargs):
        calls.append(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", counting_import)

    Cache.import_module("json")
    Cache.import_module("json")

    assert calls == ["json"], f"第二次调用不应再 import，实际导入序列为 {calls}"


# --------------------------------------------------------------------------- #
# 契约 4：缺失模块
# --------------------------------------------------------------------------- #
def test_missing_module_returns_none_without_raising():
    """契约 4：缺失模块返回 ``None`` 且不抛异常（``verbose=False`` 下的实际行为）。"""
    assert Cache.import_module("__nope__") is None
    # 第二次仍然返回 None：失败结果被记进 _exists，不再重复尝试 import
    assert Cache.import_module("__nope__") is None
    # 失败不算“注册”，注册表里不应出现它
    assert Cache.check("__nope__") is False
    assert "__nope__" not in Cache.names()
    assert "__nope__" not in Cache.descriptions()


def test_missing_module_only_raises_when_verbose():
    """契约 4：缺失模块唯一的失败出口是 ``verbose=True``，此时抛 ``ImportError``。"""
    with pytest.raises(ImportError):
        Cache.import_module("__nope_verbose__", verbose=True)
    assert Cache.check("__nope_verbose__") is False


# --------------------------------------------------------------------------- #
# 契约 5：唯一名注册 / 类与实例共用同一张表
# --------------------------------------------------------------------------- #
def test_cache_name_is_the_same_registry_as_cache_class():
    """契约 5：``cache`` 与 ``Cache`` 是同一个对象（别名），实例级调用同样能用同一张表。"""
    assert cache is Cache
    # 当前实现里 cache 就是类本身（不是类的一个实例），两种写法等价
    assert isinstance(cache, type)

    assert cache.import_module("math") is math
    assert Cache.check("math") is True

    instance = Cache()
    assert instance._entries is Cache._entries, "实例必须与类共用同一个注册表"
    # 文档里的实例级协议（__contains__/__len__/__getitem__）要用 cache() 这种实例形式；
    # 直接在类上写 ``"math" in cache`` 会走元类查找，因此不作为契约断言。
    assert "math" in instance
    assert len(instance) == 1
    assert instance["math"] is math
    assert "math" in instance.names()

    # 实例写入 -> 类级可见
    instance.import_module("statistics")
    assert Cache.import_module("statistics") is statistics
    assert set(instance.items()) == set(Cache.items())


# --------------------------------------------------------------------------- #
# 契约 6：副作用可控（进表 + 恢复全局状态）
# --------------------------------------------------------------------------- #
def test_new_module_enters_the_cache_table():
    """契约 6：新导入的模块确实写进缓存表，names/items/values/describe 都能查到。"""
    assert Cache.names() == [], "fixture 应已清空注册表"

    assert Cache.import_module("json", description="JSON 编解码") is json

    assert Cache.names() == ["json"]
    assert "json" in Cache._entries and Cache._entries["json"] is json
    assert Cache.items() == [("json", json)]
    assert Cache.values() == [json]
    assert Cache.describe("json") == "JSON 编解码"

    # 已存在时“永不覆盖”：重复调用不会再 import，也不改写描述
    assert Cache.import_module("json") is json
    assert Cache.describe("json") == "JSON 编解码"


def test_clear_empties_every_registry_store():
    """契约 6：``clear()`` 把缓存表连同“导入失败标记”一起清空，这是恢复全局状态的机制。"""
    Cache.import_module("json")
    Cache.import_module("__nope__")
    assert Cache.names() == ["json"]
    assert Cache.descriptions()
    assert Cache._exists.get("__nope__") is False

    Cache.clear()

    assert Cache.names() == []
    assert Cache.items() == []
    assert Cache.descriptions() == {}
    assert Cache._exists == {}


def test_registry_writes_do_not_leak_writer():
    """契约 6：本用例故意写脏注册表，还原由 autouse fixture 负责（与下一个用例配对）。"""
    Cache.add("pytest.cache.sentinel", json, description="隔离性探针")
    assert Cache.check("pytest.cache.sentinel") is True


def test_registry_writes_do_not_leak_reader():
    """契约 6：上一个用例写入的条目已被 fixture 还原，全局状态没有泄漏到后续用例。"""
    assert Cache.check("pytest.cache.sentinel") is False
    assert "pytest.cache.sentinel" not in Cache.names()
