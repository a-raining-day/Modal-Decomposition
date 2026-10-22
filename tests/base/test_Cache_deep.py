"""``Base.Cache`` 深度测试：导入缓存的所有契约分支 + 并发 + 快路径性能。

``cache`` 是进程级单例（类属性存状态），所以每个用例都用**唯一名字**并在结束后
恢复三张表，避免互相污染。测试分五组：

1. **键/值校验**：``_validate_name`` / ``_validate_api`` / ``_validate_description``；
2. **注册表语义**：``add`` 永不覆盖、``get`` 缺键报 KeyError（带已注册键）、
   ``describe``/``remove``/``clear``/``names``/``items`` 等视图；
3. **``import_module``**：正命中/负缓存/``verbose`` 抛错/``type(name) is str`` 快路径；
4. **``lazy`` 装饰器**：恰好一次、无参、异常不缓存、``functools.wraps``；
5. **并发**：8 线程 × 200 次仍是**一次** import（正/负两条路径）。

``table/base/cache/bench.csv`` 是命中路径的实测耗时（本文件导出）。
"""

from __future__ import annotations

import argparse
import importlib
import sys
import threading
from pathlib import Path

import numpy as np
import pytest

_TESTS = Path(__file__).resolve().parents[1]
if str(_TESTS) not in sys.path:
    sys.path.insert(0, str(_TESTS))

import _bench as B  # noqa: E402

B.bootstrap_src()

from Modal_Decomposition.Base import Cache, cache  # noqa: E402

_CacheMod = importlib.import_module("Modal_Decomposition.Base.Cache")


@pytest.fixture(autouse=True)
def _isolate_cache():
    entries = dict(cache._entries)
    exists = dict(cache._exists)
    descriptions = dict(cache._descriptions)
    try:
        yield
    finally:
        cache._entries = entries
        cache._exists = exists
        cache._descriptions = descriptions


# --------------------------------------------------------------------------- #
# 1. 校验
# --------------------------------------------------------------------------- #
def test_public_surface():
    import Modal_Decomposition.Base as BasePkg

    assert Cache is cache
    assert BasePkg.Cache is cache
    assert set(_CacheMod.__all__) == {"cache", "Cache"}
    for m in ("add", "get", "import_module", "lazy", "check", "describe",
              "remove", "clear", "names", "keys", "values", "items", "descriptions"):
        assert callable(getattr(cache, m))


def test_validate_name():
    cache._validate_name("ok") == "ok"
    for bad in (None, 1, b"x", ["a"]):
        with pytest.raises(TypeError, match="must be a str"):
            cache._validate_name(bad)
    for bad in ("", "   ", "\t\n"):
        with pytest.raises(ValueError, match="must not be empty"):
            cache._validate_name(bad)


def test_validate_api_rejects_none():
    with pytest.raises(TypeError, match="must not be None"):
        cache._validate_api(None)


def test_validate_description():
    assert cache._validate_description(None) is None
    assert cache._validate_description("") is None
    assert cache._validate_description("  ") is None
    assert cache._validate_description(" x ") == "x"
    with pytest.raises(TypeError):
        cache._validate_description(1)


# --------------------------------------------------------------------------- #
# 2. 注册表
# --------------------------------------------------------------------------- #
def test_add_and_get():
    obj = object()
    assert cache.add("deep_test.a", obj, description="d") is True
    assert cache.get("deep_test.a") is obj
    assert cache.check("deep_test.a") is True
    assert cache.describe("deep_test.a") == "d"


def test_add_never_overwrites():
    a, b = object(), object()
    assert cache.add("deep_test.dup", a) is True
    assert cache.add("deep_test.dup", b) is False
    assert cache.get("deep_test.dup") is a


def test_default_description_and_empty_description():
    cache.add("deep_test.nodesc", object())
    assert "no description" in cache.describe("deep_test.nodesc")
    cache.add("deep_test.blankdesc", object(), description="   ")
    assert "no description" in cache.describe("deep_test.blankdesc")


def test_get_missing_lists_registered_keys():
    cache.add("deep_test.known", object())
    with pytest.raises(KeyError) as ei:
        cache.get("deep_test.missing")
    assert "deep_test.missing" in str(ei.value)
    assert "deep_test.known" in str(ei.value)


def test_describe_missing_raises():
    with pytest.raises(KeyError):
        cache.describe("deep_test.nope")


def test_remove_and_clear():
    cache.add("deep_test.rm", object(), description="x")
    assert cache.remove("deep_test.rm") is True
    assert cache.check("deep_test.rm") is False
    assert cache.remove("deep_test.rm") is False
    assert "deep_test.rm" not in cache.descriptions()

    cache.add("deep_test.c1", object())
    cache.clear()
    assert len(cache()) == 0 and not cache.names()
    assert cache.descriptions() == {}


def test_views():
    cache.add("deep_test.v1", "api1", description="one")
    cache.add("deep_test.v2", "api2")
    assert set(cache.names()) == set(cache.keys()) >= {"deep_test.v1", "deep_test.v2"}
    assert cache.values() == [cache.get(n) for n in cache.names()]
    assert dict(cache.items()) == {n: cache.get(n) for n in cache.names()}
    inst = cache()
    assert "deep_test.v1" in inst
    assert inst["deep_test.v1"] == "api1"
    assert "deep_test" in repr(inst)


def test_dunders_only_work_on_instances():
    """特征化（不一致）：``__contains__`` / ``__len__`` / ``__getitem__`` / ``__repr__``
    没有 ``@classmethod``，因此 **类上直接用会 TypeError**，必须实例化 ``cache()``。

    这与 ``cache.add(...)``（classmethod，类上可用）的用法不一致：同一个类
    一半 API 是类级、一半是实例级。
    """
    cache.add("deep_test.d1", 1)
    with pytest.raises(TypeError):
        "deep_test.d1" in cache                       # noqa: B015
    with pytest.raises(TypeError):
        len(cache)
    with pytest.raises(TypeError):
        cache["deep_test.d1"]                          # noqa: B015
    inst = cache()
    assert "deep_test.d1" in inst
    assert len(inst) >= 1
    assert inst["deep_test.d1"] == 1
    with pytest.raises(KeyError):
        inst["deep_test.not-there"]


def test_instance_and_class_share_state():
    cache().add("deep_test.same", 1)
    assert cache.get("deep_test.same") == 1
    assert cache().check("deep_test.same") is True


# --------------------------------------------------------------------------- #
# 3. import_module
# --------------------------------------------------------------------------- #
def test_import_module_positive(monkeypatch):
    calls = []
    real = importlib.import_module
    monkeypatch.setattr(importlib, "import_module",
                        lambda name, *a, **k: (calls.append(name), real(name, *a, **k))[1])
    a = cache.import_module("json")
    b = cache.import_module("json")
    assert a is b and calls == ["json"], calls
    assert "imported module" in cache.describe("json")


def test_import_module_negative_cache(monkeypatch):
    calls = []
    real = importlib.import_module

    def spy(name, *a, **k):
        calls.append(name)
        return real(name, *a, **k)

    monkeypatch.setattr(importlib, "import_module", spy)
    assert cache.import_module("no_such_module_deep") is None
    assert cache.import_module("no_such_module_deep") is None
    assert calls == ["no_such_module_deep"], "负缓存后不再重试"


def test_import_module_verbose_raises():
    with pytest.raises(ImportError, match="is not pip"):
        cache.import_module("no_such_module_deep2", verbose=True)
    with pytest.raises(ImportError, match="is not pip"):
        cache.import_module("no_such_module_deep2", verbose=True)


def test_import_module_description():
    cache.import_module("math", description="stdlib math")
    assert cache.describe("math") == "stdlib math"


def test_import_module_invalid_name():
    with pytest.raises(TypeError):
        cache.import_module(123)
    with pytest.raises(ValueError):
        cache.import_module("   ")


def test_import_module_str_subclass_goes_slow_path():
    class MyStr(str):
        pass

    name = MyStr("json")
    assert cache.import_module(name) is importlib.import_module("json")


def test_existing_entry_wins_over_import():
    sentinel = object()
    cache.add("deep_test.pre", sentinel)
    assert cache.import_module("deep_test.pre") is sentinel, "永不覆盖"


# --------------------------------------------------------------------------- #
# 4. lazy
# --------------------------------------------------------------------------- #
def test_lazy_runs_once():
    calls = []

    @cache.lazy("deep_test.lazy", description="lazy obj")
    def loader():
        calls.append(1)
        return object()

    a = loader()
    b = loader()
    assert a is b and len(calls) == 1
    assert cache.describe("deep_test.lazy") == "lazy obj"


def test_lazy_rejects_arguments():
    @cache.lazy("deep_test.lazy_args")
    def loader():
        return object()

    loader()
    with pytest.raises(TypeError, match="takes no arguments"):
        loader(1)
    with pytest.raises(TypeError, match="takes no arguments"):
        loader(x=1)


def test_lazy_rejects_none_result():
    @cache.lazy("deep_test.lazy_none")
    def loader():
        return None

    with pytest.raises(TypeError, match="must not be None"):
        loader()


def test_lazy_exception_not_cached():
    state = {"n": 0}

    @cache.lazy("deep_test.lazy_err")
    def loader():
        state["n"] += 1
        if state["n"] == 1:
            raise RuntimeError("boom")
        return "ok"

    with pytest.raises(RuntimeError):
        loader()
    assert loader() == "ok"
    assert state["n"] == 2


def test_lazy_existing_key_skips_loader():
    cache.add("deep_test.lazy_pre", "existing")

    @cache.lazy("deep_test.lazy_pre")
    def loader():
        raise AssertionError("不该被调用")

    assert loader() == "existing"


def test_lazy_wraps_preserves_metadata():
    @cache.lazy("deep_test.lazy_meta")
    def loader():
        """我的文档。"""
        return 1

    assert loader.__name__ == "loader"
    assert "我的文档" in (loader.__doc__ or "")


# --------------------------------------------------------------------------- #
# 5. 并发
# --------------------------------------------------------------------------- #
def _run_threads(n_threads: int, n_calls: int, fn) -> list:
    out = []
    lock = threading.Lock()

    def worker():
        for _ in range(n_calls):
            r = fn()
            with lock:
                out.append(r)

    threads = [threading.Thread(target=worker) for _ in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return out


def test_concurrent_positive_imports_are_once(monkeypatch):
    calls = []
    real = importlib.import_module

    def spy(name, *a, **k):
        calls.append(name)
        return real(name, *a, **k)

    monkeypatch.setattr(importlib, "import_module", spy)
    cache._entries.pop("fractions", None)
    out = _run_threads(8, 200, lambda: cache.import_module("fractions"))
    assert len(out) == 1600
    assert calls.count("fractions") == 1, calls
    assert len({id(o) for o in out}) == 1


def test_concurrent_negative_imports_are_once(monkeypatch):
    calls = []
    real = importlib.import_module

    def spy(name, *a, **k):
        calls.append(name)
        return real(name, *a, **k)

    monkeypatch.setattr(importlib, "import_module", spy)
    out = _run_threads(8, 200, lambda: cache.import_module("no_such_module_deep3"))
    assert len(out) == 1600 and all(o is None for o in out)
    assert calls.count("no_such_module_deep3") == 1


def test_concurrent_add_keeps_first():
    a, b = object(), object()
    out = _run_threads(4, 50, lambda: cache.add("deep_test.race", a))
    assert out.count(True) == 1
    assert cache.get("deep_test.race") is a
    assert cache.add("deep_test.race", b) is False


# --------------------------------------------------------------------------- #
# 导出
# --------------------------------------------------------------------------- #
def benchmark() -> list[dict]:
    cache.import_module("json")
    rows = []

    def bench(label, fn, repeat=20_000):
        r = B.timeit(fn, repeat=1, warmup=1)      # 单次计时不够稳，改用批
        import time
        t0 = time.perf_counter()
        for _ in range(repeat):
            fn()
        dt = (time.perf_counter() - t0) / repeat * 1e9
        rows.append({"op": label, "ns_per_call": round(dt, 1), "repeat": repeat})
        return dt

    bench("import_module hit", lambda: cache.import_module("json"))
    bench("import_module negative", lambda: cache.import_module("no_such_module_deep3"))
    bench("get hit", lambda: cache.get("json"))
    bench("check hit", lambda: cache.check("json"))
    inst = cache()
    bench("__contains__", lambda: "json" in inst)
    bench("len", lambda: len(inst))
    return rows


def export() -> None:
    rows = benchmark()
    B.write_csv(B.TABLE / "base" / "cache" / "bench.csv", rows)

    B.plt_style()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    ax.bar([r["op"] for r in rows], [r["ns_per_call"] for r in rows])
    ax.set_ylabel("ns / call")
    ax.set_title("Base.Cache — per-call cost (incl. ~25 ns lambda/bound-method floor)")
    ax.tick_params(axis="x", labelrotation=20, labelsize=7)
    for i, r in enumerate(rows):
        ax.text(i, r["ns_per_call"], f"{r['ns_per_call']:.0f}", ha="center",
                va="bottom", fontsize=7)
    fig.tight_layout()
    p = B.savefig(fig, B.FIG / "base" / "cache" / "cache_bench.png")
    plt.close(fig)
    print(f"wrote {B.TABLE / 'base' / 'cache' / 'bench.csv'}")
    print(f"wrote {p}")
    B.print_table(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    B.add_export_flag(ap)
    args = ap.parse_args()
    if args.export:
        export()
    else:
        B.print_table(benchmark())


# keep numpy referenced (used by type checks in some parametrisations)
_ = np
