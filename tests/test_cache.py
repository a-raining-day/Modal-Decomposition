"""
Extreme-case tests for the process-wide import cache
(``Modal_Decomposition.Base.Cache.cache``).

Covers:
- basic add / get / check round-trips;
- extreme inputs: empty / whitespace-only / non-string names, None api,
  non-string descriptions, very long and unicode names;
- the uniqueness contract: duplicates never overwrite, across instances and
  across threads;
- the description function (``describe``) for registered modules;
- concurrency safety of concurrent registration.
"""

from concurrent.futures import ThreadPoolExecutor

import pytest

from Modal_Decomposition.Base import Cache, cache
from Modal_Decomposition.Base.Cache import cache as cache_direct


@pytest.fixture(autouse=True)
def _isolated_cache():
    """Each test starts and ends with an empty process-wide cache."""
    cache.clear()
    yield
    cache.clear()


# ------------------------------------------------------------------ #
# basic contract
# ------------------------------------------------------------------ #
def test_add_get_roundtrip():
    api = object()
    assert cache.add("m1", api) is True
    assert cache.get("m1") is api


def test_add_returns_true_once_then_false():
    assert cache.add("m1", object()) is True
    assert cache.add("m1", object()) is False


def test_check_before_and_after():
    assert cache.check("m1") is False
    cache.add("m1", object())
    assert cache.check("m1") is True


def test_get_missing_raises_keyerror():
    with pytest.raises(KeyError, match="m1"):
        cache.get("m1")


def test_get_missing_error_lists_known_entries():
    cache.add("known", object())
    with pytest.raises(KeyError, match="known"):
        cache.get("absent")


def test_cache_alias_is_same_class():
    assert Cache is cache is cache_direct


# ------------------------------------------------------------------ #
# extreme inputs: names
# ------------------------------------------------------------------ #
@pytest.mark.parametrize("bad_name", ["", "   ", "\t", "\n\t "])
def test_empty_or_whitespace_name_rejected(bad_name):
    with pytest.raises(ValueError, match="empty or whitespace"):
        cache.add(bad_name, object())


@pytest.mark.parametrize("bad_name", [None, 123, 3.14, b"scipy.signal", object(), ["a"]])
def test_non_str_name_rejected(bad_name):
    with pytest.raises(TypeError, match="must be a str"):
        cache.add(bad_name, object())


@pytest.mark.parametrize("bad_name", [None, 123, b"x"])
def test_non_str_name_rejected_by_get_check_describe(bad_name):
    with pytest.raises(TypeError):
        cache.get(bad_name)
    with pytest.raises(TypeError):
        cache.check(bad_name)
    with pytest.raises(TypeError):
        cache.describe(bad_name)


def test_very_long_name_accepted():
    name = "m" * 1_000_000
    assert cache.add(name, object()) is True
    assert cache.check(name) is True


def test_unicode_name_accepted():
    name = "scipy.信号处理子模块"
    api = object()
    assert cache.add(name, api) is True
    assert cache.get(name) is api


def test_name_kept_verbatim_with_surrounding_spaces():
    """Keys are exact strings: ' x ' and 'x' are two distinct entries."""
    api = object()
    assert cache.add(" x ", api) is True
    assert cache.check(" x ") is True
    assert cache.check("x") is False
    assert cache.get(" x ") is api


def test_case_sensitive_keys_are_distinct():
    """Uniqueness is per exact key; case variants never collide or merge."""
    assert cache.add("scipy.signal", object()) is True
    assert cache.add("SCIPY.signal", object()) is True
    assert cache.add("Scipy.Signal", object()) is True
    assert len(cache()) == 3


# ------------------------------------------------------------------ #
# extreme inputs: api and description
# ------------------------------------------------------------------ #
def test_none_api_rejected():
    with pytest.raises(TypeError, match="must not be None"):
        cache.add("m1", None)


def test_any_non_none_api_accepted():
    for api in (0, "", False, (), object):
        key = f"k{id(api)}"
        assert cache.add(key, api) is True
        assert cache.get(key) is api


def test_non_str_description_rejected():
    with pytest.raises(TypeError, match="description"):
        cache.add("m1", object(), description=123)


def test_whitespace_description_treated_as_absent():
    cache.add("m1", object(), description="   ")
    assert "no description provided" in cache.describe("m1")


# ------------------------------------------------------------------ #
# uniqueness contract (no duplicate registration)
# ------------------------------------------------------------------ #
def test_duplicate_keeps_original_api():
    first = object()
    assert cache.add("m1", first, description="first") is True
    assert cache.add("m1", object(), description="second") is False
    assert cache.get("m1") is first
    assert cache.describe("m1") == "first"


def test_duplicate_does_not_change_length():
    cache.add("m1", object())
    cache.add("m1", object())
    cache.add("m1", object())
    assert len(cache()) == 1


def test_store_shared_between_instances_and_class():
    c1, c2 = cache(), cache()
    api = object()
    assert c1.add("shared", api) is True
    # duplicate seen through another instance and through the class itself
    assert c2.check("shared") is True
    assert c2.add("shared", object()) is False
    assert cache.check("shared") is True
    assert cache.add("shared", object()) is False
    assert c1.get("shared") is c2.get("shared") is cache.get("shared") is api


def test_remove_is_the_only_replacement_path():
    first = object()
    second = object()
    cache.add("m1", first)
    assert cache.add("m1", second) is False  # still blocked
    assert cache.remove("m1") is True
    assert cache.add("m1", second) is True  # allowed after explicit removal
    assert cache.get("m1") is second


def test_remove_missing_returns_false():
    assert cache.remove("absent") is False


def test_clear_empties_everything():
    cache.add("m1", object(), description="d1")
    cache.add("m2", object(), description="d2")
    cache.clear()
    assert len(cache()) == 0
    assert cache.descriptions() == {}
    assert cache.check("m1") is False


# ------------------------------------------------------------------ #
# description function for registered modules
# ------------------------------------------------------------------ #
def test_describe_returns_registered_description():
    cache.add("m1", object(), description="信号处理: hilbert / butter / filtfilt")
    assert cache.describe("m1") == "信号处理: hilbert / butter / filtfilt"


def test_describe_strips_surrounding_whitespace():
    cache.add("m1", object(), description="  padded text  ")
    assert cache.describe("m1") == "padded text"


def test_describe_fallback_notice_without_description():
    cache.add("m1", object())
    desc = cache.describe("m1")
    assert "no description provided" in desc
    assert "m1" in desc


def test_describe_missing_raises_keyerror():
    with pytest.raises(KeyError, match="m1"):
        cache.describe("m1")


def test_descriptions_snapshot():
    cache.add("a", object(), description="AAA")
    cache.add("b", object(), description="BBB")
    snap = cache.descriptions()
    assert snap == {"a": "AAA", "b": "BBB"}
    # snapshot must not be a live view
    snap.clear()
    assert cache.descriptions() == {"a": "AAA", "b": "BBB"}


# ------------------------------------------------------------------ #
# views and protocol
# ------------------------------------------------------------------ #
def test_len_contains_getitem():
    api = object()
    cache.add("m1", api)
    c = cache()  # len / in / [] are instance-level protocol
    assert len(c) == 1
    assert "m1" in c
    assert "absent" not in c
    assert c["m1"] is api
    with pytest.raises(KeyError):
        _ = c["absent"]


def test_views_are_sorted_by_key():
    cache.add("b", 2, description="B")
    cache.add("a", 1, description="A")
    cache.add("c", 3, description="C")
    assert cache.names() == ["a", "b", "c"]
    assert cache.keys() == ["a", "b", "c"]
    assert cache.values() == [1, 2, 3]
    assert cache.items() == [("a", 1), ("b", 2), ("c", 3)]
    assert cache.descriptions() == {"a": "A", "b": "B", "c": "C"}


def test_repr_lists_entries():
    cache.add("m1", object())
    assert repr(cache()) == "<cache entries=['m1'] n=1>"


def test_verbose_prints_registration_and_duplicate(capsys):
    cache.add("m1", object(), verbose=True, description="AAA")
    out = capsys.readouterr().out
    assert "registered 'm1'" in out

    cache.add("m1", object(), verbose=True)
    out = capsys.readouterr().out
    assert "duplicate registration ignored" in out
    assert "registered 'm1'" not in out


# ------------------------------------------------------------------ #
# concurrency extreme cases
# ------------------------------------------------------------------ #
def test_concurrent_adds_of_distinct_names_all_succeed():
    n_threads, per_thread = 32, 64
    names = [f"mod{i}.{j}" for i in range(n_threads) for j in range(per_thread)]

    def worker(chunk):
        return [cache.add(n, object()) for n in chunk]

    chunks = [names[i::n_threads] for i in range(n_threads)]
    with ThreadPoolExecutor(max_workers=n_threads) as ex:
        per_thread_results = list(ex.map(worker, chunks))

    flat = [r for chunk_results in per_thread_results for r in chunk_results]
    assert all(flat)
    assert len(cache()) == len(names)
    assert sorted(cache.names()) == sorted(names)


def test_concurrent_adds_of_same_name_register_exactly_once():
    with ThreadPoolExecutor(max_workers=32) as ex:
        futures = [
            ex.submit(lambda i=i: cache.add("the-one", i)) for i in range(32)
        ]
        results = [f.result() for f in futures]

    assert results.count(True) == 1
    assert results.count(False) == 31
    assert len(cache()) == 1
    assert cache.get("the-one") in range(32)


def test_concurrent_gets_during_adds_are_consistent():
    """Readers either miss (KeyError) or get the final api, never a torn value."""
    cache.add("shared", "final-value")
    errors = []

    def reader():
        for _ in range(200):
            try:
                v = cache.get("shared")
            except KeyError:
                errors.append("miss")
            else:
                assert v == "final-value", f"torn value: {v!r}"

    def writer():
        for i in range(50):
            assert cache.add(f"extra-{i}", i) in (True, False)

    with ThreadPoolExecutor(max_workers=16) as ex:
        futures = [ex.submit(reader) for _ in range(8)] + [
            ex.submit(writer) for _ in range(2)
        ]
        for f in futures:
            f.result()

    assert errors == []  # "shared" was pre-registered, readers never miss
    for i in range(50):
        assert cache.get(f"extra-{i}") == i


# ------------------------------------------------------------------ #
# import_module: 集中式惰性导入抽象 (替代 "flag is None + try import")
# ------------------------------------------------------------------ #
def test_import_module_imports_and_caches_once():
    assert cache.check("json") is False
    mod = cache.import_module("json")
    import json

    assert mod is json
    assert cache.check("json") is True
    assert cache.get("json") is mod
    assert cache.import_module("json") is mod  # 之后调用纯查表, 不再 import
    assert "imported module 'json'" in cache.describe("json")


def test_import_module_description_stored():
    cache.import_module("math", description="标准库 math 模块")
    assert cache.describe("math") == "标准库 math 模块"


def test_import_module_missing_module_raises_and_not_cached():
    with pytest.raises(ImportError):
        cache.import_module("modal_decomposition_no_such_module_xyz")
    # 失败不缓存 (标志保持"未就绪"), 下次调用可重试
    assert cache.check("modal_decomposition_no_such_module_xyz") is False


def test_import_module_existing_key_wins_without_reimport():
    sentinel = object()
    cache.add("json", sentinel, description="pre-registered")
    assert cache.import_module("json") is sentinel  # 不触发真实 import


@pytest.mark.parametrize("bad_name", ["", "   ", None, 123])
def test_import_module_rejects_bad_name(bad_name):
    with pytest.raises((TypeError, ValueError)):
        cache.import_module(bad_name)


def test_import_module_verbose_prints(capsys):
    cache.import_module("math", verbose=True)
    assert "registered 'math'" in capsys.readouterr().out


# ------------------------------------------------------------------ #
# lazy: 装饰器版惰性加载 (对外透明, 恰好执行一次)
# ------------------------------------------------------------------ #
def test_lazy_loader_executes_exactly_once():
    calls = []

    @cache.lazy("lazy-once", description="exactly once")
    def _load():
        calls.append(1)
        return object()

    a = _load()
    b = _load()
    assert len(calls) == 1
    assert a is b is cache.get("lazy-once")
    assert cache.describe("lazy-once") == "exactly once"


def test_lazy_loader_parallel_exactly_once():
    calls = []

    @cache.lazy("lazy-parallel")
    def _load():
        calls.append(1)
        return object()

    with ThreadPoolExecutor(max_workers=8) as ex:
        results = list(ex.map(lambda _: _load(), range(40)))
    assert len(calls) == 1
    assert all(r is results[0] for r in results)


def test_lazy_loader_exception_not_cached_and_retryable():
    state = {"fail": True}

    @cache.lazy("lazy-retry")
    def _load():
        if state["fail"]:
            raise ImportError("no module")
        return object()

    with pytest.raises(ImportError):
        _load()
    assert cache.check("lazy-retry") is False  # 失败不缓存

    state["fail"] = False
    api = _load()
    assert cache.check("lazy-retry") is True
    assert _load() is api


def test_lazy_loader_rejects_arguments():
    @cache.lazy("lazy-args")
    def _load():
        return object()

    with pytest.raises(TypeError, match="no arguments"):
        _load(1)
    with pytest.raises(TypeError, match="no arguments"):
        _load(kw=1)
    assert _load() is not None


def test_lazy_loader_rejects_none_result():
    @cache.lazy("lazy-none")
    def _load():
        return None

    with pytest.raises(TypeError, match="must not be None"):
        _load()
    assert cache.check("lazy-none") is False


def test_lazy_existing_key_skips_factory():
    sentinel = object()
    cache.add("lazy-existing", sentinel)
    executed = []

    @cache.lazy("lazy-existing")
    def _load():
        executed.append(1)
        raise AssertionError("factory must not run")

    assert _load() is sentinel
    assert executed == []


def test_lazy_preserves_metadata():
    @cache.lazy("lazy-meta")
    def _load():
        """docstring kept"""
        return object()

    assert _load.__name__ == "_load"
    assert _load.__doc__ == "docstring kept"


def test_lazy_invalid_name_rejected_at_decoration():
    with pytest.raises(ValueError):

        @cache.lazy("")
        def _load():
            return object()


def test_lazy_verbose_prints(capsys):
    @cache.lazy("lazy-verbose", verbose=True)
    def _load():
        return object()

    _load()
    assert "registered 'lazy-verbose'" in capsys.readouterr().out
