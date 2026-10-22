"""``Base.Chunk`` 深度测试：分块语义、memmap 模式、流式填充、块长计算。

与 ``tests/base/test_Chunk.py`` 的分工：那份是契约烟测（56 条小数组），本文件专攻

1. **块长计算**（``chunk_size``）的边界与单位——含一处**单位混用**：
   ``nbytes``（字节）与 ``DEFAULT_FILL_CHUNK_SIZE``（元素数）直接比较，中间分支
   还把字节数当元素数返回。后果是**自适应块长恒覆盖整个数组**，``chunk()`` 实际上
   只有一块；夸张点说，600 MiB 输入也只切出 1 块。
2. **块迭代**：视图 vs 磁盘、边界、axis、空维、0-d、dtype 转换、临时文件生命周期
   （含**提前 break** 的清理）。
3. **流式聚合**：``chunked_map`` / ``chunked_fill`` / ``fill_memmap`` 的正确性、
   峰值内存（应为"一块 + 输出"）、类型与形状错误。
4. **契约面**：非法块长/轴/模式、列表输入在 NumPy 2 下的行为。

导出::

    .\\.venv\\Scripts\\python.exe tests/base/test_Chunk_deep.py --export

规模轴在 ``tests/size_axis/run_scale.py --tool chunk``。
"""

from __future__ import annotations

import argparse
import gc
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

_TESTS = Path(__file__).resolve().parents[1]
if str(_TESTS) not in sys.path:
    sys.path.insert(0, str(_TESTS))

import _bench as B  # noqa: E402

B.bootstrap_src()

from Modal_Decomposition.Base import (  # noqa: E402
    CHUNK_MOD,
    Chunk,
    cleanup_memmaps,
    temp_memmap,
)
import importlib

_ChunkMod = importlib.import_module("Modal_Decomposition.Base.Chunk")


DEFAULT_FILL = _ChunkMod.DEFAULT_FILL_CHUNK_SIZE


# --------------------------------------------------------------------------- #
# 1. 块长计算
# --------------------------------------------------------------------------- #
def test_default_fill_chunk_size_value():
    assert DEFAULT_FILL == 8 * 1024 * 1024 == 8388608


def test_chunk_size_small_input_returns_default():
    for nbytes in (0, 1, 8, 1 << 20, DEFAULT_FILL - 1, DEFAULT_FILL):
        assert Chunk.chunk_size(nbytes) == DEFAULT_FILL


def test_chunk_size_units_are_bytes_in_elements_out():
    """回归（本轮修好）：入参是**字节**，返回值是**元素数**。

    旧实现在超出 8 MiB 后直接 `return nbytes`（字节当元素），会让"超出预算"的输入
    拿到 8 倍过大的块（5 GiB 输入会切出 40 GiB 的块）。现在统一按 itemsize 折算。
    """
    for size_mib, dtype in ((9, np.float64), (50, np.float32)):
        arr = np.zeros(size_mib * 1024 * 1024 // np.dtype(dtype).itemsize, dtype=dtype)
        cs = Chunk.chunk_size(arr.nbytes, itemsize=arr.itemsize)
        assert isinstance(cs, int) and cs >= 1
        # 预算内 ⇒ 块长覆盖整个数组（单块是设计意图，不是"字节当元素"）
        assert cs >= arr.size, (size_mib, cs, arr.size)
        # 返回值确实是元素数：换成字节后与"块覆盖输入"自洽
        assert cs * arr.itemsize >= arr.nbytes


def test_chunk_size_within_budget_is_single_chunk():
    """在 0.75·预算以内，块长覆盖整个数组 ⇒ **单块是设计意图**（不是缺陷）。"""
    for size_mib in (1, 50, 600):
        nbytes = size_mib * 1024 * 1024
        n_elems = nbytes // 8
        assert Chunk.chunk_size(nbytes) >= n_elems, size_mib
        arr = np.zeros(min(n_elems, 1 << 16), dtype=np.float64)
        assert sum(1 for _ in Chunk.chunk(arr)) == 1


def test_chunk_size_above_budget_is_capped(monkeypatch):
    """超出预算时块长必须被压到 ``0.75 × budget`` 字节（而不是原样返回字节数）。"""
    import importlib

    CM = importlib.import_module("Modal_Decomposition.Base.Chunk")

    budget = 1 << 30                                    # 1 GiB
    monkeypatch.setattr(CM, "memory_budget",
                        lambda **kw: {"budget": budget, "ok": True})
    nbytes = 4 << 30                                    # 4 GiB
    expect8 = int(budget * 0.75) // 8
    cs = Chunk.chunk_size(nbytes, itemsize=8)
    assert cs == expect8, (cs, expect8)
    assert cs * 8 <= int(budget * 0.75) + 8
    # itemsize=4 时元素数翻倍、字节上限不变
    expect4 = int(budget * 0.75) // 4
    assert Chunk.chunk_size(nbytes, itemsize=4) == expect4


def test_chunk_size_itemsize_validation():
    with pytest.raises(TypeError):
        Chunk.chunk_size(1 << 20, itemsize=True)
    with pytest.raises(TypeError):
        Chunk.chunk_size(1 << 20, itemsize=2.0)
    with pytest.raises(ValueError):
        Chunk.chunk_size(1 << 20, itemsize=0)


def test_chunk_size_ndarray_input_warns():
    arr = np.zeros(16)
    with pytest.warns(UserWarning, match="get nbtyes from its attribute"):
        assert Chunk.chunk_size(arr) == DEFAULT_FILL


def test_chunk_size_rejects_bad_arguments():
    with pytest.raises(ValueError):
        Chunk.chunk_size(3.5)                       # 非 int
    with pytest.raises(ValueError):
        Chunk.chunk_size(DEFAULT_FILL, chunk_ratio=1.5)
    with pytest.raises(ValueError):
        Chunk.chunk_size(DEFAULT_FILL, chunk_ratio=0.0)
    with pytest.raises(ValueError):
        Chunk.chunk_size(DEFAULT_FILL, chunk_ratio=1)   # int 不是 float
    with pytest.raises(TypeError):
        Chunk.chunk_size(DEFAULT_FILL, extras_bytes=1.0)
    with pytest.raises(ValueError):
        Chunk.chunk_size(DEFAULT_FILL, extras_bytes=-1)


def test_chunk_size_extras_bytes_keeps_margin():
    big = 10 * 1024 ** 3                            # 10 GiB，必然落到 limit 分支
    a = Chunk.chunk_size(big)
    b = Chunk.chunk_size(big, extras_bytes=1 << 30)
    assert b <= a


# --------------------------------------------------------------------------- #
# 2. chunk 迭代：内存视图
# --------------------------------------------------------------------------- #
def test_chunk_memory_views():
    S = np.arange(10.0)
    chunks = list(Chunk.chunk(S, chunk_size=4))
    assert [c.size for c in chunks] == [4, 4, 2]
    assert np.array_equal(np.concatenate(chunks), S)
    assert chunks[0].base is S or chunks[0].base is not None, "应为视图而非拷贝"


def test_chunk_is_iterator_by_default_and_single_chunk_without_size():
    S = np.arange(1000.0)
    it = Chunk.chunk(S)
    assert hasattr(it, "__next__")
    chunks = list(it)
    assert len(chunks) == 1, "自适应块长覆盖全数组 ⇒ 只有 1 块"


def test_chunk_empty_dim_raises():
    with pytest.raises(ValueError, match="at least 1"):
        list(Chunk.chunk(np.array([], dtype=float)))
    with pytest.raises(ValueError, match="at least 1"):
        list(Chunk.chunk(np.zeros((0, 3))))


def test_chunk_0d():
    assert [c.shape for c in Chunk.chunk(np.array(5.0))] == [()]
    # 特征化：``S.ndim == 0 and S.size == 0`` 在 NumPy 里不可达（0-d 数组 size 恒 1），
    # 所以 "chunking requires at least 1 dimension, got 0-d array" 是死代码。
    # 真正会命中的空输入是 1-D 空数组 → 走 _iter_chunks 的 "at least 1" 分支。
    with pytest.raises(ValueError, match="at least 1"):
        list(Chunk.chunk(np.array([], dtype=float)))


def test_chunk_axis():
    S = np.arange(12.0).reshape(3, 4)
    cols = list(Chunk.chunk(S, chunk_size=2, axis=1))
    assert [c.shape for c in cols] == [(3, 2), (3, 2)]
    rows = list(Chunk.chunk(S, chunk_size=2, axis=0))
    assert [c.shape for c in rows] == [(2, 4), (1, 4)]
    with pytest.raises(ValueError, match="out of bounds"):
        list(Chunk.chunk(S, chunk_size=2, axis=2))
    with pytest.raises(ValueError, match="out of bounds"):
        list(Chunk.chunk(S, chunk_size=2, axis=-1))


def test_chunk_axis_type():
    with pytest.raises(TypeError):
        list(Chunk.chunk(np.zeros((2, 3)), chunk_size=1, axis=True))
    with pytest.raises(TypeError):
        list(Chunk.chunk(np.zeros((2, 3)), chunk_size=1, axis=1.0))


def test_chunk_rejects_bad_chunk_size():
    S = np.arange(8.0)
    with pytest.raises(ValueError):
        list(Chunk.chunk(S, chunk_size=0))
    with pytest.raises(ValueError):
        list(Chunk.chunk(S, chunk_size=-3))
    with pytest.raises(TypeError):
        list(Chunk.chunk(S, chunk_size=True))
    with pytest.raises(TypeError):
        list(Chunk.chunk(S, chunk_size=2.0))
    with pytest.raises(TypeError):
        list(Chunk.chunk(S, chunk_size=np.int64(2)))


def test_chunk_rejects_bad_mod():
    with pytest.raises(ValueError, match="mod must be one of"):
        list(Chunk.chunk(np.arange(4.0), chunk_size=2, mod="nope"))
    assert CHUNK_MOD("memory") is CHUNK_MOD.Memory
    assert CHUNK_MOD("exo-memory") is CHUNK_MOD.ExoMemory
    assert CHUNK_MOD.Memory.value == "memory"


def test_chunk_dtype_kind_restriction():
    assert list(Chunk.chunk(np.arange(4, dtype=np.uint8), chunk_size=2))
    assert list(Chunk.chunk(np.arange(4, dtype=np.int16), chunk_size=2))
    assert list(Chunk.chunk(np.arange(4.0, dtype=np.float32), chunk_size=2))
    with pytest.raises(ValueError, match="int or float"):
        list(Chunk.chunk(np.array([True, False]), chunk_size=1))
    with pytest.raises(ValueError, match="int or float"):
        list(Chunk.chunk(np.zeros(4, dtype=complex), chunk_size=2))


def test_chunk_list_input_on_numpy2():
    """NumPy ≥ 2 拒绝 ``np.asarray(list, copy=False, dtype=...)`` —— 原样抛 ValueError。"""
    with pytest.raises(ValueError):
        list(Chunk.chunk([1, 2, 3], chunk_size=2))


# --------------------------------------------------------------------------- #
# 3. chunk 迭代：exo-memory（磁盘）
# --------------------------------------------------------------------------- #
def test_exo_memory_roundtrip_and_cleanup(tmp_path):
    S = np.arange(100.0)
    chunks = list(Chunk.chunk(S, chunk_size=30, mod="exo-memory", temp_dir=str(tmp_path)))
    assert all(isinstance(c, np.memmap) for c in chunks)
    assert np.array_equal(np.concatenate([np.asarray(c) for c in chunks]), S)
    # 新契约：块在仍被引用期间保持有效；临时文件由 cleanup_memmaps/退出回收
    assert len(list(tmp_path.iterdir())) == len(chunks)
    del chunks
    gc.collect()
    assert cleanup_memmaps() is True
    assert list(tmp_path.iterdir()) == [], "drop 引用后 cleanup_memmaps 必须清空临时文件"


def test_exo_memory_cleanup_on_early_break(tmp_path):
    S = np.arange(100.0)
    for c in Chunk.chunk(S, chunk_size=30, mod="exo-memory", temp_dir=str(tmp_path)):
        del c
        break                       # 不耗尽生成器：已产出的块仍登记在 Memory 里
    gc.collect()
    assert cleanup_memmaps() is True
    assert list(tmp_path.iterdir()) == []


def test_exo_memory_dtype_override(tmp_path):
    S = np.arange(6.0)                                 # float64
    chunks = list(Chunk.chunk(S, chunk_size=4, mod="exo-memory",
                              dtype=np.float32, temp_dir=str(tmp_path)))
    assert chunks[0].dtype == np.float32
    assert np.array_equal(np.asarray(chunks[0]), np.arange(4.0, dtype=np.float32))
    del chunks
    gc.collect()
    cleanup_memmaps()


def test_exo_memory_chunks_stay_valid_after_iteration(tmp_path):
    """回归（本轮修好）：迭代结束后继续读块**不再崩溃**。

    旧实现在生成器 ``finally`` 里把所有映射一起 close、文件全删，之后 dereference
    触发 access violation（开发期实测：``np.array_equal`` 处 Windows fatal exception）。
    现在每块由 ``temp_memmap`` 拥有，只要还被引用就可读。
    """
    chunks = list(Chunk.chunk(np.arange(6.0), chunk_size=4, mod="exo-memory",
                              temp_dir=str(tmp_path)))
    assert not any(c._mmap.closed for c in chunks), "仍被引用的块不得关闭"
    assert np.array_equal(np.asarray(chunks[0]), np.array([0.0, 1.0, 2.0, 3.0]))
    assert np.array_equal(np.asarray(chunks[1]), np.array([4.0, 5.0]))
    with pytest.warns(UserWarning, match="not be closed"):
        assert cleanup_memmaps() is False, "还有活引用时删不掉，应返回 False 而不是抛 TypeError"
    assert np.array_equal(np.asarray(chunks[0]), np.array([0.0, 1.0, 2.0, 3.0]))
    del chunks
    gc.collect()


def test_exo_memory_creates_temp_dir(tmp_path):
    d = tmp_path / "nested" / "chunks"
    list(Chunk.chunk(np.arange(4.0), chunk_size=2, mod="exo-memory", temp_dir=str(d)))
    assert d.is_dir()


# --------------------------------------------------------------------------- #
# 4. chunked_map
# --------------------------------------------------------------------------- #
def test_chunked_map_basic():
    S = np.arange(10.0)
    out = Chunk.chunked_map(lambda c: c * 2.0, S, chunk_size=3)
    assert np.array_equal(out, S * 2.0)
    assert out.dtype == np.float64


def test_chunked_map_dtype_and_out():
    S = np.arange(10, dtype=np.int64)
    out32 = Chunk.chunked_map(lambda c: c, S, chunk_size=3, dtype=np.float32)
    assert out32.dtype == np.float32
    buf = np.empty_like(S)
    assert Chunk.chunked_map(lambda c: c + 1, S, chunk_size=3, out=buf) is buf
    assert np.array_equal(buf, S + 1)


def test_chunked_map_out_shape_mismatch():
    with pytest.raises(ValueError, match="must equal input shape"):
        Chunk.chunked_map(lambda c: c, np.arange(4.0), chunk_size=2, out=np.empty(5))


def test_chunked_map_empty():
    out = Chunk.chunked_map(lambda c: c, np.array([]), chunk_size=4)
    assert out.size == 0


def test_chunked_map_0d_rejected():
    with pytest.raises(ValueError, match="at least 1 dimension"):
        Chunk.chunked_map(lambda c: c, np.array(1.0))


def test_chunked_map_memmap_out(tmp_path):
    S = np.arange(9.0)
    fp = temp_memmap((9,), np.float64, temp_dir=str(tmp_path))
    out = Chunk.chunked_map(lambda c: c * 3, S, chunk_size=4, out=fp)
    assert isinstance(out, np.memmap)
    assert np.array_equal(np.asarray(out), S * 3)
    path = fp.filename
    del fp, out
    gc.collect()
    try:
        Path(path).unlink(missing_ok=True)
    except OSError:
        pass


def test_chunked_map_chunk_size_errors():
    with pytest.raises(ValueError):
        Chunk.chunked_map(lambda c: c, np.arange(4.0), chunk_size=0)
    with pytest.raises(TypeError):
        Chunk.chunked_map(lambda c: c, np.arange(4.0), chunk_size=2.0)


def test_chunked_map_peak_memory_is_output_plus_chunk():
    """恒等函数下峰值应≈「输出 + 0」（输入是视图，不复制）。

    真正做运算时每个 chunk 还会产生自己的临时数组（例如 ``c * 2.0 + 1.0`` 是两个
    chunk 大小的临时量），这属于 ``func`` 的开销，不是分块框架的。
    """
    import tracemalloc

    S = B.make_signal(20)                       # 20 MiB
    tracemalloc.start()
    try:
        out = Chunk.chunked_map(lambda c: c, S, chunk_size=1 << 20)
        peak = tracemalloc.get_traced_memory()[1] / B.MiB
    finally:
        tracemalloc.stop()
    assert out.shape == S.shape
    assert peak < 20 + 4, f"恒等映射峰值 {peak:.1f} MiB 应≈输出 20 MiB"


def test_chunked_map_func_temporaries_are_per_chunk():
    """给定 ``chunk_size`` 后，``func`` 的临时量只随**一块**增长，不随输入增长。"""
    import tracemalloc

    peaks = {}
    for chunk_mib in (1, 4):
        S = B.make_signal(20)
        tracemalloc.start()
        try:
            Chunk.chunked_map(lambda c: c * 2.0 + 1.0, S,
                              chunk_size=chunk_mib * 1024 * 1024 // 8)
            peaks[chunk_mib] = tracemalloc.get_traced_memory()[1] / B.MiB
        finally:
            tracemalloc.stop()
    assert peaks[4] > peaks[1], "块越大，func 临时量越大（证明其规模随块而非输入）"
    assert peaks[4] < 20 + 4 * 3 + 8, peaks


# --------------------------------------------------------------------------- #
# 5. chunked_fill / fill_memmap
# --------------------------------------------------------------------------- #
def test_chunked_fill_memory():
    src = np.arange(10.0)
    dst = np.empty_like(src)
    out = Chunk.chunked_fill(dst, src, chunk_size=3)
    assert out is dst and np.array_equal(dst, src)


def test_chunked_fill_type_errors():
    with pytest.raises(TypeError, match="dst must be"):
        Chunk.chunked_fill([0.0] * 3, np.arange(3.0))
    with pytest.raises(TypeError, match="src must be"):
        Chunk.chunked_fill(np.empty(3), [0.0, 1.0, 2.0])


def test_chunked_fill_src_longer_is_silently_truncated():
    """特征化（缺陷）：``dst`` 比 ``src`` 短时**不报错**，只拷前 ``len(dst)`` 个元素，
    尾部被静默丢掉；反方向（``dst`` 更长）才会在广播时抛 ValueError。
    """
    dst = np.empty(3)
    out = Chunk.chunked_fill(dst, np.arange(5.0), chunk_size=2)
    assert np.array_equal(out, np.array([0.0, 1.0, 2.0])), "尾部被静默丢弃"
    with pytest.raises(ValueError):
        Chunk.chunked_fill(np.empty(5), np.arange(3.0), chunk_size=2)


def test_chunked_fill_chunk_size_errors():
    with pytest.raises(ValueError):
        Chunk.chunked_fill(np.empty(3), np.arange(3.0), chunk_size=0)
    with pytest.raises(TypeError):
        Chunk.chunked_fill(np.empty(3), np.arange(3.0), chunk_size=False)


def test_fill_memmap(tmp_path):
    src = np.arange(100.0)
    fp = temp_memmap((100,), np.float64, temp_dir=str(tmp_path))
    Chunk.fill_memmap(fp, src)
    assert np.array_equal(np.asarray(fp), src)
    path = fp.filename
    del fp
    gc.collect()
    try:
        Path(path).unlink(missing_ok=True)
    except OSError:
        pass


def test_fill_memmap_rejects_list():
    """缺陷面：``fill_memmap`` 不校验 source 类型，list 会在 ``chunked_fill`` 里炸掉。"""
    fp = temp_memmap((4,), np.float64)
    try:
        with pytest.raises(TypeError, match="src must be an ndarray"):
            Chunk.fill_memmap(fp, [0.0, 1.0, 2.0, 3.0])
    finally:
        del fp
        gc.collect()
        cleanup_memmaps()


def test_cleanup_memmaps_with_live_reference_returns_false():
    """回归（本轮修好）：还有外部引用时不能删文件，应**告警 + 返回 False**。

    旧实现在错误分支 ``_ERROR_FILES.add(unexpected_files)`` 里塞 **list**（set 不收），
    于是抛 ``TypeError: unhashable type: 'list'``，把"清理失败"变成崩溃。
    """
    fp = temp_memmap((8,), np.float64)
    try:
        with pytest.warns(UserWarning, match="not be closed"):
            assert cleanup_memmaps() is False
    finally:
        path = fp.filename
        del fp
        gc.collect()
        try:
            Path(path).unlink(missing_ok=True)
        except OSError:
            pass


# --------------------------------------------------------------------------- #
# 6. 端到端不变式 + 契约面
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("chunk_size", [1, 2, 3, 5, 7, 16, 100])
def test_roundtrip_all_sizes(chunk_size):
    S = B.make_signal(1, kind="random")
    join = np.concatenate([np.asarray(c) for c in Chunk.chunk(S, chunk_size=chunk_size)])
    assert np.array_equal(join, S)


def test_roundtrip_two_dims():
    S = np.arange(60.0).reshape(6, 10)
    for axis in (0, 1):
        join = np.concatenate([np.asarray(c) for c in Chunk.chunk(S, chunk_size=4, axis=axis)],
                              axis=axis)
        assert np.array_equal(join, S)


def test_public_surface():
    import Modal_Decomposition.Base as BasePkg

    for name in ("Chunk", "CHUNK_MOD", "CHUNK_MOD_ARGS"):
        assert getattr(BasePkg, name) is getattr(_ChunkMod, name)
    assert set(_ChunkMod.__all__) == {"Chunk", "CHUNK_MOD", "CHUNK_MOD_ARGS"}
    for m in ("chunk", "chunk_size", "chunked_map", "chunked_fill", "fill_memmap"):
        assert callable(getattr(Chunk, m))


def test_no_source_mutation():
    S = B.make_signal(1, kind="random")
    before = S.copy()
    list(Chunk.chunk(S, chunk_size=1000))
    Chunk.chunked_map(lambda c: c * 2, S, chunk_size=1000)
    Chunk.chunked_fill(np.empty_like(S), S, chunk_size=1000)
    assert np.array_equal(S, before)


# --------------------------------------------------------------------------- #
# 导出
# --------------------------------------------------------------------------- #
def build_chunk_size_table() -> list[dict]:
    rows = []
    for size_mib in (1, 8, 9, 50, 100, 200, 500, 600):
        nbytes = size_mib * 1024 * 1024
        n = nbytes // 8
        cs = Chunk.chunk_size(nbytes)
        rows.append({
            "size_mib": size_mib, "nbytes": nbytes, "n_elements": n,
            "chunk_size_returned": cs,
            "returned_ge_n": int(cs >= n),
            "chunks_if_iterated": int(np.ceil(n / cs)),
            "note": "single chunk" if cs >= n else "split",
        })
    return rows


def build_case_table() -> list[dict]:
    rows: list[dict] = []

    def rec(case, fn):
        try:
            fn()
            rows.append({"case": case, "outcome": "ok", "detail": ""})
        except Exception as exc:                    # noqa: BLE001
            rows.append({"case": case, "outcome": type(exc).__name__,
                         "detail": str(exc)[:70]})

    S = np.arange(10.0)
    rec("chunk size=4", lambda: list(Chunk.chunk(S, chunk_size=4)))
    rec("chunk default size", lambda: list(Chunk.chunk(S)))
    rec("chunk size=0", lambda: list(Chunk.chunk(S, chunk_size=0)))
    rec("chunk size=2.0", lambda: list(Chunk.chunk(S, chunk_size=2.0)))
    rec("chunk axis=1 on 1-D", lambda: list(Chunk.chunk(S, chunk_size=2, axis=1)))
    rec("chunk bool dtype", lambda: list(Chunk.chunk(np.array([True]), chunk_size=1)))
    rec("chunk complex dtype", lambda: list(Chunk.chunk(np.zeros(2, dtype=complex))))
    rec("chunk list input", lambda: list(Chunk.chunk([1, 2, 3], chunk_size=2)))
    rec("chunk bad mod", lambda: list(Chunk.chunk(S, chunk_size=2, mod="nope")))
    rec("chunk empty", lambda: list(Chunk.chunk(np.array([]))))
    rec("chunk 0-d", lambda: list(Chunk.chunk(np.array(1.0))))
    rec("chunked_map shape mismatch",
        lambda: Chunk.chunked_map(lambda c: c, S, chunk_size=2, out=np.empty(3)))
    rec("chunked_fill dst list", lambda: Chunk.chunked_fill([0.0], np.arange(1.0)))
    rec("chunk_size float nbytes", lambda: Chunk.chunk_size(3.5))
    rec("chunk_size ratio=1.5", lambda: Chunk.chunk_size(1 << 24, chunk_ratio=1.5))
    return rows


def export() -> None:
    ct = build_chunk_size_table()
    B.write_csv(B.TABLE / "base" / "chunk" / "chunk_size_table.csv", ct)
    cases = build_case_table()
    B.write_csv(B.TABLE / "base" / "chunk" / "chunk_cases.csv", cases)

    B.plt_style()
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    ax = axes[0]
    xs = [r["size_mib"] for r in ct]
    ax.plot(xs, [r["n_elements"] for r in ct], "o-", label="elements in input")
    ax.plot(xs, [r["chunk_size_returned"] for r in ct], "s-", label="chunk_size() returned")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("input size (MiB)")
    ax.set_ylabel("count (elements)")
    ax.set_title("adaptive chunk_size >= n  =>  1 chunk")
    ax.legend(fontsize=7)

    ax = axes[1]
    ax.plot(xs, [r["chunks_if_iterated"] for r in ct], "o-", color="crimson")
    ax.set_xscale("log")
    ax.set_xlabel("input size (MiB)")
    ax.set_ylabel("number of chunks")
    ax.set_yticks([1])
    ax.set_title("actual chunk count without explicit chunk_size")
    ax.set_ylim(0.5, 2.0)

    fig.suptitle("Base.Chunk — adaptive chunk size never splits")
    fig.tight_layout()
    p = B.savefig(fig, B.FIG / "base" / "chunk" / "chunk_size.png")
    plt.close(fig)
    print(f"wrote {B.TABLE / 'base' / 'chunk'}/*.csv")
    print(f"wrote {p}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    B.add_export_flag(ap)
    args = ap.parse_args()
    if args.export:
        export()
    else:
        B.print_table(build_chunk_size_table())
