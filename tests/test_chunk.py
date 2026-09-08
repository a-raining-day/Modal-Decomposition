"""
tests for ``Modal_Decomposition.Utils.Chunk``.

覆盖:
1. 分块正确性 —— 第 0 维精确切分 / 余块 / 零拷贝视图 / 多维 / memmap 输入;
2. Chunk 类门面 —— memory / exo-memory / None 三模式与旧参数名兼容;
3. chunked_map / chunked_fill —— 与全量操作等价、预分配输出、memmap 目标;
4. adapt_chunk_size —— 64MB 门槛、绝对/比率策略、预算耗尽、psutil 缺失;
5. 复用烟测 —— Monotonicity 分块路径与 Check 流式填充走同一实现;
6. 性能烟测 (宽松上限) —— 为 docs/ChunkReport.md 提供实测数字。
"""

import os
import time

import numpy as np
import pytest

from src.Modal_Decomposition.Utils.Chunk import (
    ADAPT_MIN_BYTES,
    DEFAULT_FILL_CHUNK_ELEMS,
    MIN_CHUNK_ELEMS,
    Chunk,
    adapt_chunk_size,
    chunked_fill,
    chunked_map,
    chunks,
    default_chunk_size,
    exo_chunks,
    iter_chunks,
)


@pytest.fixture(autouse=True)
def _restore_memory_policy():
    """测试后恢复默认内存策略 (比率 0.6)。"""
    from src.Modal_Decomposition.Utils.Memory import set_memmap_ratio

    yield
    set_memmap_ratio(0.6)


# --------------------------------------------------------------------------- #
# 1. 分块正确性
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("n", "c", "n_chunks"),
    [(0, 4, 0), (1, 4, 1), (10, 4, 3), (10, 1, 10), (10, 10, 1), (10, 100, 1)],
)
def test_iter_chunks_partition(n, c, n_chunks):
    S = np.arange(n, dtype=np.int64)
    out = list(iter_chunks(S, c))
    assert len(out) == n_chunks
    assert sum(ch.size for ch in out) == n
    if n:
        assert np.concatenate(out).tolist() == list(range(n))


def test_iter_chunks_last_chunk_shorter():
    S = np.arange(10)
    out = list(iter_chunks(S, 4))
    assert [ch.size for ch in out] == [4, 4, 2]


def test_iter_chunks_zero_copy_views():
    """块是视图: 修改块即修改原数组 (无拷贝)。"""
    S = np.arange(10)
    first = next(iter_chunks(S, 4))
    first[:] = -1
    assert S[0] == -1
    assert S[3] == -1


def test_iter_chunks_multidim_splits_axis0():
    S = np.arange(18).reshape(6, 3)
    shapes = [ch.shape for ch in iter_chunks(S, 4)]
    assert shapes == [(4, 3), (2, 3)]
    assert np.array_equal(np.concatenate(list(iter_chunks(S, 4))), S)


def test_iter_chunks_memmap_input(tmp_path):
    """memmap 输入可分块且块仍共享底层磁盘映射 (逐块拷贝, 不跨迭代保留)。"""
    path = str(tmp_path / "src.dat")
    mm = np.memmap(path, dtype=np.float64, mode="w+", shape=(100,))
    mm[:] = np.arange(100)
    out = [ch.copy() for ch in iter_chunks(mm, 30)]  # 拷贝避免映射未关闭
    assert [ch.size for ch in out] == [30, 30, 30, 10]
    assert np.array_equal(np.concatenate(out), np.arange(100.0))
    del out, mm


def test_iter_chunks_rejects_bad_inputs():
    with pytest.raises(ValueError):
        list(iter_chunks(np.array(3.0), 4))  # 0-d
    for bad in (0, -1, 1.5, True, "4", None):
        with pytest.raises((TypeError, ValueError)):
            list(iter_chunks(np.arange(4), bad))


def test_chunks_alias():
    S = np.arange(5)
    assert [c.tolist() for c in chunks(S, 2)] == [c.tolist() for c in iter_chunks(S, 2)]


# --------------------------------------------------------------------------- #
# 2. Chunk 类门面
# --------------------------------------------------------------------------- #
def test_chunk_memory_mode_matches_iter_chunks():
    S = np.arange(10)
    got = [c.tolist() for c in Chunk(4).chunk(S)]
    want = [c.tolist() for c in iter_chunks(S, 4)]
    assert got == want


def test_chunk_none_mode_single_chunk():
    S = np.arange(10)
    out = list(Chunk(4, mod="None").chunk(S))
    assert len(out) == 1
    assert out[0] is S


def test_chunk_invalid_mod():
    with pytest.raises(ValueError, match="mod"):
        Chunk(4, mod="bogus")


def test_chunk_call_alias_and_repr():
    c = Chunk(8, mod="memory")
    S = np.arange(9)
    assert [x.tolist() for x in c(S)] == [x.tolist() for x in c.chunk(S)]
    assert "chunk_size=8" in repr(c)


def test_chunk_exo_memory_yields_memmap_and_cleans_up(tmp_path):
    S = np.arange(1, 11, dtype=np.int32)
    parts = []
    for chunk in Chunk(4, mod="exo-memory", memmap_pth=str(tmp_path)).chunk(S):
        assert isinstance(chunk, np.memmap)
        parts.append(np.asarray(chunk).copy())  # 拷贝: 下一块时文件会被删
    assert [p.tolist() for p in parts] == [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10]]
    assert os.listdir(tmp_path) == []  # 用后即删, 无残留


def test_chunk_exo_memory_dtype_override(tmp_path):
    S = np.arange(8)
    # 契约: 块仅在当前迭代步内有效 —— 边消费边拷贝 (勿跨迭代保留引用)
    parts = [
        np.asarray(p).copy()
        for p in Chunk(3, mod="exo-memory", memmap_pth=str(tmp_path),
                       memmap_type=np.float32).chunk(S)
    ]
    assert all(p.dtype == np.float32 for p in parts)
    assert np.concatenate(parts).tolist() == list(range(8))


def test_exo_chunks_function(tmp_path):
    S = np.arange(7)
    parts = [np.asarray(p).copy() for p in exo_chunks(S, 3, temp_dir=str(tmp_path))]
    assert [p.tolist() for p in parts] == [[0, 1, 2], [3, 4, 5], [6]]
    assert os.listdir(tmp_path) == []


def test_exo_chunks_empty_and_bad():
    assert list(exo_chunks(np.array([]), 4, temp_dir=".")) == []
    with pytest.raises(ValueError):
        list(exo_chunks(np.array(1.0), 4))


# --------------------------------------------------------------------------- #
# 3. chunked_map / chunked_fill
# --------------------------------------------------------------------------- #
def test_chunked_map_matches_whole_array():
    rng = np.random.default_rng(0)
    S = rng.standard_normal(1000)
    got = chunked_map(np.square, S, chunk_size=128)
    assert np.array_equal(got, np.square(S))
    assert got.dtype == np.square(S).dtype


def test_chunked_map_multidim_and_large_chunk():
    S = np.arange(50 * 10).reshape(50, 10)
    assert np.array_equal(chunked_map(lambda c: c * 2, S, chunk_size=7), S * 2)
    assert np.array_equal(chunked_map(lambda c: c + 1, S, chunk_size=100), S + 1)


def test_chunked_map_preallocated_out():
    rng = np.random.default_rng(1)
    S = rng.standard_normal(64)
    out = np.empty_like(S)
    assert chunked_map(np.abs, S, chunk_size=10, out=out) is out
    assert np.array_equal(out, np.abs(S))


def test_chunked_map_preallocated_memmap_out(tmp_path):
    S = np.arange(100, dtype=np.float64)
    path = str(tmp_path / "out.dat")
    out = np.memmap(path, dtype=np.float64, mode="w+", shape=S.shape)
    chunked_map(lambda c: c * 3, S, chunk_size=8, out=out)
    del out  # 关闭映射
    back = np.memmap(path, dtype=np.float64, mode="r", shape=S.shape)
    assert np.array_equal(back, S * 3)
    del back


def test_chunked_map_empty_and_dtype():
    empty = chunked_map(np.square, np.array([]), chunk_size=8, dtype=np.float32)
    assert empty.shape == (0,) and empty.dtype == np.float32


def test_chunked_map_out_shape_mismatch():
    with pytest.raises(ValueError, match="shape"):
        chunked_map(np.square, np.arange(8), chunk_size=4, out=np.empty(4))


def test_chunked_fill_ndarray_and_dtype_conversion():
    src = np.arange(100, dtype=np.float32)
    dst = np.empty(100, dtype=np.float64)
    assert chunked_fill(dst, src, chunk_size=7) is dst
    assert np.array_equal(dst, src.astype(np.float64))


def test_chunked_fill_memmap(tmp_path):
    path = str(tmp_path / "fill.dat")
    dst = np.memmap(path, dtype=np.float64, mode="w+", shape=(1000,))
    src = np.linspace(0, 1, 1000)
    chunked_fill(dst, src, chunk_size=64)
    del dst
    back = np.memmap(path, dtype=np.float64, mode="r", shape=(1000,))
    assert np.allclose(back, src)
    del back


# --------------------------------------------------------------------------- #
# 4. adapt_chunk_size (全局内存策略联动)
# --------------------------------------------------------------------------- #
def test_adapt_below_min_bytes_returns_unchanged():
    assert adapt_chunk_size(10_000_000, ADAPT_MIN_BYTES - 1) == 10_000_000


def test_adapt_absolute_strategy_shrinks_to_budget():
    from src.Modal_Decomposition.Utils.Memory import set_absolute_limit

    set_absolute_limit(150 * 1024 * 1024)  # 150MB
    nbytes = 100 * 1024 * 1024             # 100MB (>= 64MB 门槛)
    budget = 50 * 1024 * 1024
    expected = max(MIN_CHUNK_ELEMS, budget // 3)  # 50MB // 3 = 17476266
    assert expected == 17_476_266
    assert adapt_chunk_size(100_000_000, nbytes) == expected


def test_adapt_absolute_budget_exhausted_keeps_chunk():
    from src.Modal_Decomposition.Utils.Memory import set_absolute_limit

    set_absolute_limit(100 * 1024 * 1024)   # 100MB 上限
    assert adapt_chunk_size(1_000_000, 120 * 1024 * 1024) == 1_000_000


def test_adapt_ratio_strategy(monkeypatch):
    import src.Modal_Decomposition.Utils.Chunk as chunk_mod
    from src.Modal_Decomposition.Utils.Memory import set_memmap_ratio

    set_memmap_ratio(0.5)
    available = 1024 * 1024 * 1024  # 1GB
    monkeypatch.setattr(chunk_mod, "get_available_memory", lambda force=False: available)

    nbytes = 400 * 1024 * 1024  # 400MB
    budget = int(0.5 * available) - nbytes
    expected = max(MIN_CHUNK_ELEMS, budget // 3)  # 117440512 // 3 = 39146837
    assert adapt_chunk_size(100_000_000, nbytes) == expected


def test_adapt_psutil_unavailable_keeps_chunk(monkeypatch):
    import src.Modal_Decomposition.Utils.Chunk as chunk_mod
    from src.Modal_Decomposition.Utils.Memory import set_memmap_ratio

    set_memmap_ratio(0.6)
    monkeypatch.setattr(chunk_mod, "get_available_memory", lambda force=False: None)
    assert adapt_chunk_size(1_000_000, ADAPT_MIN_BYTES) == 1_000_000


# --------------------------------------------------------------------------- #
# 5. 复用烟测: Monotonicity / Check 走同一分块实现
# --------------------------------------------------------------------------- #
def test_monotonicity_chunked_path_smoke():
    from src.Modal_Decomposition.Utils.Monotonicity import is_monotonic

    arr = np.arange(10_000.0)
    assert is_monotonic(arr, chunk_size=100) is True
    arr[5000] = -1.0
    assert is_monotonic(arr, chunk_size=100) is False


def test_check_fill_memmap_delegates(monkeypatch, tmp_path):
    import src.Modal_Decomposition.Utils.Check as check_mod

    monkeypatch.setattr(check_mod, "_FILL_CHUNK_ELEMS", 5)  # 小块强制多轮
    path = str(tmp_path / "fill.dat")
    fp = np.memmap(path, dtype=np.float64, mode="w+", shape=(17,))
    src = np.arange(17, dtype=np.float64)
    check_mod._fill_memmap(fp, src)
    del fp
    back = np.memmap(path, dtype=np.float64, mode="r", shape=(17,))
    assert np.array_equal(back, src)
    del back


# --------------------------------------------------------------------------- #
# 6. 性能烟测 (宽松上限; 数字供 docs/ChunkReport.md)
# --------------------------------------------------------------------------- #
def test_performance_smoke():
    n, c = 4_000_000, 1_000_000
    S = np.linspace(0, 1, n)

    t0 = time.perf_counter()
    total = sum(float(ch.sum()) for ch in iter_chunks(S, c))
    t_iter = time.perf_counter() - t0
    assert np.isclose(total, S.sum())

    t0 = time.perf_counter()
    out = chunked_map(np.square, S, chunk_size=c)
    t_map = time.perf_counter() - t0
    assert np.array_equal(out, np.square(S))

    print(f"[chunk-perf] iter_chunks(4M, c=1M)   = {t_iter * 1e3:7.1f} ms")
    print(f"[chunk-perf] chunked_map(4M, c=1M)   = {t_map * 1e3:7.1f} ms")
    assert t_iter < 5.0 and t_map < 5.0


# --------------------------------------------------------------------------- #
# 7. Cache 注册 (Utils.get_chunk)
# --------------------------------------------------------------------------- #
def test_get_chunk_registered_in_cache():
    from src.Modal_Decomposition import Utils as U
    from src.Modal_Decomposition.Base.Cache import cache

    mod = U.get_chunk()
    assert mod is cache.get("Modal_Decomposition.Utils.Chunk")
    assert callable(mod.iter_chunks) and callable(mod.chunked_map)


# --------------------------------------------------------------------------- #
# 8. default_chunk_size (流式处理的粒度选择)
# --------------------------------------------------------------------------- #
def test_default_chunk_size_baseline_and_shrink():
    from src.Modal_Decomposition.Utils.Memory import set_absolute_limit

    # 小输入 (<64MB): 不咨询策略, base 不变
    assert default_chunk_size(1024) == DEFAULT_FILL_CHUNK_ELEMS
    assert default_chunk_size(1024, base=12345) == 12345

    # 绝对策略: 预算足够时默认 base 不动; base 超预算时收缩 (同 adapt 规则)
    set_absolute_limit(150 * 1024 * 1024)
    nbytes = 100 * 1024 * 1024
    budget = 50 * 1024 * 1024
    cap = max(MIN_CHUNK_ELEMS, budget // 3)
    assert default_chunk_size(nbytes) == DEFAULT_FILL_CHUNK_ELEMS  # 8.38M < cap
    assert default_chunk_size(nbytes, base=100_000_000) == cap      # 收缩到 cap


def test_default_chunk_size_rejects_bad_base():
    with pytest.raises((TypeError, ValueError)):
        default_chunk_size(1024, base=0)
    with pytest.raises((TypeError, ValueError)):
        default_chunk_size(1024, base=1.5)
