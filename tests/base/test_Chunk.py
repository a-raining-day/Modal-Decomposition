"""
``Modal_Decomposition.Base.Chunk`` 的模块级契约测试 (一模块一测试)。

覆盖 ``Chunk`` 的全部公开可调用面（``chunk`` / ``chunk_size`` /
``chunked_map`` / ``chunked_fill`` / ``fill_memmap``）与 ``CHUNK_MOD`` /
``CHUNK_MOD_ARGS``：

* ``__all__`` 与实际导出一致（模块级精确匹配，包级转发同一对象）
* 枚举 ``.value`` 为 ``str`` 且可按值往返；Literal 别名成员与枚举一致
* 每个公开方法一条（或多条）烟测：块长 >= 1、切分不丢样本、首尾覆盖完整
* 非法块长（0 / 负数 / 非 int / bool）与非法 ``mod`` / ``axis`` 抛错
* memmap 路径（``mod="exo-memory"``、``chunked_fill`` memmap 目标、
  ``fill_memmap``）各验证一块，并确认临时文件在迭代结束后被清理
* 边界输入（空数组 / 0-d / 2-D 双轴 / 列表输入）不崩

全部数组 <= 128 元素，不分配大数组。
"""

import importlib
import inspect
import os
import typing

import numpy as np
import pytest

import Modal_Decomposition.Base as Base
from Modal_Decomposition.Base import CHUNK_MOD, CHUNK_MOD_ARGS, Chunk
from Modal_Decomposition.Base.Chunk import DEFAULT_FILL_CHUNK_SIZE

# 注意: ``import Modal_Decomposition.Base.Chunk as m`` 拿到的是被
# ``Base/__init__.py`` 覆盖后的**类**(同名遮蔽子模块), 需用 importlib 取模块。
ChunkModule = importlib.import_module("Modal_Decomposition.Base.Chunk")

PUBLIC = ("Chunk", "CHUNK_MOD", "CHUNK_MOD_ARGS")
ENTRY_POINTS = {"chunk", "chunk_size", "chunked_fill", "chunked_map", "fill_memmap"}


def _sig_of(n: int = 128) -> np.ndarray:
    """1-D int 序列; 128 元素足够验证多次分块, 又不会占内存。"""
    return np.arange(n)


def _chunks(S, **kwargs):
    """把 ``Chunk.chunk`` 的生成器物化, 便于一次性断言。"""
    return list(Chunk.chunk(S, **kwargs))


def _partition_offsets(chunks) -> list:
    """由块长还原切分边界 (起始下标), 用于验证首尾覆盖。"""
    offs, acc = [], 0
    for c in chunks:
        offs.append(acc)
        acc += c.shape[0]
    return offs


# --------------------------------------------------------------------------- #
# 1. 导出面: __all__ / 包级转发 / 签名
# --------------------------------------------------------------------------- #
def test_module_all_exactly_public_triplet():
    assert list(ChunkModule.__all__) == list(PUBLIC)
    assert len(ChunkModule.__all__) == len(set(ChunkModule.__all__))
    for name in PUBLIC:
        assert getattr(ChunkModule, name) is not None


def test_base_package_reexports_same_objects():
    for name in PUBLIC:
        assert getattr(Base, name) is getattr(ChunkModule, name)
    # 备注: Base/__init__.py 当前未定义包级 __all__(仅 Chunk.py 定义);
    # 若将来补上, 这三个名字必须在其中。
    if hasattr(Base, "__all__"):
        assert set(PUBLIC) <= set(Base.__all__)


def test_public_surface_is_exactly_five_callables():
    exposed = {n for n in dir(Chunk) if not n.startswith("_")}
    assert exposed == ENTRY_POINTS
    for name in ENTRY_POINTS:
        assert callable(getattr(Chunk, name))


def test_entry_point_signatures_have_documented_defaults():
    def required(sig):
        return [
            n for n, v in sig.parameters.items()
            if v.default is inspect.Parameter.empty
            and v.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        ]

    sig = inspect.signature(Chunk.chunk)
    p = sig.parameters
    assert p["chunk_size"].default is None
    assert p["mod"].default == "memory"
    assert p["axis"].default == 0
    assert p["dtype"].default is None
    assert p["temp_dir"].default is None
    assert required(sig) == ["S"]

    sig = inspect.signature(Chunk.chunk_size)
    p = sig.parameters
    assert p["chunk_ratio"].default == 0.75
    assert p["extras_bytes"].default == 0
    assert required(sig) == ["nbytes"]

    sig = inspect.signature(Chunk.chunked_map)
    p = sig.parameters
    assert p["chunk_size"].default is None
    assert p["out"].default is None
    assert p["dtype"].default is None
    assert required(sig) == ["func", "S"]

    sig = inspect.signature(Chunk.chunked_fill)
    p = sig.parameters
    assert p["chunk_size"].default == DEFAULT_FILL_CHUNK_SIZE
    assert isinstance(p["chunk_size"].default, int) and p["chunk_size"].default >= 1
    assert required(sig) == ["dst", "src"]


# --------------------------------------------------------------------------- #
# 2. 枚举与 Literal 别名
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("member", list(CHUNK_MOD))
def test_enum_value_is_str(member):
    assert isinstance(member.value, str)
    assert member.value  # 非空


def test_enum_members_are_memory_and_exo_memory():
    assert [m.value for m in CHUNK_MOD] == ["memory", "exo-memory"]


@pytest.mark.parametrize("member", list(CHUNK_MOD))
def test_enum_roundtrip_by_value_and_member(member):
    assert CHUNK_MOD(member.value) is member
    assert CHUNK_MOD(member) is member


def test_enum_invalid_value_raises_value_error():
    with pytest.raises(ValueError):
        CHUNK_MOD("disk")


def test_literal_alias_matches_enum_values():
    assert set(typing.get_args(CHUNK_MOD_ARGS)) == {m.value for m in CHUNK_MOD}


# --------------------------------------------------------------------------- #
# 3. Chunk.chunk —— 分块正确性
# --------------------------------------------------------------------------- #
def test_chunk_memory_sizes_and_no_sample_loss():
    S = _sig_of(128)
    chunks = _chunks(S, chunk_size=32)
    assert [c.shape for c in chunks] == [(32,)] * 4
    assert all(c.shape[0] >= 1 for c in chunks)
    assert np.array_equal(np.concatenate(chunks), S)


def test_chunk_first_last_cover_whole_range():
    S = _sig_of(128)
    chunks = _chunks(S, chunk_size=32)
    offs = _partition_offsets(chunks)
    assert offs[0] == 0
    assert chunks[0][0] == S[0]
    assert chunks[-1][-1] == S[-1]
    assert offs[-1] + chunks[-1].shape[0] == S.shape[0]
    # 相邻块首尾相接 (无空洞/重叠): 每块的起始下标 = 前块起始 + 前块长
    for k in range(1, len(offs)):
        assert offs[k] == offs[k - 1] + chunks[k - 1].shape[0]
    for k, c in enumerate(chunks):
        assert np.array_equal(c, S[offs[k]:offs[k] + c.shape[0]])


def test_chunk_ragged_last_chunk():
    S = _sig_of(128)
    chunks = _chunks(S, chunk_size=30)
    assert [c.shape[0] for c in chunks] == [30, 30, 30, 30, 8]
    assert np.array_equal(np.concatenate(chunks), S)


@pytest.mark.parametrize("n", [1, 2, 7])
def test_chunk_extreme_short_input(n):
    """极短输入 + 块长 1: 块长恒为 1, 样本数不丢。"""
    S = _sig_of(n)
    chunks = _chunks(S, chunk_size=1)
    assert [c.shape[0] for c in chunks] == [1] * n
    assert np.array_equal(np.concatenate(chunks), S)


def test_chunk_default_chunk_size_yields_single_chunk():
    S = _sig_of(128)
    chunks = _chunks(S)                      # chunk_size=None -> 自适应默认值
    assert len(chunks) == 1
    assert chunks[0].shape == (128,)
    assert np.array_equal(chunks[0], S)


def test_chunk_str_and_enum_mod_are_equivalent():
    S = _sig_of(128)
    a = _chunks(S, chunk_size=32, mod="memory")
    b = _chunks(S, chunk_size=32, mod=CHUNK_MOD.Memory)
    assert [c.shape for c in a] == [c.shape for c in b]
    for ca, cb in zip(a, b):
        assert np.array_equal(ca, cb)


def test_chunk_axis_0_and_axis_1():
    S = np.arange(12).reshape(6, 2)          # 小 2-D 数组
    assert [c.shape for c in _chunks(S, chunk_size=4, axis=0)] == [(4, 2), (2, 2)]
    assert [c.shape for c in _chunks(S, chunk_size=4, axis=1)] == [(6, 2)]
    assert np.array_equal(np.concatenate(_chunks(S, chunk_size=4, axis=0), axis=0), S)


def test_chunk_list_input_numpy2_behavior():
    """列表输入在 NumPy>=2 下走 ``np.asarray(..., copy=False)``, 无法避免拷贝时
    实现会原样抛 ``ValueError`` (可接受的显式失败); NumPy 1.x 下可直接接受。"""
    S = [1.0, 2.0, 3.0, 4.0]
    try:
        np.asarray(S, copy=False)
    except ValueError:                      # NumPy>=2: 必须拷贝
        with pytest.raises(ValueError):
            _chunks(S, chunk_size=2)
    else:                                   # NumPy 1.x: 零拷贝接受
        chunks = _chunks(S, chunk_size=2)
        assert [c.shape for c in chunks] == [(2,), (2,)]
        assert np.array_equal(np.concatenate(chunks), np.asarray(S))


def test_chunk_0d_yields_single_chunk():
    chunks = _chunks(np.array(5.0))
    assert len(chunks) == 1
    assert chunks[0].shape == ()
    assert float(chunks[0]) == 5.0


def test_chunk_empty_raises_value_error():
    with pytest.raises(ValueError):
        _chunks(np.empty((0,), dtype=np.float64))


def test_chunk_non_numeric_dtype_raises_value_error():
    with pytest.raises(ValueError):
        _chunks(np.array(["a", "b"]))


def test_chunk_bad_mod_raises_value_error():
    with pytest.raises(ValueError):
        _chunks(_sig_of(8), mod="disk")


def test_chunk_bad_axis_raises():
    with pytest.raises(TypeError):
        _chunks(_sig_of(8), axis=True)
    with pytest.raises(ValueError):
        _chunks(_sig_of(8), axis=1)


# --------------------------------------------------------------------------- #
# 4. Chunk.chunk_size —— 自适应块长
# --------------------------------------------------------------------------- #
def test_chunk_size_small_nbytes_uses_floor_and_is_positive_int():
    got = Chunk.chunk_size(128)
    assert got == DEFAULT_FILL_CHUNK_SIZE
    assert isinstance(got, int) and got >= 1
    assert got >= 128                          # 至少覆盖输入, 一块即可


def test_chunk_size_accepts_ndarray_and_warns():
    with pytest.warns(UserWarning):
        got = Chunk.chunk_size(np.arange(16))
    assert got == DEFAULT_FILL_CHUNK_SIZE


def test_chunk_size_large_nbytes_follows_memory_budget():
    """超过默认块容量时退化为 `memory_budget * chunk_ratio`, 仍是正整数。"""
    got = Chunk.chunk_size(10 ** 15)           # 只传数字, 不分配大数组
    assert isinstance(got, int) and got >= 1
    assert got <= 10 ** 15


@pytest.mark.parametrize(
    "kwargs, exc",
    [
        ({"chunk_ratio": 1.5}, ValueError),
        ({"chunk_ratio": 1}, ValueError),      # 要求 float, int 也拒绝
        ({"extras_bytes": -1}, ValueError),
        ({"extras_bytes": 1.0}, TypeError),
    ],
)
def test_chunk_size_rejects_bad_arguments(kwargs, exc):
    with pytest.raises(exc):
        Chunk.chunk_size(128, **kwargs)


def test_chunk_size_rejects_non_int_nbytes():
    with pytest.raises(ValueError):
        Chunk.chunk_size(1.5)                  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# 5. 非法块长 (0 / 负数 / 非 int / bool)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("bad", [0, -1, -5])
def test_chunk_rejects_non_positive_chunk_size(bad):
    with pytest.raises(ValueError):
        _chunks(_sig_of(16), chunk_size=bad)


@pytest.mark.parametrize("bad", [0, -1])
def test_chunked_map_and_fill_reject_non_positive_chunk_size(bad):
    S = _sig_of(16)
    with pytest.raises(ValueError):
        Chunk.chunked_map(lambda c: c, S, chunk_size=bad)
    with pytest.raises(ValueError):
        Chunk.chunked_fill(np.empty_like(S), S, chunk_size=bad)


@pytest.mark.parametrize("bad", [True, False, 8.0, np.int64(8)])
def test_chunk_rejects_non_int_chunk_size(bad):
    with pytest.raises(TypeError):
        _chunks(_sig_of(16), chunk_size=bad)


# --------------------------------------------------------------------------- #
# 6. Chunk.chunked_map
# --------------------------------------------------------------------------- #
def test_chunked_map_matches_elementwise_reference():
    S = _sig_of(128)
    out = Chunk.chunked_map(lambda c: c * 2, S, chunk_size=32)
    assert out.shape == S.shape
    assert np.array_equal(out, S * 2)
    assert out.dtype == S.dtype


def test_chunked_map_default_chunk_size():
    S = _sig_of(128)
    out = Chunk.chunked_map(lambda c: c.astype(np.float32), S)
    assert out.shape == S.shape
    assert out.dtype == np.float32
    assert np.array_equal(out, S.astype(np.float32))


def test_chunked_map_with_preallocated_out():
    S = _sig_of(128)
    buf = np.empty_like(S)
    ret = Chunk.chunked_map(lambda c: c + 1, S, chunk_size=32, out=buf)
    assert ret is buf
    assert np.array_equal(buf, S + 1)


def test_chunked_map_out_shape_mismatch_raises():
    S = _sig_of(16)
    with pytest.raises(ValueError):
        Chunk.chunked_map(lambda c: c, S, out=np.empty(8))


def test_chunked_map_empty_input_returns_empty():
    S = np.empty((0,), dtype=np.float64)
    out = Chunk.chunked_map(lambda c: c, S)
    assert out.shape == (0,)


def test_chunked_map_0d_raises():
    with pytest.raises(ValueError):
        Chunk.chunked_map(lambda c: c, np.array(1.0))


# --------------------------------------------------------------------------- #
# 7. Chunk.chunked_fill
# --------------------------------------------------------------------------- #
def test_chunked_fill_returns_dst_and_copies_all():
    S = _sig_of(128)
    dst = np.empty_like(S)
    ret = Chunk.chunked_fill(dst, S, chunk_size=64)
    assert ret is dst
    assert np.array_equal(dst, S)


def test_chunked_fill_rejects_non_array_arguments():
    S = _sig_of(16)
    with pytest.raises(TypeError):
        Chunk.chunked_fill([0] * 16, S)        # type: ignore[arg-type]
    with pytest.raises(TypeError):
        Chunk.chunked_fill(np.empty_like(S), list(range(16)))  # type: ignore[arg-type]


def test_chunked_fill_into_memmap_persists(tmp_path):
    p = tmp_path / "dst.dat"
    S = _sig_of(128)
    fp = np.memmap(p, dtype=S.dtype, mode="w+", shape=S.shape)
    try:
        Chunk.chunked_fill(fp, S, chunk_size=32)
        assert fp.flush() is None
        assert np.array_equal(np.asarray(fp), S)
    finally:
        fp._mmap.close()
        del fp
    # 落盘内容可被重新映射读取
    again = np.memmap(p, dtype=S.dtype, mode="r", shape=S.shape)
    try:
        assert np.array_equal(np.asarray(again), S)
    finally:
        again._mmap.close()
        del again


# --------------------------------------------------------------------------- #
# 8. Chunk.fill_memmap
# --------------------------------------------------------------------------- #
def test_fill_memmap_streams_source(tmp_path):
    p = tmp_path / "fp.dat"
    S = _sig_of(128)
    fp = np.memmap(p, dtype=S.dtype, mode="w+", shape=S.shape)
    try:
        assert Chunk.fill_memmap(fp, S) is None
        assert np.array_equal(np.asarray(fp), S)
    finally:
        fp._mmap.close()
        del fp


# --------------------------------------------------------------------------- #
# 9. memmap 模式 (exo-memory): 一块 + 临时文件生命周期
# --------------------------------------------------------------------------- #
def test_chunk_exo_memory_roundtrip_and_cleanup(tmp_path):
    S = _sig_of(128)
    got = []
    for c in Chunk.chunk(S, chunk_size=32, mod="exo-memory", temp_dir=str(tmp_path)):
        assert isinstance(c, np.memmap)
        assert c.shape[0] >= 1
        got.append(np.array(c))                # 仅在迭代步内消费
    assert [c.shape for c in got] == [(32,)] * 4
    assert np.array_equal(np.concatenate(got), S)
    # 迭代结束后临时 .dat 文件应被清理
    assert os.listdir(tmp_path) == []


def test_chunk_exo_memory_creates_missing_temp_dir_and_applies_dtype(tmp_path):
    S = _sig_of(16)
    temp_dir = tmp_path / "nested" / "chunks"     # 不存在 -> 由实现创建
    got = [np.array(c) for c in Chunk.chunk(
        S, chunk_size=8, mod="exo-memory", dtype=np.float32, temp_dir=str(temp_dir)
    )]
    assert temp_dir.is_dir()
    assert [c.dtype for c in got] == [np.dtype(np.float32)] * 2
    assert np.array_equal(np.concatenate(got), S.astype(np.float32))
    assert os.listdir(temp_dir) == []


def test_chunk_exo_memory_enum_member_equivalent(tmp_path):
    S = _sig_of(16)
    got = [np.array(c) for c in Chunk.chunk(
        S, chunk_size=8, mod=CHUNK_MOD.ExoMemory, temp_dir=str(tmp_path)
    )]
    assert np.array_equal(np.concatenate(got), S)
