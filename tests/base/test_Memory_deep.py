"""``Base.Memory`` 深度测试：预算公式、快照缓存、临时 memmap 生命周期。

内存预算决定 `should_use_memmap` 的阈值，而后者决定"大数组走内存还是磁盘"，是
分解法能否在 4 GB 机器上跑 600 MB 输入的关键。所以这里不仅测类型，还验证公式
与"预算 ≤ 物理可用"这类不变量。

临时 memmap 的用例都落在 pytest 的 ``tmp_path`` 里，并在结束时 ``drop_memmap``/
``cleanup_memmaps``，避免污染系统临时目录（Windows 上忘了删会被后续用例看到）。
"""

from __future__ import annotations

import importlib
import sys
import time
from pathlib import Path

import numpy as np
import pytest

_TESTS = Path(__file__).resolve().parents[1]
if str(_TESTS) not in sys.path:
    sys.path.insert(0, str(_TESTS))

import _bench as B  # noqa: E402

B.bootstrap_src()

_Mem = importlib.import_module("Modal_Decomposition.Base.Memory")
MemoryPolicy = _Mem.MemoryPolicy


@pytest.fixture(autouse=True)
def _restore_policy():
    saved = MemoryPolicy.memory_reserve_ratio
    try:
        yield
    finally:
        MemoryPolicy.memory_reserve_ratio = saved
        _Mem.cleanup_memmaps()
        _Mem.cleanup_memmaps()


# --------------------------------------------------------------------------- #
# 1. 快照与预算
# --------------------------------------------------------------------------- #
def test_snapshot_keys():
    snap = _Mem.get_memory_snapshot(force=True)
    assert set(snap) == {"total", "available", "swap_total", "swap_free",
                         "commit_available", "process_rss", "process_private"}
    assert snap["total"] > 0 and snap["available"] > 0
    assert snap["commit_available"] == snap["available"] + snap["swap_free"]


def test_snapshot_ttl_and_force():
    a = _Mem.get_memory_snapshot(force=True)
    b = _Mem.get_memory_snapshot()               # 命中 TTL 缓存
    assert a is b or a == b
    c = _Mem.get_memory_snapshot(force=True)
    assert c is not b


def test_getters():
    assert _Mem.get_available_memory(force=True) > 0
    assert _Mem.get_commit_available(force=True) >= _Mem.get_available_memory(force=True)
    pm = _Mem.get_process_memory(force=True)
    assert set(pm) == {"rss", "private"} and pm["rss"] > 0


def test_budget_formula(monkeypatch):
    """预算 = min(物理可用, 可提交余量) × (1 - reserve) 再扣 extras。

    用**固定的假快照**做断言，避免"两次取快照之间可用内存变了"造成的抖动
    （真机上差几百 MB 很常见，曾经因此偶发失败）。
    """
    MemoryPolicy.memory_reserve_ratio = 0.25
    fake = {
        "total": 16 << 30, "available": 8 << 30,
        "swap_total": 0, "swap_free": 0,
        "commit_available": 8 << 30,
        "process_rss": 0, "process_private": 0,
    }
    monkeypatch.setattr(_Mem, "get_memory_snapshot", lambda force=False: dict(fake))

    mb = _Mem.memory_budget(force=True)
    physical = fake["available"] * 0.75
    commit = fake["commit_available"] * 0.75 * MemoryPolicy.commit_ratio_limit
    expect = int(min(physical, commit))
    assert mb["budget"] == expect
    assert mb["limited_by"] in ("physical", "commit")
    assert mb["ok"] is True
    assert mb["reserve"] == int(min(fake["available"], fake["commit_available"]) * 0.25)

    # extras 直接从预算里扣
    mb2 = _Mem.memory_budget(extras_bytes=1 << 20, force=True)
    assert mb2["budget"] == expect - (1 << 20)


def test_budget_live_sanity():
    """真实快照下：预算 > 0，且不超过物理可用。"""
    mb = _Mem.memory_budget(force=True)
    snap = _Mem.get_memory_snapshot(force=True)
    assert mb["budget"] > 0
    assert mb["budget"] <= snap["available"]
    assert mb["physical"] <= snap["available"]


def test_budget_extras_reduces():
    a = _Mem.memory_budget(force=True)
    b = _Mem.memory_budget(extras_bytes=1 << 30, force=True)
    assert b["budget"] <= a["budget"]


def test_policy_set_and_validate():
    MemoryPolicy.set_memory_reserve(0.1)
    assert MemoryPolicy.memory_reserve_ratio == 0.1
    assert MemoryPolicy.policy()["memory_reserve_ratio"] == 0.1
    for bad in (-0.1, 0.95, "0.5", None):
        with pytest.raises(ValueError):
            MemoryPolicy.set_memory_reserve(bad)
    assert MemoryPolicy.set_memory_reserve(0) == 0.0
    assert MemoryPolicy.set_memory_reserve(0.9) == 0.9


def test_should_use_memmap_threshold():
    mb = _Mem.memory_budget(force=True)
    small = 1 << 20
    big = int(mb["budget"] * 2)
    assert _Mem.should_use_memmap(small) is False
    assert _Mem.should_use_memmap(big) is True
    assert _Mem.should_use_memmap(big - 1, extra=1) is True


def test_should_use_memmap_accepts_array():
    arr = np.zeros(16)
    assert _Mem.should_use_memmap(arr) is False
    with pytest.raises(ValueError):
        _Mem.should_use_memmap(3.5)
    with pytest.raises(ValueError):
        _Mem.should_use_memmap(1 << 20, extra=1.0)


# --------------------------------------------------------------------------- #
# 2. 临时 memmap 生命周期
# --------------------------------------------------------------------------- #
def test_temp_memmap_creates_registered_file(tmp_path):
    fp = _Mem.temp_memmap((16,), np.float32, temp_dir=str(tmp_path))
    try:
        assert isinstance(fp, np.memmap) and fp.shape == (16,) and fp.dtype == np.float32
        assert fp.filename in _Mem._MM_FILES
        assert Path(fp.filename).exists()
        assert Path(fp.filename).parent == tmp_path
    finally:
        _Mem.drop_memmap(fp, force=True)


def test_temp_memmap_persist_not_registered(tmp_path):
    fp = _Mem.temp_memmap((8,), float, temp_dir=str(tmp_path), persist=True)
    assert fp.filename not in _Mem._MM_FILES
    path = Path(fp.filename)
    # persist=True 的条目不在注册表里，drop_memmap 会拒绝（它只处理注册过的）
    with pytest.raises(ValueError, match="not registered"):
        _Mem.drop_memmap(fp)
    fp._mmap.close()
    del fp
    path.unlink(missing_ok=True)


def test_temp_memmap_persist_type_check(tmp_path):
    with pytest.raises(TypeError, match="persist"):
        _Mem.temp_memmap((8,), float, temp_dir=str(tmp_path), persist=1)


def test_drop_memmap(tmp_path):
    fp = _Mem.temp_memmap((8,), float, temp_dir=str(tmp_path))
    path = fp.filename
    assert _Mem.drop_memmap(fp, force=True) is True
    assert not Path(path).exists()
    assert path not in _Mem._MM_FILES


def test_drop_memmap_errors(tmp_path):
    with pytest.raises(TypeError):
        _Mem.drop_memmap(np.zeros(3))
    fp = _Mem.temp_memmap((8,), float, temp_dir=str(tmp_path), persist=True)
    try:
        with pytest.raises(ValueError, match="not registered"):
            _Mem.drop_memmap(fp)
    finally:
        del fp


def test_cleanup_memmaps_removes_all(tmp_path):
    paths = []
    fps = []
    for _ in range(3):
        fp = _Mem.temp_memmap((8,), float, temp_dir=str(tmp_path))
        fps.append(fp)
        paths.append(fp.filename)
    del fps, fp                    # 必须把最后一个引用也删掉，否则 Windows 删不掉
    assert _Mem.cleanup_memmaps() is True
    assert all(not Path(p).exists() for p in paths)
    assert _Mem._MM_FILES == {}


def test_cleanup_memmaps_validation():
    for bad in (True, 1.0, "5"):
        with pytest.raises(TypeError):
            _Mem.cleanup_memmaps(try_times=bad)
    with pytest.raises(TypeError):
        _Mem.cleanup_memmaps(force=1)


def test_cleanup_is_idempotent():
    assert _Mem.cleanup_memmaps() is True
    assert _Mem.cleanup_memmaps() is True


def test_temp_memmap_bad_dtype_returns_none_with_warning(tmp_path):
    """特征化：坏 dtype 不抛异常，而是 **warn + 返回 None**（`except BaseException` 路径）。"""
    with pytest.warns(UserWarning, match="Error happen when constructing"):
        out = _Mem.temp_memmap((4,), "not-a-dtype", temp_dir=str(tmp_path))
    assert out is None


def test_memmap_write_read_roundtrip(tmp_path):
    fp = _Mem.temp_memmap((5, 4), np.float64, temp_dir=str(tmp_path))
    try:
        data = np.arange(20.0).reshape(5, 4)
        fp[:] = data
        fp.flush()
        again = np.memmap(fp.filename, dtype=np.float64, mode="r", shape=(5, 4))
        assert np.array_equal(np.asarray(again), data)
        again._mmap.close()
        del again
    finally:
        _Mem.drop_memmap(fp, force=True)


def test_public_surface():
    import Modal_Decomposition.Base as BasePkg

    for name in _Mem.__all__:
        assert hasattr(BasePkg, name) or hasattr(_Mem, name)
    assert "temp_memmap" in _Mem.__all__ and "cleanup_memmaps" in _Mem.__all__


def test_budget_snapshot_table():
    rows = []
    for reserve in (0.1, 0.25, 0.5):
        MemoryPolicy.memory_reserve_ratio = reserve
        mb = _Mem.memory_budget(force=True)
        rows.append({"reserve_ratio": reserve, "budget_mib": round(mb["budget"] / B.MiB, 1),
                     "limited_by": mb["limited_by"],
                     "ok": int(mb["ok"])})
    B.write_csv(B.TABLE / "base" / "memory" / "budget_table.csv", rows)
    B.print_table(rows)


if __name__ == "__main__":
    test_budget_snapshot_table()
