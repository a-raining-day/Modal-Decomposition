"""
``Modal_Decomposition.Base.Memory`` 单模块测试。

覆盖: ``__all__`` 与导出一致性、``should_use_memmap`` 的尺寸判定、内存
快照 / 预算 / 余量的返回契约、``MemoryPolicy`` 的比例校验与恢复、
``temp_memmap`` / ``drop_memmap`` / ``cleanup_memmaps`` 的生命周期与
临时文件清理。

约定: 不分配大数组 (最大 ``(1024,)``), 任何改全局状态的用例都在
``try/finally`` 里恢复, 每个用例结束后由 autouse fixture 兜底清理
``_MM_FILES`` 与 ``Snapshot`` 缓存。
"""

import os
import tempfile

import numpy as np
import pytest

import Modal_Decomposition.Base as Base
from Modal_Decomposition.Base import (
    MemoryPolicy,
    cleanup_memmaps,
    drop_memmap,
    get_available_memory,
    get_commit_available,
    get_memory_snapshot,
    get_process_memory,
    memory_budget,
    should_use_memmap,
    temp_memmap,
)
from Modal_Decomposition.Base import Memory as MemoryModule

# MemoryPolicy 的出厂默认值: 用例结束时必须回到这个值
DEFAULT_RESERVE = 0.25


# --------------------------------------------------------------------------- #
# 隔离: 每个用例前重置全局状态, 用例后关闭并删除所有临时 memmap
# --------------------------------------------------------------------------- #
@pytest.fixture(autouse=True)
def _restore_memory_globals():
    MemoryPolicy.memory_reserve_ratio = DEFAULT_RESERVE
    MemoryModule.Snapshot.snapshot = None
    MemoryModule.Snapshot.now = 0.0
    yield
    MemoryPolicy.memory_reserve_ratio = DEFAULT_RESERVE
    MemoryModule.Snapshot.snapshot = None
    MemoryModule.Snapshot.now = 0.0
    cleanup_memmaps(force=True)


# --------------------------------------------------------------------------- #
# __all__ / 导出
# --------------------------------------------------------------------------- #
def test_module_all_matches_public_api():
    """``__all__`` 无重复、全部存在, 且与模块内定义的公开对象一致。"""
    all_ = MemoryModule.__all__
    assert isinstance(all_, list) and all_
    assert all(isinstance(n, str) for n in all_)
    assert len(set(all_)) == len(all_), "__all__ 不应有重复项"

    # 每一项都真实存在且不是私有名
    for name in all_:
        assert hasattr(MemoryModule, name), f"__all__ 中的 {name!r} 不存在"
        assert not name.startswith("_")

    # 模块内自定义的公开类/函数都必须在 __all__ 里
    # (Snapshot 是快照缓存容器, 刻意不导出)
    internal_public = {
        n
        for n in dir(MemoryModule)
        if not n.startswith("_")
        and getattr(getattr(MemoryModule, n), "__module__", "") == MemoryModule.__name__
    }
    assert internal_public - set(all_) == {"Snapshot"}, (
        "除 Snapshot 外的模块内公开对象都应出现在 __all__ 中: "
        f"{sorted(internal_public - set(all_))}"
    )

    # Base 包把 Memory 的全部公开 API 原样再导出
    for name in all_:
        assert getattr(Base, name) is getattr(MemoryModule, name), f"Base 未正确再导出 {name!r}"


# --------------------------------------------------------------------------- #
# should_use_memmap
# --------------------------------------------------------------------------- #
def test_should_use_memmap_monotone_in_size():
    """极小尺寸走内存, 天文尺寸必然走 memmap。"""
    assert should_use_memmap(0) is False
    assert should_use_memmap(10 ** 15) is True

    # 单调性: 加上 extra 只会更容易走 memmap
    assert should_use_memmap(10 ** 15, extra=10 ** 15) is True
    assert should_use_memmap(0, extra=10 ** 15) is True


def test_should_use_memmap_mid_size_returns_bool():
    """中间尺寸只做宽松断言: 不崩且返回真正的 bool。"""
    decision = should_use_memmap(10 ** 6)
    assert isinstance(decision, bool)
    assert decision in (True, False)


def test_should_use_memmap_rejects_non_int():
    for bad in (1.0, "a", None):
        with pytest.raises(ValueError):
            should_use_memmap(bad)


@pytest.mark.parametrize("factory", [lambda: np.zeros(10), lambda: temp_memmap((64,))])
def test_should_use_memmap_accepts_arrays(factory):
    """传入 ndarray / memmap 时按 ``.nbytes`` 处理, 返回 bool。"""
    obj = factory()
    if obj is None:  # 平台不允许 memmap
        pytest.skip("temp_memmap returned None on this platform")
    assert should_use_memmap(obj) is False  # 640 B 远小于预算


# --------------------------------------------------------------------------- #
# 内存快照 / 预算 / 余量
# --------------------------------------------------------------------------- #
def test_memory_snapshot_non_empty_dict():
    snap = get_memory_snapshot(force=True)
    assert isinstance(snap, dict) and snap

    expected = {
        "total",
        "available",
        "swap_total",
        "swap_free",
        "commit_available",
        "process_rss",
        "process_private",
    }
    assert expected <= set(snap)
    for key in expected:
        assert isinstance(snap[key], int) and snap[key] >= 0, f"{key} 应为非负 int"

    assert snap["total"] >= snap["available"]
    assert snap["swap_total"] >= snap["swap_free"]
    # 可提交余量 = 物理可用 + 页面文件空闲
    assert snap["commit_available"] == snap["available"] + snap["swap_free"]


def test_memory_budget_non_empty_dict():
    extras = 123456
    budget = memory_budget(extras_bytes=extras, force=True)
    assert isinstance(budget, dict) and budget
    assert {"budget", "physical", "commit", "reserve", "limited_by", "ok"} <= set(budget)

    assert budget["limited_by"] in {"physical", "commit", "psutil-missing"}
    assert budget["ok"] is (budget["budget"] > 0)
    for key in ("budget", "physical", "commit", "reserve"):
        assert isinstance(budget[key], int)
    assert budget["physical"] >= 0
    assert budget["commit"] >= 0
    assert budget["reserve"] >= 0

    # budget = 物理/可提交两者较小值 - extras (取整误差 <= 1)
    expected = int(min(budget["physical"], budget["commit"])) - extras
    assert abs(budget["budget"] - expected) <= 1
    # physical 由可用内存扣掉系统余量得到
    assert budget["physical"] <= get_memory_snapshot(force=True)["available"]


def test_get_available_and_commit_memory_non_negative():
    available = get_available_memory(force=True)
    commit = get_commit_available(force=True)
    assert isinstance(available, int) and available >= 0
    assert isinstance(commit, int) and commit >= 0
    assert commit >= available  # 可提交余量包含物理可用


def test_get_process_memory_returns_non_negative_numbers():
    """注: 本进程口径返回 dict ``{"rss", "private"}`` 而非标量。"""
    proc = get_process_memory(force=True)
    assert isinstance(proc, dict)
    assert {"rss", "private"} <= set(proc)
    assert isinstance(proc["rss"], int) and proc["rss"] >= 0
    assert isinstance(proc["private"], int) and proc["private"] >= 0

    # 与快照同一口径。注意两次采样之间进程内存会波动（全量跑时尤其明显），
    # 因此只要求量级一致，不做逐字节相等——否则这是一个 flaky 断言。
    snap = get_memory_snapshot(force=True)
    ref = snap["process_rss"]
    assert proc["rss"] > 0 and ref > 0
    assert abs(proc["rss"] - ref) <= max(0.1 * ref, 64 * 1024 * 1024)
    assert proc["private"] >= proc["rss"]  # vms >= rss


# --------------------------------------------------------------------------- #
# MemoryPolicy
# --------------------------------------------------------------------------- #
def test_memory_policy_set_memory_reserve_applies():
    old = MemoryPolicy.memory_reserve_ratio
    try:
        returned = MemoryPolicy.set_memory_reserve(0.3)
        assert returned == 0.3
        assert MemoryPolicy.memory_reserve_ratio == 0.3
        assert MemoryPolicy.policy()["memory_reserve_ratio"] == 0.3

        # 预算随余量变化: 余量越大, 物理预算越小
        # 先强制刷新一次快照, 后续 memory_budget(force=False) 命中同一份缓存,
        # 否则两次采样之间 available 会漂移, 断言不稳。
        snap = get_memory_snapshot(force=True)
        budget = memory_budget(force=False)
        assert budget["physical"] == int(snap["available"] * (1.0 - 0.3))

        MemoryPolicy.set_memory_reserve(0.0)
        budget_zero = memory_budget(force=False)  # 同一份快照
        assert budget_zero["physical"] == int(snap["available"] * 1.0)
        assert budget["physical"] < budget_zero["physical"]
        assert budget["reserve"] >= budget_zero["reserve"]
    finally:
        MemoryPolicy.set_memory_reserve(DEFAULT_RESERVE)
    assert MemoryPolicy.memory_reserve_ratio == DEFAULT_RESERVE == old


@pytest.mark.parametrize("bad", [-0.1, 1.0, 0.91, 1.5, -1])
def test_memory_policy_rejects_out_of_range(bad):
    """校验区间是闭区间 ``[0, 0.9]``, 越界抛 ValueError 且不改变当前值。"""
    before = MemoryPolicy.memory_reserve_ratio
    with pytest.raises(ValueError):
        MemoryPolicy.set_memory_reserve(bad)
    assert MemoryPolicy.memory_reserve_ratio == before


def test_memory_policy_accepts_boundary_values():
    try:
        assert MemoryPolicy.set_memory_reserve(0.0) == 0.0
        assert MemoryPolicy.memory_reserve_ratio == 0.0
        assert MemoryPolicy.set_memory_reserve(0.9) == 0.9
        assert MemoryPolicy.memory_reserve_ratio == 0.9
    finally:
        MemoryPolicy.set_memory_reserve(DEFAULT_RESERVE)
    assert MemoryPolicy.memory_reserve_ratio == DEFAULT_RESERVE


# --------------------------------------------------------------------------- #
# temp_memmap / drop_memmap / cleanup_memmaps
# --------------------------------------------------------------------------- #
def test_temp_memmap_writable_readable_and_droppable():
    mm = temp_memmap((64,), dtype=np.float64)
    if mm is None:  # 平台不允许 memmap 时退化为跳过
        pytest.skip("temp_memmap returned None on this platform")

    path = mm.filename
    try:
        assert isinstance(mm, np.memmap)
        assert mm.shape == (64,)
        assert mm.dtype == np.dtype(np.float64)
        assert mm.flags.writeable
        assert os.path.exists(path)

        data = np.arange(64, dtype=np.float64)
        mm[:] = data
        mm.flush()
        assert np.array_equal(np.asarray(mm), data)
        assert float(np.asarray(mm).sum()) == float(data.sum())
    finally:
        assert drop_memmap(mm, force=True) is True

    # 后备文件已被删除, 且不再登记在册
    assert not os.path.exists(path)
    assert path not in MemoryModule._MM_FILES


def test_temp_memmap_respects_dtype_and_shape():
    mm = temp_memmap((1024,), dtype=np.int32)
    if mm is None:
        pytest.skip("temp_memmap returned None on this platform")
    path = mm.filename
    try:
        assert mm.shape == (1024,)
        assert mm.dtype == np.dtype(np.int32)
        assert mm.nbytes == 1024 * 4
        mm[0] = 7
        mm[1023] = -1
        assert int(mm[0]) == 7
        assert int(mm[1023]) == -1
    finally:
        assert drop_memmap(mm, force=True) is True
    assert not os.path.exists(path)


@pytest.mark.parametrize("bad", [np.zeros(3), "not-a-memmap", None])
def test_drop_memmap_rejects_non_memmap(bad):
    with pytest.raises(TypeError):
        drop_memmap(bad, force=True)


def test_drop_memmap_unregistered_raises():
    mm = temp_memmap((8,))
    if mm is None:
        pytest.skip("temp_memmap returned None on this platform")
    path = mm.filename
    assert cleanup_memmaps(force=True) is True
    assert not os.path.exists(path)
    # 已经清理过的 memmap 不允许再 drop
    with pytest.raises(ValueError):
        drop_memmap(mm, force=True)


@pytest.mark.parametrize("kwargs", [{"try_times": "x"}, {"try_times": True}, {"force": 1}])
def test_cleanup_memmaps_rejects_bad_types(kwargs):
    with pytest.raises(TypeError):
        cleanup_memmaps(**kwargs)


def test_cleanup_memmaps_removes_all_temp_files():
    """cleanup_memmaps 清空登记表, 并真正删除后备文件。"""
    before = set(MemoryModule._MM_FILES)

    mm = temp_memmap((128,), dtype=np.int64)
    if mm is None:
        pytest.skip("temp_memmap returned None on this platform")
    path = mm.filename
    assert os.path.exists(path)
    assert path in MemoryModule._MM_FILES

    assert cleanup_memmaps(force=True) is True
    assert not os.path.exists(path)
    assert MemoryModule._MM_FILES == {}
    assert set(MemoryModule._MM_FILES) == before == set()

    # 幂等: 再清一次不报错
    assert cleanup_memmaps(force=True) is True


def test_temp_memmap_files_are_not_left_behind():
    """所有创建的 md_memmap_* 后备文件在清理后都不存在。"""
    prefix = "md_memmap_"
    created = []
    for shape in ((16,), (64,), (1024,)):
        mm = temp_memmap(shape, dtype=np.float32)
        if mm is None:
            pytest.skip("temp_memmap returned None on this platform")
        assert os.path.basename(mm.filename).startswith(prefix)
        created.append(mm.filename)

    assert cleanup_memmaps(force=True) is True
    for path in created:
        assert not os.path.exists(path), f"临时文件未被删除: {path}"

    # 目录里不再残留本次创建的 md_memmap_*
    leftovers = [
        p
        for p in (os.path.join(tempfile.gettempdir(), n)
                  for n in os.listdir(tempfile.gettempdir())
                  if n.startswith(prefix))
        if p in created
    ]
    assert leftovers == []
