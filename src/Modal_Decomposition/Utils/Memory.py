"""
Process-wide memory policy: decide when disk-backed (``np.memmap``) processing
is preferable to in-RAM processing, and how much a caller may safely allocate.

Two mutually exclusive strategies are available; ``USE_RATIO_STRATEGY``
selects which one is active (setting one always switches away from the
other):

- Ratio strategy (``USE_RATIO_STRATEGY = True``): use memmap when the
  projected usage reaches ``MEMMAP_RATIO_LIMIT`` (default ``0.6``, i.e. 60%)
  of the **safe budget** (see below).
- Absolute strategy (``USE_RATIO_STRATEGY = False``): use memmap when the
  projected usage reaches ``ABSOLUTE_MEMMAP_LIMIT_BYTES`` (default 2 GiB).

安全预算 (safe budget) —— 为什么不能只看"物理可用内存"
-------------------------------------------------------
旧版直接用 ``psutil.virtual_memory().available`` 判定, 在实测中**判定失效**:
本机物理可用内存只有 3.1 GB, 但页面文件另有 15.3 GB 空闲, 于是进程"提交 (commit)"
到 10 GB 也不会失败 —— 只是整机被拖入换页/内存压缩, 而策略始终认为"内存充足"。
失效有三个成分, 本模块逐条修:

1. **只看物理可用, 不看可提交**: 换页让提交量远超物理可用, 但提交配额耗尽时分配
   才真正失败。``get_commit_available()`` 给出"物理可用 + 页面文件空闲"的可提交
   余量; 预算取两者的较小值。
2. **不扣本进程已占用**: 调用方往往已持有输入/中间量, 只看系统余量会高估。
   ``get_process_memory()`` 提供本进程 RSS/私有提交量, ``memory_budget(extra=...)``
   把它算进去。
3. **没有系统余量保留**: 把可用内存吃到 0 会让 OS 与其它进程一起换页。
   ``MEMORY_RESERVE_RATIO`` (默认 0.25) 的份额永远不参与分配。

预算公式::

    budget = min(物理可用, 可提交余量) × (1 - MEMORY_RESERVE_RATIO) - 已占用(可选)

本模块只做决策与读取, 从不分配内存; 策略是进程级状态, 影响所有
:func:`should_use_memmap` 调用方 (``Utils.Check`` 输入层、``Utils.Monotonicity``
分块粒度、``Utils.Chunk.default_chunk_size``)。

References
----------
实测报告: ``docs/Memory_Detection_and_FFT_Backend_Report.md``
"""

import time
from typing import Optional

try:
    import psutil
except ImportError:  # psutil is a declared dependency; degrade gracefully
    psutil = None

from ..Base.ConstDefine import SIZE

__all__ = [
    "USE_RATIO_STRATEGY",
    "MEMMAP_RATIO_LIMIT",
    "ABSOLUTE_MEMMAP_LIMIT_BYTES",
    "MEMORY_RESERVE_RATIO",
    "COMMIT_RATIO_LIMIT",
    "should_use_memmap",
    "set_memmap_ratio",
    "set_absolute_limit",
    "set_memory_reserve",
    "get_memory_policy",
    "get_available_memory",
    "get_commit_available",
    "get_process_memory",
    "get_memory_snapshot",
    "memory_budget",
    "format_bytes",
]

# Process-wide policy state. Exactly one strategy is active at any time.
USE_RATIO_STRATEGY: bool = True
MEMMAP_RATIO_LIMIT: float = 0.6
#: 默认绝对阈值 = 2GB (大小统一取自 Base.ConstDefine.SIZE)。
ABSOLUTE_MEMMAP_LIMIT_BYTES: int = 2 * SIZE["1GB"]

#: 永不参与分配的系统余量比例 (给 OS / 其它进程留出的份额)。
MEMORY_RESERVE_RATIO: float = 0.25

#: 可提交余量中最多可用的比例 (提交配额也不该吃干, 否则换页会先拖垮整机)。
COMMIT_RATIO_LIMIT: float = 0.75

# 短 TTL 快照缓存: 一次取齐物理/提交/进程信息, 避免每次判定都调多次 psutil。
# TTL 由 1.0s 收紧到 0.25s —— 大数组分配期间内存变化很快, 1s 的快照会明显过期。
_SNAPSHOT_TTL: float = 0.25
_cache_ts: float = 0.0
_cache_value: int = 0
_snapshot_ts: float = 0.0
_snapshot: Optional[dict] = None


def format_bytes(n: Optional[int]) -> str:
    """把字节数格式化成 ``1.25 GB`` 这样的字符串 (None → ``"n/a"``)。"""
    if n is None:
        return "n/a"
    x = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(x) < 1024.0 or unit == "TB":
            return "%.2f %s" % (x, unit)
        x /= 1024.0
    return "%.2f TB" % x


def get_memory_snapshot(force: bool = False) -> dict:
    """
    返回内存/提交状态的快照 (字节), 带 ``_SNAPSHOT_TTL`` 秒缓存。

    Returns
    -------
    dict
        ``total``/``available`` (物理), ``swap_total``/``swap_free`` (页面文件),
        ``commit_available`` (可提交余量 = available + swap_free),
        ``process_rss``/``process_private`` (本进程), ``psutil`` (是否可用)。
    """
    global _snapshot_ts, _snapshot, _cache_ts, _cache_value

    now = time.monotonic()
    if not force and _snapshot is not None and now - _snapshot_ts < _SNAPSHOT_TTL:
        return _snapshot

    if psutil is None:
        snap = {
            "psutil": False, "total": None, "available": None,
            "swap_total": None, "swap_free": None, "commit_available": None,
            "process_rss": None, "process_private": None,
        }
    else:
        vm = psutil.virtual_memory()
        try:
            sm = psutil.swap_memory()
            swap_total, swap_free = int(sm.total), int(sm.free)
        except Exception:
            swap_total = swap_free = 0
        try:
            mi = psutil.Process().memory_info()
            rss, vms = int(mi.rss), int(mi.vms)
        except Exception:
            rss = vms = 0
        snap = {
            "psutil": True,
            "total": int(vm.total),
            "available": int(vm.available),
            "swap_total": swap_total,
            "swap_free": swap_free,
            # 可提交余量: 物理可用 + 页面文件空闲 (Windows 提交上限 ≈ 物理 + 页面文件)
            "commit_available": int(vm.available) + int(swap_free),
            "process_rss": rss,
            "process_private": vms,
        }

    _snapshot = snap
    _snapshot_ts = now
    # 兼容旧字段: get_available_memory 的缓存与快照同源。
    _cache_ts, _cache_value = now, (snap["available"] or 0)
    return snap


def get_available_memory(force: bool = False) -> Optional[int]:
    """
    Return the remaining **physical** available memory in bytes.

    Kept for backward compatibility. Prefer :func:`memory_budget` for allocation
    decisions: physical availability alone does **not** bound what a process can
    commit (the page file does), which is exactly how the old policy failed.

    Returns ``None`` when psutil is unavailable.
    """
    snap = get_memory_snapshot(force=force)
    return snap["available"]


def get_commit_available(force: bool = False) -> Optional[int]:
    """
    Return the approximate bytes still available to **commit** (physical
    available + free page file).

    This is the quantity that actually bounds allocation on a swapping system;
    ``None`` when psutil is unavailable.
    """
    snap = get_memory_snapshot(force=force)
    return snap["commit_available"]


def get_process_memory(force: bool = False) -> dict:
    """本进程当前占用: ``{"rss": bytes, "private": bytes}`` (提交量口径为 vms)。"""
    snap = get_memory_snapshot(force=force)
    return {"rss": snap["process_rss"], "private": snap["process_private"]}


def memory_budget(nbytes: int = 0, force: bool = False) -> dict:
    """
    计算调用方**一次分配**的安全预算与判定依据。

    Parameters
    ----------
    nbytes : int
        额外需要考虑的已占用/待占用字节 (例如调用方已持有的中间数组)。
    force : bool
        跳过快照缓存。

    Returns
    -------
    dict
        ``budget``    — 本次可安全分配的字节数 (已扣除系统余量与 ``nbytes``);
        ``physical``  — 物理可用 (扣除余量后);
        ``commit``    — 可提交余量 (扣除余量与 COMMIT_RATIO_LIMIT 后);
        ``reserve``   — 保留给系统的份额;
        ``limited_by``— ``"physical"`` / ``"commit"`` / ``"psutil-missing"``;
        ``ok``        — ``budget > 0``。
    """
    snap = get_memory_snapshot(force=force)
    if not snap["psutil"]:
        return {"budget": None, "physical": None, "commit": None, "reserve": None,
                "limited_by": "psutil-missing", "ok": True}

    physical = snap["available"] * (1.0 - MEMORY_RESERVE_RATIO)
    commit = snap["commit_available"] * (1.0 - MEMORY_RESERVE_RATIO) * COMMIT_RATIO_LIMIT
    limit = min(physical, commit)
    limited_by = "physical" if physical <= commit else "commit"
    budget = int(limit) - int(nbytes)
    return {
        "budget": budget,
        "physical": int(physical),
        "commit": int(commit),
        "reserve": int(min(snap["available"], snap["commit_available"]) * MEMORY_RESERVE_RATIO),
        "limited_by": limited_by,
        "ok": budget > 0,
    }


def set_memory_reserve(ratio: float) -> None:
    """
    设置永不参与分配的系统余量比例。

    Parameters
    ----------
    ratio : float
        ``[0, 0.9]`` 之间的比例 (0.25 = 留出 25%)。

    Raises
    ------
    ValueError
        比例越界。
    """
    global MEMORY_RESERVE_RATIO
    if not isinstance(ratio, (int, float)) or not (0.0 <= float(ratio) <= 0.9):
        raise ValueError(f"ratio must be in [0, 0.9], got {ratio!r}")
    MEMORY_RESERVE_RATIO = float(ratio)


def set_memmap_ratio(ratio: float) -> None:
    """
    Switch to the ratio strategy: memmap when the projected usage reaches
    ``ratio`` of the remaining available memory.

    Parameters
    ----------
    ratio : float
        Threshold fraction in ``(0, 1]``.

    Raises
    ------
    ValueError
        On a ratio outside ``(0, 1]``.
    """
    global USE_RATIO_STRATEGY, MEMMAP_RATIO_LIMIT

    if not isinstance(ratio, (int, float)) or not (0.0 < ratio <= 1.0):
        raise ValueError(f"ratio must be in (0, 1], got {ratio!r}")

    MEMMAP_RATIO_LIMIT = float(ratio)
    USE_RATIO_STRATEGY = True


def set_absolute_limit(limit_bytes: int) -> None:
    """
    Switch to the absolute strategy: memmap when the projected usage reaches
    ``limit_bytes``.

    Parameters
    ----------
    limit_bytes : int
        Threshold in bytes; must be positive.

    Raises
    ------
    ValueError
        On a non-positive limit.
    """
    global USE_RATIO_STRATEGY, ABSOLUTE_MEMMAP_LIMIT_BYTES

    if int(limit_bytes) <= 0:
        raise ValueError(f"limit_bytes must be positive, got {limit_bytes!r}")

    ABSOLUTE_MEMMAP_LIMIT_BYTES = int(limit_bytes)
    USE_RATIO_STRATEGY = False


def get_memory_policy() -> dict:
    """Return a read-only view of the current policy (策略 + 安全余量参数)。"""
    return {
        "use_ratio_strategy": USE_RATIO_STRATEGY,
        "memmap_ratio_limit": MEMMAP_RATIO_LIMIT,
        "absolute_memmap_limit_bytes": ABSOLUTE_MEMMAP_LIMIT_BYTES,
        "memory_reserve_ratio": MEMORY_RESERVE_RATIO,
        "commit_ratio_limit": COMMIT_RATIO_LIMIT,
        "snapshot_ttl": _SNAPSHOT_TTL,
    }


def should_use_memmap(nbytes: int, extra: int = 0) -> bool:
    """
    Whether the projected memory usage should be served from a memmap.

    Parameters
    ----------
    nbytes : int
        Size of the data in bytes.
    extra : int
        Additional projected working memory in bytes (调用方已持有/还将持有的量,
        例如输入 + 镜像 + 单边谱)。

    Notes
    -----
    比例策略现在以 :func:`memory_budget` 的 ``budget`` 为基准 (物理可用与可提交
    余量取小, 再扣除 ``MEMORY_RESERVE_RATIO`` 的系统余量与 ``extra``), 而不再用
    裸的 ``available`` —— 这正是旧版"检测失效"的修法: 只看物理可用会让
    "提交 10GB 而物理只剩 3GB" 的调用被判为内存充足。
    psutil 缺失时退化为绝对阈值判定。
    """
    total = int(nbytes) + int(extra)

    if USE_RATIO_STRATEGY:
        budget = memory_budget()
        if budget["budget"] is not None:
            usable = max(int(budget["budget"]), 0)
            return total >= MEMMAP_RATIO_LIMIT * usable
        # psutil unavailable: degrade to the absolute judgement.
        return (
            ABSOLUTE_MEMMAP_LIMIT_BYTES > 0
            and total >= ABSOLUTE_MEMMAP_LIMIT_BYTES
        )

    return total >= ABSOLUTE_MEMMAP_LIMIT_BYTES
