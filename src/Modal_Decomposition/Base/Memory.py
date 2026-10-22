"""
Memory

本模块必须自实现, 不得引入其他任何同属下的子模块
"""

import os
import time
import atexit
import psutil
import tempfile
import numpy as np
from warnings import warn
from typing import Optional

__all__ = \
    [
        "should_use_memmap",

        "get_memory_snapshot",
        "get_available_memory",
        "get_commit_available",
        "get_process_memory",

        "memory_budget",

        "MemoryPolicy",

        "temp_memmap",
        "drop_memmap",
        "cleanup_memmaps"
    ]


"""
Memmap
"""
_MM_FILES: dict = {}
_ERROR_FILES: set = set()

def temp_memmap(shape, dtype=np.float64, temp_dir: Optional[str]=None, *, persist: bool=False) -> np.memmap | None:
    """Create a writable temporary memmap in the system temp dir.

    Parameters
    ----------
    persist : bool, default False
        ``False`` 时后备文件登记进 ``_MM_FILES``，由 :func:`cleanup_memmaps` 或
        解释器退出时删除（先关 ``mmap`` 句柄再删 —— Windows 上顺序不能反）；
        操作系统临时目录清理器是硬杀进程时的兜底。
        ``True`` 时文件**不登记**：本函数与 :func:`cleanup_memmaps` 都不会删它，
        所有权交给调用方，适合需要跨进程 / 跨会话保留的结果。
    """
    if type(persist) is not bool:
        raise TypeError(f"`persist` must be bool, get {type(persist).__name__}!")

    global _MM_FILES, _ERROR_FILES
    fd, path = tempfile.mkstemp(prefix="md_memmap_", suffix=".dat", dir=temp_dir)
    os.close(fd)

    try:
        mm = np.memmap(path, dtype=np.dtype(dtype), mode="w+", shape=tuple(shape))
        if not persist:
            _MM_FILES[path] = mm
        return mm
    except BaseException as be:
        warn(f"Error happen when constructing the memmap\n>> {be}")
        try:
            os.remove(path)
        except OSError as oe:
            warn(f"Error happen when removing the path of the failed memmap\n>> {oe}")
            _ERROR_FILES.add(path)

        return None

def cleanup_memmaps(try_times: int=5, force: bool=False) -> bool:
    """Close and delete every temporary memmap backing file created here.

    Idempotent; errors are swallowed (the OS cleaner remains the fallback).
    """
    global _ERROR_FILES

    if isinstance(try_times, bool):
        raise TypeError(f"`try_times` must be a int, get 'bool'!")

    if not isinstance(try_times, int):
        raise TypeError(f"`try_times` must be a int, get {type(try_times)!r}!")

    if not isinstance(force, bool):
        raise TypeError(f"`force` must be a bool, get {type(force)!r}!")

    unexpected_files = []

    for path in list(_MM_FILES.keys()):
        mm = _MM_FILES.pop(path, None)  # 从 dict 移除，只剩局部引用

        if mm is None:
            continue

        if force:
            mm.flush()
            m = mm._mmap
            del mm
            m.close()
        else:
            del mm

        try:
            os.remove(path)
        except FileNotFoundError:
            continue                     # 已被外部删掉（如 tmp 清理器/测试手删）: 视作成功
        except PermissionError:
            dose_del = False
            for times in range(try_times):
                try:
                    os.remove(path)
                    dose_del = True
                    break
                except FileNotFoundError:
                    dose_del = True
                    break
                except PermissionError:
                    pass

            if not dose_del:
                unexpected_files.append(path)

    if unexpected_files:
        warn(f"There are unexpected temporary memmap files which not be closed! ->\n>> {unexpected_files!r}")
        _ERROR_FILES.update(unexpected_files)   # 是 list: 用 update 而不是 add

    _MM_FILES.clear()
    return True if not unexpected_files else False

def drop_memmap(mm: np.memmap, try_times: int = 5, force: bool = False) -> bool:
    global _ERROR_FILES

    if not isinstance(mm, np.memmap):
        raise TypeError(f"`mm` must be np.memmap, get {type(mm)!r}!")
    if isinstance(try_times, bool):
        raise TypeError(f"`try_times` must be a int, get 'bool'!")
    if not isinstance(try_times, int):
        raise TypeError(f"`try_times` must be a int, get {type(try_times)!r}!")

    path = getattr(mm, "filename", None)
    if path not in _MM_FILES:
        raise ValueError(f"the {path!r} is not registered in `_MM_FILES`!")

    _MM_FILES.pop(path, None)

    if force:
        # 调用方必须保证此后不再使用 mm，否则会踩到已关闭的 mmap
        mm.flush()
        m = mm._mmap
        m.close()
        del mm

    for _ in range(try_times):
        try:
            os.remove(path)
            return True
        except FileNotFoundError:
            return True
        except PermissionError:
            continue

    _ERROR_FILES.add(path)
    warn(f"the {path} cannot be removed!")
    return False

atexit.register(lambda: cleanup_memmaps(force=True))

"""
Memory
"""
class Snapshot:
    ttl: float = 0.2
    now: float = 0.0

    snapshot: dict | None = None

class MemoryPolicy:
    memory_reserve_ratio: float = 0.25  # 永不参与分配的系统余量比例
    commit_ratio_limit: float = 0.6  # 可提交余量中最多可用的比例

    @classmethod
    def set_memory_reserve(cls, ratio: float) -> float:
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

        if not isinstance(ratio, (int, float)) or not (0.0 <= float(ratio) <= 0.9):
            raise ValueError(f"ratio must be in [0, 0.9], got {ratio!r}")

        cls.memory_reserve_ratio = float(ratio)
        return cls.memory_reserve_ratio

    @classmethod
    def policy(cls) -> dict:
        return {
            "memory_reserve_ratio": cls.memory_reserve_ratio,
            "commit_ratio_limit": cls.commit_ratio_limit,
            "snapshot_ttl": Snapshot.ttl,
        }

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
    now = time.monotonic()
    if not force and Snapshot.snapshot is not None and now - Snapshot.now < Snapshot.ttl:
        return Snapshot.snapshot

    vm = psutil.virtual_memory()

    sm = psutil.swap_memory()
    swap_total, swap_free = int(sm.total), int(sm.free)

    mi = psutil.Process().memory_info()
    rss, vms = int(mi.rss), int(mi.vms)

    snap = {
        "total": int(vm.total),
        "available": int(vm.available),
        "swap_total": swap_total,
        "swap_free": swap_free,
        "commit_available": int(vm.available) + int(swap_free),  # 可提交余量: 物理可用 + 页面文件空闲 (Windows 提交上限 ~= 物理 + 页面文件)
        "process_rss": rss,
        "process_private": vms,
    }

    Snapshot.snapshot = snap
    Snapshot.now = now

    return snap

def get_available_memory(force: bool = False) -> int:
    """
    Return the remaining **physical** available memory in bytes.

    Kept for backward compatibility. Prefer :func:`memory_budget` for allocation
    decisions: physical availability alone does **not** bound what a process can
    commit (the page file does), which is exactly how the old policy failed.

    Returns ``None`` when psutil is unavailable.
    """
    return get_memory_snapshot(force=force)["available"]

def get_commit_available(force: bool = False) -> int:
    """
    Return the approximate bytes still available to **commit** (physical
    available + free page file).

    This is the quantity that actually bounds allocation on a swapping system;
    ``None`` when psutil is unavailable.
    """
    return get_memory_snapshot(force=force)["commit_available"]

def get_process_memory(force: bool = False) -> dict:
    """本进程当前占用: ``{"rss": bytes, "private": bytes}`` (提交量口径为 vms)。"""
    snap = get_memory_snapshot(force=force)
    return {"rss": snap["process_rss"], "private": snap["process_private"]}


def memory_budget(extras_bytes: int = 0, force: bool = False) -> dict:
    """
    计算调用方**一次分配**的安全预算与判定依据

    Parameters
    ----------
    extras_bytes : int
        额外需要考虑的已占用/待占用字节 (例如调用方已持有的中间数组)
    force : bool
        跳过快照缓存

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

    physical = snap["available"] * (1.0 - MemoryPolicy.memory_reserve_ratio)

    commit = snap["commit_available"] * (1.0 - MemoryPolicy.memory_reserve_ratio) * MemoryPolicy.commit_ratio_limit

    limit = min(physical, commit)
    limited_by = "physical" if physical <= commit else "commit"

    budget = int(limit) - int(extras_bytes)

    return {
        "budget": budget,
        "physical": int(physical),
        "commit": int(commit),
        "reserve": int(min(snap["available"], snap["commit_available"]) * MemoryPolicy.memory_reserve_ratio),
        "limited_by": limited_by,
        "ok": budget > 0,
    }


def should_use_memmap(nbytes: int, extra: int = 0) -> bool:
    """
    Whether the projected memory usage should be served from a memmap.

    Parameters
    ----------
    nbytes : int
        Size of the data in bytes.
    extra : int
        Additional projected working memory in bytes (调用方已持有/还将持有的量,例如输入 + 镜像 + 单边谱)。

    Notes
    -----
    比例策略现在以 :func:`memory_budget` 的 ``budget`` 为基准
    (物理可用与可提交余量取小, 再扣除 ``MEMORY_RESERVE_RATIO`` 的系统余量与 ``extra``)
    """
    if isinstance(nbytes, (np.ndarray, np.memmap)):
        try:
            nbytes = nbytes.nbytes
        except Exception:
            raise ValueError("the input not the nbtyes! error happened when try to get .nbtyes from input!")

    if not isinstance(nbytes, int):
        raise ValueError("nbytes must be int! not {}".format(type(nbytes)))

    if not isinstance(extra, int):
        raise ValueError("extra must be int! not {}".format(type(extra)))

    total = nbytes + extra

    budget = memory_budget()
    usable = max(budget["budget"], 0)

    return total >= usable