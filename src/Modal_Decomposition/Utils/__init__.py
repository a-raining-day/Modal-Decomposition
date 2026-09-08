"""
Utility subpackage: validation, monotonicity, memory policy, seed
management, Hilbert / Spline / Envelope tooling.

统一约定: 各工具模块自身不接触缓存; 本包通过 ``_UTILS_MODULES`` 目录提供
同名 getter (get_check / get_memory / get_monotonicity / get_seed_module /
get_hilbert / get_spline / get_envelope), 首次访问时惰性 import 并注册进
进程级 import 缓存 (``Base.Cache.cache``), 之后全库/全局可
``cache.get(key)`` 取得同一实例。速度对比见 docs/CacheSpeedReport.md。
"""

from .Check import Check_Time_and_Signal, detect_dtype, is_uniform, require_ndim, to_signal
from .Memory import get_available_memory, get_memory_policy, set_absolute_limit, set_memmap_ratio, should_use_memmap
from .Monotonicity import Monotony, is_monotonic, monotonic
from .Seed import get_seed, resolve_seed, set_seed

__all__ = [
    "Check_Time_and_Signal",
    "is_uniform",
    "require_ndim",
    "to_signal",
    "detect_dtype",
    "get_available_memory",
    "set_memmap_ratio",
    "set_absolute_limit",
    "get_memory_policy",
    "should_use_memmap",
    "Monotony",
    "monotonic",
    "is_monotonic",
    "set_seed",
    "get_seed",
    "resolve_seed",
    "get_check",
    "get_memory",
    "get_monotonicity",
    "get_seed_module",
    "get_hilbert",
    "get_spline",
    "get_envelope",
    "get_chunk",
    "get_peaks",
]

# (cache 键, 描述) —— Utils 全部工具模块的惰性注册目录。
# 统一约定: 模块自身不接触缓存; cache 只在本层的 getter 中引入。
_UTILS_MODULES = {
    "Check": (
        "Modal_Decomposition.Utils.Check",
        "Utils.Check: 信号/时间轴校验与 memmap 输入层 (Check_Time_and_Signal 等)",
    ),
    "Memory": (
        "Modal_Decomposition.Utils.Memory",
        "Utils.Memory: 进程级内存/memmap 策略 (should_use_memmap 等)",
    ),
    "Monotonicity": (
        "Modal_Decomposition.Utils.Monotonicity",
        "Utils.Monotonicity: 单调性检测 (monotonic/is_monotonic/Monotony)",
    ),
    "Seed": (
        "Modal_Decomposition.Utils.Seed",
        "Utils.Seed: 两级随机种子管理 (set_seed/get_seed/resolve_seed)",
    ),
    "Hilbert": (
        "Modal_Decomposition.Utils.Hilbert",
        "Utils.Hilbert: Hilbert/FHT 后端分发薄封装 (Scipy/FHT 已实现)",
    ),
    "Spline": (
        "Modal_Decomposition.Utils.Spline",
        "Utils.Spline: scipy.interpolate 统一样条封装 (默认 UnivariateSpline)",
    ),
    "Envelope": (
        "Modal_Decomposition.Utils.Envelope",
        "Utils.Envelope: 包络提取 (Hilbert/Lowpass/IQ/PeakInterpolation)",
    ),
    "Chunk": (
        "Modal_Decomposition.Utils.Chunk",
        "Utils.Chunk: 统一分块工具 (iter_chunks/chunked_map/chunked_fill/exo_chunks/"
        "adapt_chunk_size)",
    ),
    "Peaks": (
        "Modal_Decomposition.Utils.Peaks",
        "Utils.Peaks: 统一峰检测 (scipy/numpy/numba 三后端, 统一返回契约)",
    ),
}


def _get_cached_module(module_name: str, key: str, description: str):
    """
    惰性取用 import 缓存中的模块: 首次访问时 import + 注册 (幂等)。

    后续访问不再 import, 直接返回 ``cache.get(key)`` 指向的进程内唯一实例;
    即使另一条 import 路径 (如 ``src.Modal_Decomposition`` 与
    ``Modal_Decomposition``) 先注册了同键, ``cache.add`` 也不覆盖 ——
    返回的始终是最先注册的那一个实例。
    """
    from ..Base.Cache import cache

    if not cache.check(key):
        from importlib import import_module

        module = import_module(f"{__name__}.{module_name}")
        cache.add(key, module, description=description)
    return cache.get(key)


def get_check():
    """Return the Check module via the import cache (lazy import + register)."""
    key, desc = _UTILS_MODULES["Check"]
    return _get_cached_module("Check", key, desc)


def get_memory():
    """Return the Memory module via the import cache (lazy import + register)."""
    key, desc = _UTILS_MODULES["Memory"]
    return _get_cached_module("Memory", key, desc)


def get_monotonicity():
    """
    Return the Monotonicity module (``monotonic``/``is_monotonic``/``Monotony``)
    via the import cache (lazy import + register).
    """
    key, desc = _UTILS_MODULES["Monotonicity"]
    return _get_cached_module("Monotonicity", key, desc)


def get_seed_module():
    """
    Return the Seed module via the import cache (lazy import + register).

    命名避开与 ``get_seed()`` (读取全局种子的函数) 冲突。
    """
    key, desc = _UTILS_MODULES["Seed"]
    return _get_cached_module("Seed", key, desc)


def get_hilbert():
    """
    Return the Hilbert facade module (``Utils.Hilbert``) via the import cache.

    首次访问时惰性 import 并注册 (键
    ``"Modal_Decomposition.Utils.Hilbert"``); 之后任意组件可用
    ``cache.get(...)`` 取得同一进程级实例。
    """
    key, desc = _UTILS_MODULES["Hilbert"]
    return _get_cached_module("Hilbert", key, desc)


def get_spline():
    """
    Return the Spline module (``Utils.Spline``) via the import cache.

    首次访问时惰性 import 并注册 (键
    ``"Modal_Decomposition.Utils.Spline"``); 之后任意组件可用
    ``cache.get(...)`` 取得同一进程级实例。
    """
    key, desc = _UTILS_MODULES["Spline"]
    return _get_cached_module("Spline", key, desc)


def get_envelope():
    """
    Return the Envelope module (``Utils.Envelope``) via the import cache.

    首次访问时惰性 import 并注册 (键
    ``"Modal_Decomposition.Utils.Envelope"``); 之后任意组件可用
    ``cache.get(...)`` 取得同一进程级实例。
    """
    key, desc = _UTILS_MODULES["Envelope"]
    return _get_cached_module("Envelope", key, desc)


def get_chunk():
    """
    Return the Chunk module (``Utils.Chunk``) via the import cache.

    首次访问时惰性 import 并注册 (键
    ``"Modal_Decomposition.Utils.Chunk"``); 之后任意组件可用
    ``cache.get(...)`` 取得同一进程级实例。
    """
    key, desc = _UTILS_MODULES["Chunk"]
    return _get_cached_module("Chunk", key, desc)


def get_peaks():
    """
    Return the Peaks module (``Utils.Peaks``) via the import cache.

    首次访问时惰性 import 并注册 (键
    ``"Modal_Decomposition.Utils.Peaks"``); 之后任意组件可用
    ``cache.get(...)`` 取得同一进程级实例。
    """
    key, desc = _UTILS_MODULES["Peaks"]
    return _get_cached_module("Peaks", key, desc)




