"""
Slepian (DPSS) 序列生成的统一入口 —— 三后端分发 (numpy / scipy / C)。

定位
----
本模块是"对外表现的生成 Slepian 序列入口": 统一参数与返回契约, 后端实现在
``_Slepian/`` 下 (``numpy_slepian.py`` 纯 NumPy; ``C.py`` ctypes 绑定 ``_C/`` 的 C 核心;
``scipy`` 直接走 ``scipy.signal.windows.dpss``)。三者数值语义一致 (同一套三对角 +
中心对称折半数学, 集中比同为 FFT 自相关法, 符号约定同 Percival & Walden 1993 pg379),
实测互差 <= 1e-11 (见 ``docs/Slepian_Backend_Report.md``)。

返回契约 (三后端统一)
--------------------
始终返回二维 ``(nTapers, N)`` 数组 (行 = 阶数, 集中比降序), ``return_ratios=True``
时返回 ``(tapers, ratios)``; ``nTapers=None`` 取 ``max(1, floor(2*halfBW - 1))`` 并
截断到可用上限。scipy 后端在 ``Kmax=None`` 时会退化为返回单条一维窗, 本模块统一把它
规范成二维数组, 避免调用方分支。

后端选择
--------
* ``mod=None`` 用 ``Base.ConstDefine.SLEPIAN_BACKEND`` (默认 "scipy", 见基准报告
  ``docs/Slepian_Backend_Report.md`` 的选型依据); **默认路径**下若该后端不可用
  (如 C 未编译), 发一次 ``UserWarning`` 并降级 numpy;
* ``mod="auto"``: ``N`` 超过 numpy 的稠密矩阵预算 (``SLEPIAN_NUMPY_MAX_BYTES``) 且
  C 可用时用 "C", 否则用默认后端;
* 显式 ``mod`` 不可用时按"缺依赖 → 报错"处理 (含可直接照抄的编译/安装提示)。

Python version: 3.10

Lib and Version:
    numpy - 2.2.6 (必需)
    scipy - 1.15.3 (可选, mod="scipy")
    _slepian_native - 本仓库编译产物 (可选, mod="C")

Only accessed by: 全库 Slepian 调用点 (经 Utils.get_slepian() 惰性注册)

Modify:
    2026.9.15
"""

import collections
import threading
import warnings

import numpy as np

from ..Base.ConstDefine import (
    SLEPIAN_BACKEND,
    SLEPIAN_BACKEND_LIST,
    SLEPIAN_CACHE,
    SLEPIAN_CACHE_MAX_BYTES,
    SLEPIAN_CACHE_SIZE,
    SLEPIAN_NUMPY_MAX_BYTES,
    SLEPIAN_SMALL_N_ORDER,
)
from ..Error import RealizationError
from ._Slepian.C import available as _c_available
from ._Slepian.C import generate_slepian_c
from ._Slepian.numpy_slepian import generate_slepian_numpy

__all__ = [
    "slepian",
    "resolve_backend",
    "available_backends",
    "cache_info",
    "clear_cache",
]

_NORMS = (None, 2, "approximate")


def _numpy_sample_limit() -> int:
    """numpy 后端可承受的最大样本数 (由稠密矩阵预算反推: N_max = 2*sqrt(B/8))。"""
    return 2 * int(np.sqrt(SLEPIAN_NUMPY_MAX_BYTES // np.dtype(np.float64).itemsize))


def available_backends() -> dict:
    """
    各后端可用性: ``{"numpy": True, "scipy": True/False, "C": True/False}``。

    scipy 只查 ``find_spec`` (不导入); C 查库文件存在且能装载 (结果被缓存)。
    """
    import importlib.util

    return {
        "numpy": True,
        "scipy": importlib.util.find_spec("scipy") is not None,
        "C": _c_available(),
    }


def _canon_mod(mod) -> str:
    """后端名 → canonical ("numpy" / "scipy" / "C")。"""
    key = str(mod).strip()
    table = {name.lower(): name for name in SLEPIAN_BACKEND_LIST}
    table["c"] = "C"
    if key.lower() not in table:
        raise ValueError(
            f"unsupported slepian mod {mod!r}; expected one of {SLEPIAN_BACKEND_LIST} or 'auto'"
        )
    return table[key.lower()]


def _small_n_backend() -> str:
    """
    ``scipy`` 用不了时 (``NW >= N/2``) 的替代后端: 按 ``SLEPIAN_SMALL_N_ORDER``
    取第一个可用者 (默认 "C"; C 未编译则退回 "numpy")。

    实测小 N (N<=6, NW=3, K=5) 单次开销: C 4.5-6.7 us / 2.3-3.2 KB,
    numpy 38-67 us / 7.1-7.6 KB ⇒ C 快 7-10x 且内存小 2.4x, 故排在首位。
    """
    for name in SLEPIAN_SMALL_N_ORDER:
        name = _canon_mod(name)
        if name == "C" and not _c_available():
            continue
        return name
    return "numpy"


def resolve_backend(N: int, mod=None, halfBW: float = None) -> str:
    """
    解析生效后端名 (不会是 "auto")。

    Parameters
    ----------
    N : int
        序列长度。
    mod : {"numpy","scipy","C","auto",None}
        ``None`` → ``SLEPIAN_BACKEND``; ``"auto"`` → 大 N 且 C 可用时用 "C"。
    halfBW : float, optional
        时间-带宽积。给出时才应用 scipy 的固有规模限制: ``dpss`` 要求
        ``NW < N/2``, 不满足 (默认参数下即 ``N <= 6``) 时**即使显式指定 scipy** 也
        会发 ``UserWarning`` 并改用 ``SLEPIAN_SMALL_N_ORDER`` 中的后端 —— 因为该
        情形下 scipy 必然抛错, 换后端比报错更有用。不给出 ``halfBW`` 时不做此判断
        (纯后端名解析)。

    Notes
    -----
    **仅默认路径** (``mod=None``/``"auto"``) 在后端不可用 (如 C 未编译) 时降级
    numpy; 显式指定的后端不可用时由 :func:`slepian` 报错。
    """
    explicit = not (mod is None or str(mod).strip().lower() == "auto")

    if mod is None:
        name = _canon_mod(SLEPIAN_BACKEND)
    elif not explicit:
        name = "C" if (int(N) > _numpy_sample_limit() and _c_available()) \
            else _canon_mod(SLEPIAN_BACKEND)
    else:
        name = _canon_mod(mod)

    # 默认路径降级: 默认后端 (C) 不可用时改用 numpy
    if not explicit and name == "C" and not _c_available():
        warnings.warn(
            "Slepian 后端 'C' 未编译, 本次降级为 'numpy'; 构建后可恢复 "
            "(pip install -e . 或见 Utils/_Slepian/C.py 头部编译命令)",
            UserWarning,
            stacklevel=3,
        )
        return "numpy"

    # scipy 固有规模限制: NW >= N/2 时 scipy.dpss 必然抛错 -> 换更快更省的后端
    if name == "scipy" and halfBW is not None and 2.0 * float(halfBW) >= int(N):
        alt = _small_n_backend()
        warnings.warn(
            f"scipy 后端无法处理 NW >= N/2 (N={int(N)}, halfBW={float(halfBW)!r}); "
            f"本次改用 '{alt}' 后端 (小 N 下更快且更省内存)。"
            f"如需 scipy 请增大 N 或减小 halfBW。",
            UserWarning,
            stacklevel=3,
        )
        return alt
    return name


# --------------------------------------------------------------------------- #
# Tier-1 进程内缓存 (纯函数记忆化; 有界 LRU + 精确键 + 只读主副本)
# --------------------------------------------------------------------------- #
#: 缓存版本戳: 任何会改变数值结果的改动 (算法/符号约定/归一化/后端实现) 都必须提升
#: 该版本, 否则旧条目会被错误复用。同时进 ``cache_info()`` 便于诊断。
_CACHE_VERSION = "slepian-cache-v1"

#: 缓存条目 = (tapers_readonly, ratios_readonly_or_None, nbytes)
_Entry = tuple


class _TaperCache:
    """
    有界 LRU 缓存 (线程安全)。

    容量同时受**条数**与**总字节**约束: 超限时按 LRU 淘汰; 单条超过字节上限时
    调用方不应写入 (:meth:`_freeze` 返回 None, 计入 ``skipped``)。
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._data = collections.OrderedDict()
        self._bytes = 0
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.skipped = 0

    def get(self, key):
        """命中返回条目并置为最近使用; 未命中返回 None (计数 hits/misses)。"""
        with self._lock:
            entry = self._data.get(key)
            if entry is None:
                self.misses += 1
                return None
            self._data.move_to_end(key)
            self.hits += 1
            return entry

    def put(self, key, entry, maxsize: int, maxbytes: int = None) -> None:
        """写入条目; 已存在则先移除旧值, 再按条数/字节上限 LRU 淘汰。"""
        if maxbytes is None:
            maxbytes = int(SLEPIAN_CACHE_MAX_BYTES)
        with self._lock:
            old = self._data.pop(key, None)
            if old is not None:
                self._bytes -= old[2]
            self._data[key] = entry
            self._bytes += entry[2]
            while self._data and (len(self._data) > maxsize or self._bytes > maxbytes):
                _, victim = self._data.popitem(last=False)
                self._bytes -= victim[2]
                self.evictions += 1

    def clear(self) -> None:
        with self._lock:
            self._data.clear()
            self._bytes = 0

    def info(self) -> dict:
        with self._lock:
            return {
                "version": _CACHE_VERSION,
                "size": len(self._data),
                "bytes": self._bytes,
                "hits": self.hits,
                "misses": self.misses,
                "evictions": self.evictions,
                "skipped": self.skipped,
            }

    def reset_stats(self) -> None:
        with self._lock:
            self.hits = self.misses = self.evictions = self.skipped = 0


_CACHE = _TaperCache()


def _resolve_cache(cache) -> tuple:
    """``cache`` 选项 → (是否启用, 条数上限)。"""
    if cache is None:
        return bool(SLEPIAN_CACHE), int(SLEPIAN_CACHE_SIZE)
    if isinstance(cache, bool):
        return cache, int(SLEPIAN_CACHE_SIZE)
    if isinstance(cache, int) and cache >= 1:
        return True, int(cache)
    raise ValueError(
        f"cache must be None, a bool, or a positive int (maxsize), got {cache!r}"
    )


def _cache_key(backend, N, halfBW, nTapers, sym, norm, return_ratios) -> tuple:
    """
    缓存键: 与结果一一对应, 且**浮点按位精确** (``float.hex()``)。

    ``NW=3.0`` 与 ``NW=3.0000000000000004`` 给出不同的序列, 故不允许容差匹配;
    后端、是否需要集中比也入键 (避免"缺 ratios 的条目"被当成命中)。
    """
    return (
        _CACHE_VERSION,
        backend,
        int(N),
        float(halfBW).hex(),
        int(nTapers),
        bool(sym),
        str(norm),
        bool(return_ratios),
    )


def _freeze(result, return_ratios: bool):
    """
    把计算结果冻结成缓存条目 (主副本只读); 超过字节上限时返回 None (只算不存)。
    """
    tapers, ratios = result if return_ratios else (result, None)
    tapers = np.ascontiguousarray(tapers, dtype=np.float64)
    if ratios is not None:
        ratios = np.ascontiguousarray(ratios, dtype=np.float64)
    nbytes = tapers.nbytes + (0 if ratios is None else ratios.nbytes)
    if nbytes > SLEPIAN_CACHE_MAX_BYTES:
        with _CACHE._lock:
            _CACHE.skipped += 1
        return None
    master_t = tapers.copy()
    master_t.setflags(write=False)
    master_r = None
    if ratios is not None:
        master_r = ratios.copy()
        master_r.setflags(write=False)
    return (master_t, master_r, nbytes)


def _thaw(entry, return_ratios: bool):
    """命中时返回**副本** (调用方可自由改写), 缓存内主副本保持只读。"""
    if return_ratios:
        if entry[1] is None:                     # 防御: 理论上键已区分, 不会走到
            return None
        return entry[0].copy(), entry[1].copy()
    return entry[0].copy()


def cache_info() -> dict:
    """
    缓存统计: ``{version, size, bytes, hits, misses, evictions, skipped}``。

    ``skipped`` = 因单条超过 ``SLEPIAN_CACHE_MAX_BYTES`` 而**只算不存**的次数。
    """
    return _CACHE.info()


def clear_cache(reset_stats: bool = True) -> None:
    """清空缓存 (并可重置统计)。"""
    _CACHE.clear()
    if reset_stats:
        _CACHE.reset_stats()


def _check_args(N, halfBW, nTapers, norm, sym) -> tuple:
    """统一参数校验 (与各后端内部校验一致, 报错更早、信息更一致)。"""
    N = int(N)
    if N < 1:
        raise ValueError(f"N must be >= 1, got {N}")
    if not np.isfinite(halfBW) or float(halfBW) <= 0.0:
        raise ValueError(f"halfBW must be > 0, got {halfBW!r}")
    if norm not in _NORMS:
        raise ValueError(f"norm must be one of {_NORMS}, got {norm!r}")
    if not isinstance(sym, bool):
        raise ValueError(f"sym must be a bool, got {sym!r}")
    solve_n = N if sym else N + 1
    if nTapers is None:
        nTapers = max(1, int(np.floor(2.0 * float(halfBW) - 1)))
    nTapers = int(nTapers)
    if nTapers < 1:
        raise ValueError(f"nTapers must be >= 1, got {nTapers}")
    return N, float(halfBW), min(nTapers, solve_n), solve_n


def _scipy_backend(N, halfBW, nTapers, return_ratios, norm, sym):
    """
    scipy 后端: 直接调 ``scipy.signal.windows.dpss``。

    必须传**原始** ``N`` (而不是 sym=False 时的 N+1): scipy 内部会自行按 ``sym``
    做延拓/截断, 本层只负责把 ``Kmax=None`` 的单条一维窗规范成二维数组。
    """
    try:
        from scipy.signal.windows import dpss as _dpss
    except ImportError as exc:                       # pragma: no cover - 环境相关
        raise ImportError(
            "Slepian 后端 'scipy' 需要 scipy: python -m pip install scipy"
        ) from exc

    out = _dpss(int(N), float(halfBW), nTapers, sym=sym, norm=norm,
                return_ratios=return_ratios)
    if return_ratios:
        tapers, ratios = out
        ratios = np.atleast_1d(np.asarray(ratios, dtype=np.float64))
    else:
        tapers = out
    tapers = np.atleast_2d(np.asarray(tapers, dtype=np.float64))
    return (tapers, ratios) if return_ratios else tapers


def slepian(N: int, halfBW: float, nTapers: int = None, mod=None,
            return_ratios: bool = False, norm=None, sym: bool = True,
            cache=None) -> np.ndarray:
    """
    生成 Slepian (DPSS) 序列 —— 统一入口。

    Parameters
    ----------
    N : int
        序列长度 (>= 1)。
    halfBW : float
        时间-带宽积 ``NW`` (> 0); 半带宽 ``W = halfBW / N``。
    nTapers : int | None
        返回条数 (阶数 0 … nTapers-1, 集中比降序)。``None`` →
        ``max(1, floor(2*halfBW - 1))``, 并截断到 ``solve_n = N`` (``sym=False`` 时
        ``N+1``)。
    mod : {"numpy", "scipy", "C", "auto", None}
        后端选择; ``None`` 用 ``ConstDefine.SLEPIAN_BACKEND`` (默认 "C"),
        ``"auto"`` 按规模与可用性挑。默认路径下后端不可用会降级 numpy 并发一次
        ``UserWarning``; 显式指定时不可用则报错。
    return_ratios : bool
        为 True 时同时返回集中比 (降序, 与 scipy 同式)。
    norm : {None, 2, "approximate"}
        ``None``/``2`` = 单位能量; ``"approximate"`` = 峰值 1 并按
        ``M^2/(M^2+NW)`` 修正 (scipy "subsample" 只由 scipy 后端支持)。
    sym : bool
        True (默认) 长度 ``N`` 的对称序列; False 周期性 (DFT-even)。
    cache : bool | int | None
        **Tier-1 进程内缓存开关** (纯函数记忆化, 不落盘)。
        ``None`` (默认) 用 ``ConstDefine.SLEPIAN_CACHE``; ``True``/``False`` 显式开/关;
        正整数 = 本次调用的条数上限 (覆盖 ``SLEPIAN_CACHE_SIZE``)。
        键为 ``(缓存版本戳, 后端, N, halfBW 的精确位模式, nTapers, sym, norm,
        return_ratios)``: **浮点按位精确匹配**(``NW=3.0`` 与 ``3.0000000000000004``
        是两个不同结果, 不存在容差匹配); 后端与是否要集中比都入键。命中返回**副本**
        (调用方可自由改写), 缓存内的主副本只读。缓存容量同时受条数
        (``SLEPIAN_CACHE_SIZE``) 与总字节 (``SLEPIAN_CACHE_MAX_BYTES``) 约束, 超限按
        LRU 淘汰, 单条超过字节上限则只算不存 (``cache_info()["skipped"]`` 计数)。

    Returns
    -------
    np.ndarray | tuple
        ``(nTapers, N)`` 序列; ``return_ratios=True`` 时返回 ``(tapers, ratios)``。

    Notes
    -----
    ``scipy`` 后端无法处理 ``NW >= N/2`` (默认参数下即 ``N <= 6``): 该情形下
    :func:`resolve_backend` 会发 ``UserWarning`` 并自动改用 ``SLEPIAN_SMALL_N_ORDER``
    中的后端 (默认 "C", 实测小 N 下更快更省内存), 而不是抛错。

    Raises
    ------
    ValueError
        参数非法、后端名未知或 ``cache`` 取值非法。
    RealizationError
        显式指定 "C" 但未编译。
    ImportError
        显式指定 "scipy" 但未安装。
    """
    N, halfBW, nTapers, solve_n = _check_args(N, halfBW, nTapers, norm, sym)
    backend = resolve_backend(N, mod, halfBW=halfBW)
    use_cache, maxsize = _resolve_cache(cache)

    key = None
    if use_cache:
        key = _cache_key(backend, N, halfBW, nTapers, sym, norm, return_ratios)
        hit = _CACHE.get(key)
        if hit is not None:
            return _thaw(hit, return_ratios)

    if backend == "numpy":
        result = generate_slepian_numpy(N, halfBW, nTapers,
                                        return_ratios=return_ratios, norm=norm, sym=sym)
    elif backend == "C":
        if not _c_available():
            raise RealizationError(
                "Slepian 后端 'C' 未编译: 请构建 _slepian_native "
                "(pip install -e . 或见 Utils/_Slepian/C.py 头部的编译命令)"
            )
        result = generate_slepian_c(N, halfBW, nTapers,
                                    return_ratios=return_ratios, norm=norm, sym=sym)
    elif backend == "scipy":
        result = _scipy_backend(N, halfBW, nTapers, return_ratios, norm, sym)
    else:                                                # pragma: no cover
        raise ValueError(f"unsupported backend {backend!r}")

    if use_cache:
        entry = _freeze(result, return_ratios)
        if entry is not None:
            _CACHE.put(key, entry, maxsize, SLEPIAN_CACHE_MAX_BYTES)
    return result
