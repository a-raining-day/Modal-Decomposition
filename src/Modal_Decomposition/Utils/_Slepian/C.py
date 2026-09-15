"""
C backend for Slepian (DPSS) generation — 通过 ``ctypes`` 绑定 ``_C/`` 下编译好的核心库。

设计 (与项目"最终一致性"约定一致)
--------------------------------
* **零第三方依赖**: 只用标准库 ``ctypes`` 装载共享库; C 核心不分配堆内存 —— 输出
  与工作区都由本层用 NumPy 预分配后把指针传进去 (``out`` 为 ``k*solve_n`` 行主序
  ``float64``, 工作区长度由 ``md_slepian_worklen`` 给出), 跨语言零内存管理;
* **库文件可缺失**: 未编译时 ``load_library()`` 抛 ``RealizationError``, 由
  ``Utils.Slepian`` 决定回退到其他后端; 编译入口见 ``setup.py`` 的
  ``Modal_Decomposition.Utils._Slepian._C._slepian_native`` 扩展, 或手工::

      cd src/Modal_Decomposition/Utils/_Slepian/_C
      cl /nologo /O2 /LD /utf-8 /Fe:_slepian_native.dll slepian_dpss.c     # MSVC
      gcc -O2 -shared -fPIC -o _slepian_native.so slepian_dpss.c -lm       # GCC

* **本模块自身不接触 import cache**: 共享库句柄以模块级单例持有 (模块本身由
  ``sys.modules`` 缓存, 故每进程只装载一次), 缓存约定由 ``Utils.get_slepian()`` 层负责。

数值语义与 ``numpy`` / ``scipy`` 后端完全一致 (同一套三对角 + 中心对称折半数学,
集中比同为 FFT 自相关法, 符号约定同 Percival & Walden 1993 pg379)。

Python version: 3.10

Lib and Version:
    numpy - 2.2.6 (仅用于缓冲分配)
    共享库 - _slepian_native (本仓库 _C/slepian_dpss.c, 无第三方依赖)

Only accessed by: Utils.Slepian (经 Utils.get_slepian() 惰性注册)

Modify:
    2026.9.15
"""

import ctypes
import glob
import os
from ctypes import POINTER, c_double, c_int, c_long

import numpy as np

from ...Base.ConstDefine import DEFAULT_NUMPY_TYPE
from ...Error import RealizationError

__all__ = [
    "generate_slepian_c",
    "load_library",
    "library_path",
    "available",
]

_HERE = os.path.dirname(os.path.abspath(__file__))
_C_DIR = os.path.join(_HERE, "_C")
_BASENAME = "_slepian_native"

#: 扩展名以外的构建中间产物 (不作为库文件候选)。
_SKIP_EXT = (".c", ".h", ".obj", ".exp", ".lib", ".pyx", ".py")

_LIB = None                     # 模块级单例 (每进程装载一次)


def _candidates():
    """候选共享库路径 (按优先级): 裸名 -> 带 ABI 标记名的 setuptools 产物。"""
    for ext in (".pyd", ".dll", ".so", ".dylib"):
        yield os.path.join(_C_DIR, _BASENAME + ext)
    for path in sorted(glob.glob(os.path.join(_C_DIR, _BASENAME + ".*"))):
        if not path.endswith(_SKIP_EXT):
            yield path


def library_path():
    """返回实际装载的共享库路径; 未编译时返回 None。"""
    for path in _candidates():
        if os.path.isfile(path):
            return path
    return None


def load_library():
    """
    装载并校验 C 核心 (模块级单例)。

    Returns
    -------
    ctypes.CDLL
        已设置好 ``argtypes`` / ``restype`` 的库句柄。

    Raises
    ------
    RealizationError
        库文件不存在 (未编译), 或文件存在但不是本项目的 Slepian 核心
        (用 ``md_slepian_version()`` 校验, 防止误加载同名文件)。
    OSError
        库文件存在但无法装载 (架构/依赖不匹配)。
    """
    global _LIB
    if _LIB is not None:
        return _LIB

    path = library_path()
    if path is None:
        raise RealizationError(
            "_slepian_native 未编译: 请先构建 C 后端 "
            "(pip install -e . 或见 Utils/_Slepian/C.py 头部的编译命令)"
        )

    lib = ctypes.CDLL(path)
    lib.md_slepian_version.argtypes = []
    lib.md_slepian_version.restype = ctypes.c_char_p
    lib.md_slepian_worklen.argtypes = [c_int, c_int, c_int, c_int]
    lib.md_slepian_worklen.restype = c_long
    lib.md_slepian_dpss.argtypes = [
        c_int, c_double, c_int, c_int,
        POINTER(c_double), POINTER(c_double), c_int,
        POINTER(c_double), c_long,
    ]
    lib.md_slepian_dpss.restype = c_int

    tag = lib.md_slepian_version()
    if not tag or not tag.startswith(b"md_slepian"):
        raise RealizationError(
            f"{path} 不是本项目的 Slepian C 核心 (version tag={tag!r})"
        )

    _LIB = lib
    return _LIB


def available() -> bool:
    """C 后端是否可用 (库文件存在且能被装载)。"""
    try:
        load_library()
        return True
    except (RealizationError, OSError):
        return False


def _norm_code(norm) -> int:
    """归一化策略 → C 端编码: 0 = 单位能量 (2/None), 1 = 峰值 1 ("approximate")。"""
    if norm is None or norm == 2:
        return 0
    if norm == "approximate":
        return 1
    raise ValueError(
        f"norm must be one of (2, None, 'approximate') (norm={norm!r} 请用 mod='scipy')"
    )


def generate_slepian_c(N: int, halfBW: float, nTapers: int = None,
                       return_ratios: bool = False, norm=None,
                       sym: bool = True) -> np.ndarray:
    """
    生成 Slepian (DPSS) 序列 —— C 后端。

    Parameters
    ----------
    N : int
        序列长度 (>= 1)。
    halfBW : float
        时间-带宽积 ``NW`` (> 0)。
    nTapers : int | None
        返回条数; ``None`` 时取 ``max(1, floor(2*halfBW - 1))`` 并截断到可用上限。
    return_ratios : bool
        为 True 时同时返回集中比。
    norm : {2, None, "approximate"}
        2/None = 单位能量; "approximate" = 峰值 1 并按 ``M^2/(M^2+NW)`` 修正。
    sym : bool
        True (默认) 对称序列; False 周期性 (解 N+1 后去末点)。

    Returns
    -------
    np.ndarray | tuple
        ``(nTapers, N)``, 可选 ``(tapers, ratios)``。

    Raises
    ------
    ValueError
        ``N < 1`` / ``halfBW <= 0`` / ``nTapers < 1`` / 未知 ``norm``。
    RealizationError
        C 后端未编译。
    RuntimeError
        C 端返回非零错误码 (含错误码说明)。
    """
    N = int(N)
    if N < 1:
        raise ValueError(f"N must be >= 1, got {N}")
    if not np.isfinite(halfBW) or float(halfBW) <= 0.0:
        raise ValueError(f"halfBW must be > 0, got {halfBW!r}")
    code = _norm_code(norm)
    sym_flag = 1 if sym else 0

    solve_n = N if sym else N + 1
    if nTapers is None:
        nTapers = max(1, int(np.floor(2.0 * float(halfBW) - 1)))
    nTapers = int(nTapers)
    if nTapers < 1:
        raise ValueError(f"nTapers must be >= 1, got {nTapers}")
    nTapers = min(nTapers, solve_n)

    lib = load_library()
    work_len = int(lib.md_slepian_worklen(N, nTapers, sym_flag, 1 if return_ratios else 0))
    work = np.empty(work_len, dtype=DEFAULT_NUMPY_TYPE)
    out = np.empty((nTapers, solve_n), dtype=DEFAULT_NUMPY_TYPE)
    ratios = np.empty(nTapers, dtype=DEFAULT_NUMPY_TYPE) if return_ratios else None

    rc = lib.md_slepian_dpss(
        N, float(halfBW), nTapers, sym_flag,
        out.ctypes.data_as(POINTER(c_double)),
        ratios.ctypes.data_as(POINTER(c_double)) if return_ratios else None,
        code,
        work.ctypes.data_as(POINTER(c_double)),
        c_long(work_len),
    )
    if rc != 0:
        msg = {
            -1: "N must be >= 1",
            -2: "halfBW must be > 0",
            -3: "nTapers out of range",
            -4: "out buffer is NULL",
            -5: "work buffer too small",
            -6: "internal layout error",
        }.get(rc, "unknown error")
        raise RuntimeError(f"md_slepian_dpss failed with code {rc} ({msg})")

    tapers = out if sym else np.ascontiguousarray(out[:, :N])
    if return_ratios:
        return tapers, ratios
    return tapers
