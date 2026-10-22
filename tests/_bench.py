"""测试侧公共设施：计时、峰值内存、csv/json 落盘、绘图风格、规模档位。

本模块只被 ``tests/`` 使用，不进 `src/` 包，不参与库运行时。

目录约定（与 `table/README.md`、`fig/README.md` 一致）::

    table/utils/<tool>/*.csv     工具深度测试的汇总表
    table/base/<tool>/*.csv
    table/size_axis/*.csv        1 MiB – 600 MiB 规模轴
    fig/utils/<tool>/*.png
    fig/size_axis/*.png

所有规模档位 **MB = MiB = 2**20 B**。
"""

from __future__ import annotations

import csv
import gc
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TABLE = ROOT / "table"
FIG = ROOT / "fig"

#: 规模轴档位（MiB）。1 MiB = 1 048 576 B = 131 072 个 float64。
SIZES_MIB: tuple[int, ...] = (1, 50, 100, 200, 500, 600)

MiB = 1024 * 1024


# --------------------------------------------------------------------------- #
# 目录与落盘
# --------------------------------------------------------------------------- #
def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_csv(path: Path, rows: Sequence[dict], fieldnames: Iterable[str] | None = None) -> Path:
    """把 ``rows`` 写成 csv（列名取首行键序，或显式给出）。"""
    ensure_dir(path.parent)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(fieldnames))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return path


def write_json(path: Path, payload: Any) -> Path:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# 规模/信号
# --------------------------------------------------------------------------- #
def n_elements(size_mib: int, dtype=np.float64) -> int:
    """``size_mib`` MiB 的数据在 ``dtype`` 下的元素数。"""
    return int(size_mib * MiB // np.dtype(dtype).itemsize)


def make_signal(size_mib: int, dtype=np.float64, kind: str = "mixed") -> np.ndarray:
    """构造一条 ``size_mib`` MiB 的确定性信号。

    kind
    ----
    ``mixed``      三音调 + 缓慢趋势（有极值、不单调，接近真实信号）
    ``ramp``       严格递增（单调、等间距）
    ``random``     标准正态（无结构）
    ``flat``       常数（全等值，极值检测的病态输入）
    """
    n = n_elements(size_mib, dtype)
    t = np.linspace(0.0, 1.0, n, endpoint=False)
    if kind == "mixed":
        x = (np.sin(2 * np.pi * 7 * t) + 0.5 * np.sin(2 * np.pi * 91 * t)
             + 0.25 * np.sin(2 * np.pi * 701 * t) + 0.1 * t)
    elif kind == "ramp":
        x = t
    elif kind == "random":
        x = np.random.default_rng(0).standard_normal(n)
    elif kind == "flat":
        x = np.full(n, 3.0)
    else:  # pragma: no cover
        raise ValueError(f"unknown kind {kind!r}")
    return np.ascontiguousarray(x, dtype=dtype)


# --------------------------------------------------------------------------- #
# 计时
# --------------------------------------------------------------------------- #
def timeit(fn: Callable[[], Any], repeat: int = 5, warmup: int = 1) -> dict:
    """同一进程内重复调用，返回 best / median / mean / spread（毫秒）。"""
    for _ in range(warmup):
        fn()
    xs: list[float] = []
    for _ in range(max(1, repeat)):
        gc.collect()
        t0 = time.perf_counter()
        fn()
        xs.append((time.perf_counter() - t0) * 1e3)
    return {
        "repeat": len(xs),
        "best_ms": min(xs),
        "median_ms": statistics.median(xs),
        "mean_ms": statistics.fmean(xs),
        "max_ms": max(xs),
    }


def peak_rss_mib() -> float:
    """本进程**历史峰值**工作集（MiB）。不可用时报 ``nan``。"""
    try:
        import psutil

        info = psutil.Process().memory_info()
        peak = getattr(info, "peak_wset", None)
        if peak is None:                      # 非 Windows / 旧版 psutil
            peak = info.rss
        return peak / MiB
    except Exception:                          # pragma: no cover
        return float("nan")


def rss_mib() -> float:
    import psutil

    return psutil.Process().memory_info().rss / MiB


def reset_peak_rss() -> None:
    """把本进程的峰值工作集重置为当前值（Windows 专用，失败静默）。

    ``psutil`` 只能读峰值，读回来当基线再用差值即可；这里用
    ``SetProcessWorkingSetSize`` 之外最简单的办法：记录当前 rss 作为基线，
    由调用方做差。见 :func:`peak_delta_mib`。
    """
    try:
        import psutil

        psutil.Process().memory_info()
    except Exception:                          # pragma: no cover
        pass


# --------------------------------------------------------------------------- #
# 绘图
# --------------------------------------------------------------------------- #
def plt_style() -> None:
    """统一的英文图注风格：默认字体不含中文字形，图里只写 ASCII。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.dpi": 130,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "font.size": 9,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def savefig(fig, path: Path) -> Path:
    ensure_dir(path.parent)
    fig.savefig(path)
    return path


# --------------------------------------------------------------------------- #
# 自举：允许 `python tests/utils/test_X_deep.py` 直接跑导出分支
# --------------------------------------------------------------------------- #
def bootstrap_src() -> None:
    src = str(ROOT / "src")
    if src not in sys.path:
        sys.path.insert(0, src)


# --------------------------------------------------------------------------- #
# 命令行
# --------------------------------------------------------------------------- #
def add_export_flag(parser) -> None:
    parser.add_argument("--export", action="store_true",
                        help="写出 table/ 与 fig/ 产物（默认只打印）")


def print_table(rows: Sequence[dict], fields: Sequence[str] | None = None) -> None:
    """简易等宽表，供脚本直接运行时看数。"""
    fields = list(fields or rows[0].keys())
    widths = {f: max(len(str(f)), *(len(_fmt(r.get(f))) for r in rows)) for f in fields}
    print("  ".join(str(f).ljust(widths[f]) for f in fields))
    print("  ".join("-" * widths[f] for f in fields))
    for r in rows:
        print("  ".join(_fmt(r.get(f)).ljust(widths[f]) for f in fields))


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)


if __name__ == "__main__":  # pragma: no cover
    print("ROOT ", ROOT)
    print("TABLE", TABLE)
    print("FIG  ", FIG)
    print("env  ", os.name, sys.version.split()[0], "numpy", np.__version__)
    for s in SIZES_MIB:
        print(f"{s:4d} MiB -> {n_elements(s):>12,d} float64 elements")
