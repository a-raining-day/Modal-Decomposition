"""规模轴：低成本工具在 1 / 50 / 100 / 200 / 500 / 600 MiB 上的实测。

分解法**不做**规模轴（见 `docs/test/methods/Large_Array_Memory_Report.md` 的说明：
单模态就要 6–10 GB，远超本机 4.2 GB 预算），本文件只覆盖"读一遍数据"量级的工具。

用法::

    .\\.venv\\Scripts\\python.exe tests/size_axis/run_scale.py --tool all --export
    .\\.venv\\Scripts\\python.exe tests/size_axis/run_scale.py --tool monotonicity --export --sizes 1 50

产物：

* ``table/size_axis/<tool>_<op>.csv``   逐 op 一张：每个规模一行
* ``table/size_axis/summary.csv``       所有 op 合并（供画图/总览）
* ``fig/size_axis/<tool>.png``          每个工具一张：耗时 + 峰值分配随规模的变化

计量口径
--------
* ``best_ms`` / ``median_ms``：同进程内 ``repeat`` 次独立调用的最好/中位；
  大档位只做 2 次（600 MiB 一档就要几秒）。
* ``alloc_peak_mib``：``tracemalloc`` 在**该 op 调用期间**的峰值（NumPy 的大缓冲
  按一次分配记账，所以这项开销很小），口径是"这个 op 额外申请了多少"。
* ``rss_delta_mib``：op 前后进程 RSS 差（稳态增量，含未被 tracemalloc 记账的部分）。
* ``mb_per_s``：按**输入 MiB** 计的有效吞吐。
"""

from __future__ import annotations

import argparse
import gc
import sys
import tracemalloc
from pathlib import Path
from typing import Any, Callable

import numpy as np

_TESTS = Path(__file__).resolve().parents[1]
if str(_TESTS) not in sys.path:
    sys.path.insert(0, str(_TESTS))

import _bench as B  # noqa: E402

B.bootstrap_src()

from Modal_Decomposition.Base import Chunk  # noqa: E402
from Modal_Decomposition.Utils import Check, Check_Time_and_Signal  # noqa: E402
from Modal_Decomposition.Utils.Monotonicity import (  # noqa: E402
    give_monotony,
    monotonic,
)
from Modal_Decomposition.Utils import (  # noqa: E402
    envelope,
    fast_poly,
    find_peaks,
    hilbert,
    mirror,
)


# --------------------------------------------------------------------------- #
# op 注册表：tool -> [(op 名, 输入类型, 函数)]
# 函数签名 fn(S, size_mib) -> 任意（返回值只用于防优化/记录）
# --------------------------------------------------------------------------- #
Op = tuple[str, str, Callable[[np.ndarray, int], Any]]

REGISTRY: dict[str, list[Op]] = {
    "monotonicity": [
        ("monotonic_full_scan", "ramp", lambda S, s: monotonic(S, mod="increasing")),
        ("monotonic_strict_scan", "ramp", lambda S, s: monotonic(S, strict=True, mod="increasing")),
        ("monotonic_early_exit", "mixed", lambda S, s: monotonic(S, mod="increasing")),
        ("give_monotony", "mixed", lambda S, s: give_monotony(S)),
        ("monotonic_chunked_1mib", "ramp",
         lambda S, s: monotonic(S, mod="increasing", chunk_size=1048576)),
    ],
    "chunk": [
        ("chunk_iter_default", "mixed", lambda S, s: _count_chunks(S, None)),
        ("chunk_iter_1mib", "mixed", lambda S, s: _count_chunks(S, 1048576)),
        ("chunk_size_calc", "mixed", lambda S, s: Chunk.chunk_size(S.nbytes)),
        ("chunked_map_x2p1", "mixed", lambda S, s: _chunked_map(S)),
        ("chunked_fill_to_mem", "mixed", lambda S, s: _fill_mem(S)),
        ("chunked_fill_to_disk", "mixed", lambda S, s: _fill_disk(S)),
    ],
    "poly": [
        # x_ref/y_ref 固定 64 点；规模轴量的是 "x_new 很大" 时的 Vandermonde 放大
        ("eval_deg3", "mixed", lambda S, s: float(fast_poly(_XREF, _YREF, 3, S)[0])),
        ("eval_deg8", "mixed", lambda S, s: float(fast_poly(_XREF, _YREF, 8, S)[0])),
    ],
    "peaks": [
        ("numpy_max", "mixed", lambda S, s: find_peaks(S, mod="max", backend="numpy")[0].size),
        ("numpy_both", "mixed", lambda S, s: find_peaks(S, mod="both", backend="numpy")[0].size),
        ("scipy_max", "mixed", lambda S, s: find_peaks(S, mod="max", backend="scipy")[0].size),
        ("numba_both", "mixed", lambda S, s: find_peaks(S, mod="both", backend="numba")[0].size),
    ],
    "envelope": [
        ("hilbert", "mixed", lambda S, s: float(np.abs(envelope(S, method="Hilbert"))[-1])),
        ("lowpass", "mixed", lambda S, s: float(envelope(
            S, method="Lowpass", fs=2.0, cf=0.05)[-1])),
        ("iq", "mixed", lambda S, s: float(envelope(
            S, method="IQ", fs=2.0, fc=0.1, cf=0.05)[-1])),
        ("peak_interp", "mixed", lambda S, s: float(envelope(
            S, method="PeakInterpolation")[-1])),
    ],
    "hilbert": [
        ("fft", "mixed", lambda S, s: float(np.abs(hilbert(S, mod="FFT")[-1]))),
        ("fht", "mixed", lambda S, s: float(np.abs(hilbert(S, mod="FHT")[-1]))),
        ("scipy", "mixed", lambda S, s: float(np.abs(hilbert(S, mod="scipy")[-1]))),
    ],
    "mirror": [
        ("mirror_nbsym8", "mixed", lambda S, s: mirror(S, mod="mirror", nbsym=8).size),
        ("extrema_nbsym4", "mixed",
         lambda S, s: mirror(S, mod="extrema", nbsym=4)[0].size),
        ("blcse", "mixed", lambda S, s: mirror(S, mod="BLCSE", nbsym=4)[0].size),
    ],
    "check": [
        ("is_uniform", "ramp", lambda S, s: Check.is_uniform(S)),
        ("is_duplicates", "mixed", lambda S, s: Check.is_duplicates(S)),
        ("is_infinite", "mixed", lambda S, s: Check.is_infinite(S)),
        ("to_array_noop", "mixed", lambda S, s: Check.to_array(S, ndim=1)),
        ("ctas_copy", "mixed",
         lambda S, s: Check_Time_and_Signal(S, ndim={1}, method="scale", need_T=False)),
        ("ctas_no_copy", "mixed",
         lambda S, s: Check_Time_and_Signal(S, ndim={1}, method="scale",
                                            need_T=False, copy=False)),
    ],
}


_XREF = np.linspace(-1.0, 1.0, 64)
_YREF = np.sin(3.0 * _XREF)


def _count_chunks(S: np.ndarray, chunk_size: int | None) -> tuple[int, int]:
    n = 0
    total = 0
    for c in Chunk.chunk(S, chunk_size=chunk_size):
        n += 1
        total += int(c.size)
    return n, total


def _chunked_map(S: np.ndarray) -> float:
    out = Chunk.chunked_map(lambda c: c * 2.0 + 1.0, S, chunk_size=1048576)
    return float(out[-1])


def _fill_mem(S: np.ndarray) -> float:
    dst = np.empty_like(S)
    Chunk.chunked_fill(dst, S, chunk_size=1048576)
    return float(dst[-1])


def _fill_disk(S: np.ndarray) -> float:
    """往临时 memmap 流式写一遍，再用官方 ``drop_memmap`` 关句柄+删文件。

    注意：这里必须走 ``drop_memmap(force=True)``，否则 ``_MM_FILES`` 仍持有映射，
    Windows 上文件删不掉、脏页留在 RSS 里，后续档位的 ΔRSS 会被污染。
    """
    from Modal_Decomposition.Base import drop_memmap, temp_memmap

    fp = temp_memmap(S.shape, S.dtype)
    if fp is None:                     # pragma: no cover
        return float("nan")
    Chunk.fill_memmap(fp, S)
    v = float(fp[-1])
    drop_memmap(fp, force=True)
    del fp
    gc.collect()
    return v


# --------------------------------------------------------------------------- #
# 单格测量
# --------------------------------------------------------------------------- #
def measure(fn: Callable[[], Any], size_mib: int) -> dict:
    repeat = 2 if size_mib >= 500 else 3
    rss0 = B.rss_mib()
    tracemalloc.start()
    try:
        res = B.timeit(fn, repeat=repeat, warmup=1)
        alloc_peak = tracemalloc.get_traced_memory()[1] / B.MiB
    finally:
        tracemalloc.stop()
    rss1 = B.rss_mib()
    return {
        "best_ms": round(res["best_ms"], 3),
        "median_ms": round(res["median_ms"], 3),
        "alloc_peak_mib": round(alloc_peak, 1),
        "rss_delta_mib": round(rss1 - rss0, 1),
        "peak_process_mib": round(B.peak_rss_mib(), 1),
        "repeat": res["repeat"],
    }


def run_tool(tool: str, sizes: tuple[int, ...]) -> list[dict]:
    rows: list[dict] = []
    for op_name, kind, fn in REGISTRY[tool]:
        for size_mib in sizes:
            S = B.make_signal(size_mib, kind=kind)
            nbytes = S.nbytes
            row = {
                "tool": tool, "op": op_name, "input": kind,
                "size_mib": size_mib, "n": int(S.size), "nbytes": nbytes,
                "dtype": str(S.dtype),
            }
            try:
                row.update(measure(lambda: fn(S, size_mib), size_mib))
                row["mb_per_s"] = round(size_mib / (row["best_ms"] / 1e3), 2) if row["best_ms"] else None
                row["ok"] = 1
                row["error"] = ""
            except MemoryError as exc:
                row.update({"best_ms": None, "median_ms": None, "alloc_peak_mib": None,
                            "rss_delta_mib": None, "peak_process_mib": round(B.peak_rss_mib(), 1),
                            "repeat": 0, "mb_per_s": None, "ok": 0,
                            "error": f"MemoryError: {exc}"[:120]})
            except Exception as exc:        # noqa: BLE001 - 规模轴要记录而不是中断
                row.update({"best_ms": None, "median_ms": None, "alloc_peak_mib": None,
                            "rss_delta_mib": None, "peak_process_mib": round(B.peak_rss_mib(), 1),
                            "repeat": 0, "mb_per_s": None, "ok": 0,
                            "error": f"{type(exc).__name__}: {exc}"[:120]})
            rows.append(row)
            print(f"  {tool:13s} {op_name:24s} {size_mib:4d} MiB  "
                  f"{'ok ' if row['ok'] else 'ERR'} {row['best_ms']!s:>10} ms  "
                  f"alloc {row['alloc_peak_mib']!s:>8} MiB", flush=True)
            del S
            gc.collect()
    return rows


# --------------------------------------------------------------------------- #
# 图
# --------------------------------------------------------------------------- #
def plot_tool(tool: str, rows: list[dict]) -> Path:
    B.plt_style()
    import matplotlib.pyplot as plt

    ops = sorted({r["op"] for r in rows}, key=lambda o: [x["op"] for x in rows].index(o))
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))

    for op in ops:
        sub = [r for r in rows if r["op"] == op and r["ok"]]
        if not sub:
            continue
        xs = [r["size_mib"] for r in sub]
        axes[0].plot(xs, [r["best_ms"] for r in sub], "o-", label=op, ms=3.5)
        axes[1].plot(xs, [r["alloc_peak_mib"] for r in sub], "o-", label=op, ms=3.5)

    axes[0].set_xlabel("input size (MiB, 1 MiB = 2^20 B)")
    axes[0].set_ylabel("best wall time (ms)")
    axes[0].set_yscale("log")
    axes[0].set_xscale("log")
    axes[0].set_title(f"{tool}: wall time vs size")
    axes[1].set_xlabel("input size (MiB)")
    axes[1].set_ylabel("tracemalloc peak (MiB)")
    axes[1].set_xscale("log")
    axes[1].set_title(f"{tool}: allocation peak vs size")
    axes[1].plot([1, 600], [1, 600], "k--", lw=0.8, label="1x input")
    axes[0].legend(fontsize=6.5, ncol=1)
    axes[1].legend(fontsize=6.5, ncol=1)
    fig.tight_layout()
    return B.savefig(fig, B.FIG / "size_axis" / f"{tool}.png")


def rebuild_summary() -> Path:
    """把 ``table/size_axis/<tool>_<op>.csv`` 合并成 ``summary.csv``（按 tool/op/size 排序）。

    summary 是**派生文件**：只由逐 op 的 csv 拼出来，保证"跑一个工具"不会把别的
    工具的行冲掉。
    """
    import csv as _csv

    rows: list[dict] = []
    files = sorted(p for p in (B.TABLE / "size_axis").glob("*.csv") if p.name != "summary.csv")
    for p in files:
        with open(p, newline="", encoding="utf-8") as fh:
            rows.extend(_csv.DictReader(fh))
    rows.sort(key=lambda r: (r["tool"], r["op"], int(r["size_mib"])))
    out = B.TABLE / "size_axis" / "summary.csv"
    if rows:
        B.write_csv(out, rows, list(rows[0].keys()))
    return out


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tool", default="all",
                    help=f"one of {sorted(REGISTRY)} or 'all'")
    ap.add_argument("--sizes", type=int, nargs="*", default=list(B.SIZES_MIB))
    ap.add_argument("--export", action="store_true")
    ap.add_argument("--rebuild-summary", action="store_true",
                    help="只由逐 op csv 重建 summary.csv")
    args = ap.parse_args()

    if args.rebuild_summary:
        p = rebuild_summary()
        print(f"rebuilt {p}")
        return 0

    tools = sorted(REGISTRY) if args.tool == "all" else [args.tool]
    sizes = tuple(sorted(args.sizes))

    all_rows: list[dict] = []
    for tool in tools:
        if tool not in REGISTRY:
            raise SystemExit(f"unknown tool {tool!r}; choose from {sorted(REGISTRY)}")
        print(f"== {tool} ==", flush=True)
        rows = run_tool(tool, sizes)
        all_rows.extend(rows)
        if args.export:
            for op in sorted({r["op"] for r in rows}):
                B.write_csv(B.TABLE / "size_axis" / f"{tool}_{op}.csv",
                            [r for r in rows if r["op"] == op])
            p = plot_tool(tool, rows)
            print(f"  -> {p}")

    if args.export:
        p = rebuild_summary()
        print(f"  -> {p} ({len(all_rows)} rows in this run)")
    else:
        B.print_table(all_rows, ["tool", "op", "size_mib", "n", "best_ms", "alloc_peak_mib", "ok"])
    return 0 if all(r["ok"] for r in all_rows) else 1


if __name__ == "__main__":
    sys.exit(main())
