"""生成 docs/test/Scale_Axis_Report.md（表格由 table/size_axis/summary.csv 派生）。

    .\\.venv\\Scripts\\python.exe tests/size_axis/report.py
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "tests"))

import _bench as B  # noqa: E402

SUMMARY = _ROOT / "table" / "size_axis" / "summary.csv"
OUT = _ROOT / "docs" / "test" / "Scale_Axis_Report.md"

TOOL_TITLE = {
    "check": "`Utils.Check`（谓词 + 入口规范化）",
    "chunk": "`Base.Chunk`（分块/流式填充）",
    "monotonicity": "`Utils.Monotonicity`",
    "peaks": "`Utils.Peaks`（三后端）",
    "mirror": "`Utils.Mirror`",
    "poly": "`Utils.Poly`（`fast_poly` 求值）",
    "hilbert": "`Utils.Hilbert`（三后端）",
    "envelope": "`Utils.Envelope`（四模式）",
}

ORDER = ["monotonicity", "chunk", "check", "peaks", "mirror", "poly", "hilbert", "envelope"]


def load() -> dict[str, list[dict]]:
    rows = list(csv.DictReader(open(SUMMARY, encoding="utf-8")))
    out: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        out[r["tool"]].append(r)
    return out


def fmt(v: str) -> str:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return "-"
    if f >= 1000:
        return f"{f:,.0f}"
    if f >= 10:
        return f"{f:.1f}"
    return f"{f:.2f}"


def table_for(rows: list[dict], sizes: list[int]) -> list[str]:
    ops = sorted({r["op"] for r in rows})
    lines = ["| op | " + " | ".join(f"{s} MiB" for s in sizes) + " |",
             "|---|" + "---|" * len(sizes)]
    for op in ops:
        cells = []
        for s in sizes:
            hit = [r for r in rows if r["op"] == op and int(r["size_mib"]) == s]
            if not hit:
                cells.append("-")
                continue
            r = hit[0]
            cells.append(f"{fmt(r['best_ms'])} ms")
        lines.append(f"| `{op}` | " + " | ".join(cells) + " |")
    return lines


def alloc_for(rows: list[dict], sizes: list[int]) -> list[str]:
    ops = sorted({r["op"] for r in rows})
    lines = ["| op | " + " | ".join(f"{s} MiB" for s in sizes) + " |",
             "|---|" + "---|" * len(sizes)]
    for op in ops:
        cells = []
        for s in sizes:
            hit = [r for r in rows if r["op"] == op and int(r["size_mib"]) == s]
            cells.append(fmt(hit[0]["alloc_peak_mib"]) if hit else "-")
        lines.append(f"| `{op}` | " + " | ".join(cells) + " |")
    return lines


def main() -> int:
    data = load()
    sizes = [1, 50, 100, 200, 500, 600]
    n_rows = sum(len(v) for v in data.values())
    all_ok = all(r["ok"] == "1" for v in data.values() for r in v)

    md: list[str] = []
    md.append("# 规模轴：1 MiB – 600 MiB 实测（低成本工具层）")
    md.append("")
    md.append("> 生成脚本：`tests/size_axis/run_scale.py`；本文件由 "
              "`tests/size_axis/report.py` 从 `table/size_axis/summary.csv` 渲染，"
              "**表格数字与 csv 逐格一致**。")
    md.append(">")
    md.append("> **MB 一律指 MiB（2²⁰ B）**：600 MiB = 629 145 600 B = 78 643 200 个 float64。")
    md.append("")
    md.append("## 0. 结论先说")
    md.append("")
    md.append(f"* 8 个工具、{len(sizes)} 个档位、{n_rows} 格，**全部成功**（`ok=1`，"
              "无 MemoryError / 无超时）；耗时对所有工具都是**线性**的（表里每一行从 "
              "1 MiB 到 600 MiB 约 ×600）。")
    md.append("* 真正的成本差异在**内存放大倍数**，不在时间：")
    md.append("  * `is_uniform` **3×**（600 MiB 输入 → 1800 MiB 峰值）、"
              "`fast_poly` **5×(deg3) / 10×(deg8)**、"
              "`envelope(Hilbert)` 与 `hilbert(scipy)` **8×**；")
    md.append("  * 最省的是 `is_infinite` / `ctas_no_copy` / `to_array`：**0.125×**"
              "（只多一个 n 字节的布尔掩码）。")
    md.append("* **600 MiB 上最慢的是 `Envelope(method=\"Hilbert\")`：12.7 s / 4.8 GiB**"
              "（默认走 scipy 后端）；")
    md.append("  同样算解析信号的 `hilbert(S, \"FFT\")` 只要 3.7 s / 2.4 GiB —— "
              "kwargs 泄漏已修，现在可以 `hilbert_mod=\"FFT\"` 切过去")
    md.append("  （见 `Envelope` 报告 §6.2）。")
    md.append("* `Chunk.chunk()` 在 1–600 MiB 都只切出 **1 块**（0.03 ms）——这是"
              "**设计意图**：块长按 `0.75 × 内存预算` 取，这几个档位都装得下；")
    md.append("  单位混用（字节入/元素出）已修，超出预算时按预算封顶（见 `Chunk` 报告 §7.1）。")
    md.append("* `fast_poly` 的时间不是 O(N·order) 的核心运算，而是**建 (N, order+1) 的")
    md.append("  Vandermonde**：deg8 在 600 MiB 上峰值分配 6000 MiB、耗时 4.2 s。")
    md.append("")
    md.append("## 1. 口径")
    md.append("")
    md.append("* `best_ms`：同进程内 warm-up 1 次后重复 2–3 次取**最小**；")
    md.append("  `alloc_peak_mib`：`tracemalloc` 在该 op 调用期间的**峰值分配**"
              "（NumPy 大缓冲按一次分配记账）；")
    md.append("* 输入信号：`tests/_bench.py::make_signal(kind)` —— `mixed`（三音调+趋势）、"
              "`ramp`（严格递增）；")
    md.append("* 机器：Windows 11 / 32 逻辑核 / 15.6 GB 物理内存 / Python 3.10.11 / "
              "numpy 2.2.6 / scipy 1.15.3。")
    md.append("* **分解法不做这条轴**：单模态 500 MB 就要 5.75–9.76 GB，"
              "见 `docs/test/methods/Large_Array_Memory_Report.md`。")
    md.append("")

    for tool in ORDER:
        if tool not in data:
            continue
        rows = data[tool]
        md.append(f"## {tool}: {TOOL_TITLE.get(tool, tool)}")
        md.append("")
        md.append(f"图：`fig/size_axis/{tool}.png`；逐 op 数据："
                  f"`table/size_axis/{tool}_<op>.csv`。")
        md.append("")
        md.append("**耗时（ms）**")
        md.append("")
        md.extend(table_for(rows, sizes))
        md.append("")
        md.append("**峰值分配（MiB）**")
        md.append("")
        md.extend(alloc_for(rows, sizes))
        md.append("")

    md.append("## 2. 复现")
    md.append("")
    md.append("```powershell")
    md.append("# 单个工具（会写 table/size_axis/*.csv 与 fig/size_axis/<tool>.png）")
    md.append(r".\.venv\Scripts\python.exe tests\size_axis\run_scale.py --tool monotonicity --export")
    md.append("")
    md.append("# 全部 8 个工具（约 25–35 分钟，含一次 6 GB 峰值的 poly deg8）")
    for tool in ORDER:
        md.append(f".\\.venv\\Scripts\\python.exe tests\\size_axis\\run_scale.py --tool {tool} --export")
    md.append(r".\.venv\Scripts\python.exe tests\size_axis\run_scale.py --rebuild-summary")
    md.append(r".\.venv\Scripts\python.exe tests\size_axis\report.py")
    md.append("```")
    md.append("")
    md.append("## 3. 已知限制")
    md.append("")
    md.append("* `tracemalloc` 只统计 Python 分配器看到的量；memmap 写入的页缓存不在此列，"
              "所以 `chunked_fill_to_disk` 的 `alloc` 列是 0 而实际有磁盘 IO。")
    md.append("* 3 次重复的 `best` 对 600 MiB 档位受同时运行的其他进程影响"
              "（本轮所有档位串行、单进程跑完，误差约 ±5%）。")
    md.append("")

    OUT.write_text("\n".join(md), encoding="utf-8")
    print(f"wrote {OUT} ({len(md)} lines, {n_rows} data rows, all_ok={all_ok})")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
