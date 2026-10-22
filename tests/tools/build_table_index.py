"""把 `table/` 下所有 csv 汇总成 `table/index.xlsx`（每个 csv 一个 sheet + 索引页）。

用法::

    .\\.venv\\Scripts\\python.exe tests/tools/build_table_index.py

约定
----
* sheet 名由相对路径压缩而来（`methods/vmd/timing_raw.csv` → `methods_vmd_timing_raw`），
  超过 Excel 的 31 字符上限时截断并加序号；
* `INDEX` 页列出：相对路径、sheet 名、行列数、列名、一句话说明（来自本文件的
  ``DESCRIPTIONS`` 前缀表，没有匹配就留空）；
* 数值列尽量保留原样，不做类型推断（csv 里的字符串如 `None` 保持文本）。
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
TABLE = ROOT / "table"
OUT = TABLE / "index.xlsx"

#: path 前缀 -> 说明
DESCRIPTIONS: dict[str, str] = {
    "size_axis/summary": "规模轴总表（8 工具 × 6 档位，1–600 MiB）",
    "size_axis/monotonicity": "Monotonicity 规模轴",
    "size_axis/chunk": "Chunk 规模轴",
    "size_axis/check": "Check 规模轴",
    "size_axis/peaks": "Peaks 规模轴",
    "size_axis/mirror": "Mirror 规模轴",
    "size_axis/poly": "Poly 规模轴",
    "size_axis/hilbert": "Hilbert 规模轴",
    "size_axis/envelope": "Envelope 规模轴",
    "utils/check": "Check 契约矩阵",
    "utils/monotonicity": "Monotonicity 半群/真值表/穷举/分块不变性",
    "utils/poly": "fast_poly vs polyfit",
    "utils/peaks": "Peaks 三后端一致性与计时",
    "utils/envelope": "Envelope 四模式精度",
    "utils/hilbert": "Hilbert 三后端一致性与契约矩阵",
    "utils/mirror": "Mirror 五模式几何",
    "utils/slepian": "Slepian 三后端一致性与 N 扫描",
    "base/chunk": "Chunk 块长表与用例矩阵",
    "base/cache": "Cache 每次调用耗时",
    "base/cpu": "set_allocable 取值表",
    "base/memory": "内存预算表",
    "methods/": "分解法逐 run 计时（timing_raw.csv）",
    "memory/": "分解法内存轴汇总",
    "comparison/": "跨方法对照汇总",
}


def describe(rel: str) -> str:
    for prefix, desc in DESCRIPTIONS.items():
        if rel.startswith(prefix):
            return desc
    return ""


def sheet_name(rel: str, used: set[str]) -> str:
    name = re.sub(r"[^0-9A-Za-z_]+", "_", rel.replace(".csv", "")).strip("_")
    if len(name) > 31:
        name = name[:28] + f"_{sum(1 for u in used if u.startswith(name[:20])) + 1}"
    base, i = name[:31], 1
    while name in used or not name:
        i += 1
        name = f"{base[:28]}_{i}"
    used.add(name)
    return name


def main() -> int:
    files = sorted(p for p in TABLE.rglob("*.csv"))
    if not files:
        print("no csv found under table/")
        return 1

    used: set[str] = {"INDEX"}
    index_rows: list[dict] = []

    with pd.ExcelWriter(OUT, engine="xlsxwriter") as xw:
        for path in files:
            rel = path.relative_to(TABLE).as_posix()
            try:
                df = pd.read_csv(path, dtype=str, keep_default_na=False)
            except Exception as exc:                  # noqa: BLE001
                print(f"!! skip {rel}: {exc}")
                continue
            sheet = sheet_name(rel, used)
            df.to_excel(xw, sheet_name=sheet, index=False)
            index_rows.append({
                "csv": rel,
                "sheet": sheet,
                "rows": len(df),
                "cols": len(df.columns),
                "columns": ", ".join(df.columns),
                "description": describe(rel),
            })
            print(f"  {sheet:31s} <- {rel}  ({len(df)}×{len(df.columns)})")

        pd.DataFrame(index_rows).to_excel(xw, sheet_name="INDEX", index=False)

        # 列宽美化（INDEX 页）
        book = xw.book
        ws = book.get_worksheet_by_name("INDEX")
        if ws is not None:
            for col, width in enumerate([46, 32, 8, 8, 60, 44]):
                ws.set_column(col, col, width)
            ws.freeze_panes(1, 0)

    total_rows = sum(r["rows"] for r in index_rows)
    print(f"\nwrote {OUT}  ({len(index_rows)} sheets, {total_rows} data rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
