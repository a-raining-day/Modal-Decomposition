"""终验：逐条核对本轮目标的 4 个可验证条件，打印 PASS/FAIL。"""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

ROOT = Path(".").resolve()
sys.path.insert(0, str(ROOT / "tests"))
import _bench as B  # noqa: E402

FAILS: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {label}" + (f"  — {detail}" if detail else ""))
    if not ok:
        FAILS.append(label)


def n_files(base: str, pattern: str) -> int:
    return len([p for p in (ROOT / base).rglob(pattern) if "__pycache__" not in str(p)])


# ---- ① 每个 Utils 工具都有深度测试 ------------------------------------------
UTILS = ["Check", "Envelope", "Hilbert", "Mirror", "Monotonicity", "Peaks", "Poly", "Slepian"]
BASE_TOOLS = ["Cache", "Chunk", "CPU", "Memory", "Seed"]
missing = [t for t in UTILS if not (ROOT / "tests/utils" / f"test_{t}_deep.py").exists()]
missing += [t for t in BASE_TOOLS if not (ROOT / "tests/base" / f"test_{t}_deep.py").exists()]
check("① Utils 8 工具 + Base 5 工具均有深度测试", not missing, f"缺: {missing}" if missing else "13 份")

# Utils 下是否还有未被覆盖的公开模块
utils_mods = sorted(p.stem for p in (ROOT / "src/Modal_Decomposition/Utils").glob("*.py")
                    if p.stem != "__init__")
uncov = [m for m in utils_mods if m not in UTILS]
check("① Utils/*.py 全部被覆盖", not uncov, f"模块: {utils_mods}")

# ---- ② 规模轴 6 档 × 低成本工具 ---------------------------------------------
rows = list(csv.DictReader(open(ROOT / "table/size_axis/summary.csv", encoding="utf-8")))
sizes = sorted({int(r["size_mib"]) for r in rows})
want_sizes = [1, 50, 100, 200, 500, 600]
tools = sorted({r["tool"] for r in rows})
check("② 规模档位 = 1/50/100/200/500/600 MiB", sizes == want_sizes, str(sizes))
check("② 规模轴全部成功", all(r["ok"] == "1" for r in rows),
      f"{len(rows)} 格 / {len(tools)} 工具 / 失败 {sum(r['ok'] != '1' for r in rows)}")
per = {t: len([r for r in rows if r["tool"] == t]) for t in tools}
check("② 每个工具都跑满 6 档", all(v % 6 == 0 for v in per.values()), str(per))

# ---- ③ 材料分级落位 ---------------------------------------------------------
outside_png = [p for p in ROOT.rglob("*.png")
               if "fig" not in p.relative_to(ROOT).parts[:1]
               and not any(x in p.parts for x in (".venv", ".scratch", "build", "dist",
                                                  "__pycache__", "ref", "explore"))]
check("③ 图片全部在 fig/ 下", not outside_png, f"越界: {[str(p) for p in outside_png][:3]}")
outside_csv = [p for p in ROOT.rglob("*.csv")
               if p.relative_to(ROOT).parts[0] != "table"
               and not any(x in p.parts for x in (".venv", ".scratch", "build", "dist",
                                                  "__pycache__", "ref", "explore"))]
check("③ csv 全部在 table/ 下", not outside_csv, f"越界: {[str(p) for p in outside_csv][:3]}")
xlsx = list((ROOT / "table").rglob("*.xlsx"))
check("③ table/index.xlsx 存在", any(p.name == "index.xlsx" for p in xlsx), str(len(xlsx)))
fig_groups = {p.relative_to(ROOT / "fig").parts[0] for p in (ROOT / "fig").rglob("*.png")}
check("③ fig/ 已分级", fig_groups >= {"methods", "utils", "base", "size_axis", "comparison"},
      str(sorted(fig_groups)))
doc_groups = {p.relative_to(ROOT / "docs/test").parts[0] for p in (ROOT / "docs/test").rglob("*.md")}
check("③ docs/test 已分级", {"methods", "utils", "base"} <= doc_groups, str(sorted(doc_groups)))

# ---- ④ 链接与索引 -----------------------------------------------------------
r = subprocess.run([sys.executable, "tests/tools/check_links.py"],
                   capture_output=True, text=True, encoding="utf-8")
check("④ 链接/资源自检通过", r.returncode == 0, r.stdout.strip().splitlines()[-1])

# 每份深度报告都链接到自己的 csv 与 png（抽查 6 个）
DOC_MAP = {
    "utils/Check_Engine_Report.md": ["table/utils/check/", "fig/utils/check/"],
    "utils/Peaks_Engine_Report.md": ["table/utils/peaks/", "fig/utils/peaks/"],
    "utils/Slepian_Engine_Report.md": ["table/utils/slepian/", "fig/utils/slepian/"],
    "base/Chunk_Engine_Report.md": ["table/base/chunk/", "fig/base/chunk/"],
    "base/Cache_Engine_Report.md": ["table/base/cache/", "fig/base/cache/"],
    "base/Memory_Engine_Report.md": ["table/base/memory/"],
}
bad = []
for rel, needles in DOC_MAP.items():
    text = (ROOT / "docs/test" / rel).read_text(encoding="utf-8")
    for nd in needles:
        if nd not in text:
            bad.append(f"{rel} 缺 {nd}")
check("④ 报告里给出数据/图路径", not bad, "; ".join(bad) or "6 份抽查")

# ---- 测试总数 ---------------------------------------------------------------
print("\n文件统计:")
for label, base, pat in (("docs md", "docs", "*.md"), ("fig png", "fig", "*.png"),
                         ("table csv", "table", "*.csv"), ("tests py", "tests", "*.py")):
    print(f"  {label:10s} {n_files(base, pat)}")

print("\n结果:", "ALL PASS" if not FAILS else f"{len(FAILS)} FAIL -> {FAILS}")
sys.exit(0 if not FAILS else 1)
