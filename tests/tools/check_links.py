"""链接与资源引用自检。

检查两类东西：

1. **markdown 内联链接** —— ``[text](target)`` 的 ``target`` 必须存在（跳过 http/#）；
2. **代码里点名的资源** —— 反引号里的 ``fig/…png`` / ``table/…csv`` /
   ``docs/…md`` / ``tests/…json`` 路径必须存在。

用法::

    .\\.venv\\Scripts\\python.exe tests/tools/check_links.py        # 只报错
    .\\.venv\\Scripts\\python.exe tests/tools/check_links.py -v     # 也列通过项

退出码 0 = 全部可解析，1 = 有失效引用。
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

SKIP_DIRS = {".git", ".venv", "build", "dist", "__pycache__", ".pytest_cache",
             ".pytest_tmp", ".idea", ".vscode", "node_modules", ".scratch"}

#: 只在仓库根跳过的目录（注意：`docs/explore/` 是本项目文档，必须检查）。
SKIP_ROOT_DIRS = {"ref", "explore"}

LINK_RE = re.compile(r"\]\(([^)\s]+)\)")
CODE_SPAN_RE = re.compile(r"`([^`]+)`")

#: 只有这些前缀的代码内路径会被当作"资源引用"去校验。
ASSET_PREFIXES = ("fig/", "table/", "docs/", "tests/")

#: 明确允许不存在的引用（历史文件、已删除报告等）。写清楚理由。
ALLOW_MISSING = {
    # 表格里描述的"已删除"历史文件，不是现行资源
    "docs/EMD_Large_Signal_Memory_Report.md",
    "docs/EMD_faster_Branch_Comparison_Report.md",
}


def _iter_markdown():
    import os

    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        if Path(dirpath) == ROOT:
            dirnames[:] = [d for d in dirnames if d not in SKIP_ROOT_DIRS]
        for name in filenames:
            if name.endswith(".md"):
                yield Path(dirpath) / name


def check(verbose: bool = False) -> int:
    bad: list[str] = []
    n_links = n_assets = 0

    for md in _iter_markdown():
        text = md.read_text(encoding="utf-8", errors="surrogateescape")
        rel_md = md.relative_to(ROOT).as_posix()
        for m in LINK_RE.finditer(text):
            target = m.group(1)
            if target.startswith(("http://", "https://", "mailto:", "#", "<")):
                continue
            base = target.partition("#")[0]
            if not base:
                continue
            n_links += 1
            if (md.parent / base).exists():
                if verbose:
                    print(f"  ok  {rel_md} -> {target}")
            else:
                bad.append(f"[link] {rel_md} -> {target}")

        for m in CODE_SPAN_RE.finditer(text):
            span = m.group(1).strip()
            if not span.startswith(ASSET_PREFIXES):
                continue
            if any(ch in span for ch in "*?{}$<>…()") or " " in span or "::" in span:
                continue                      # 通配/占位（<op>、{a,b}、…），跳过
            path = span.split("#")[0]
            if path in ALLOW_MISSING:
                continue
            n_assets += 1
            if (ROOT / path).exists():
                if verbose:
                    print(f"  ok  {rel_md} -> {span}")
            else:
                bad.append(f"[asset] {rel_md} -> {span}")

    print(f"checked {n_links} markdown links, {n_assets} asset references")
    for b in bad:
        print("  BROKEN", b)
    print("RESULT:", "OK" if not bad else f"{len(bad)} broken")
    return 0 if not bad else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--verbose", action="store_true")
    return check(ap.parse_args().verbose)


if __name__ == "__main__":
    sys.exit(main())
