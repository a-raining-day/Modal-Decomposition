"""
Summarize the **three-way VMD** comparison (MD-VMD / vmdpy / PySDKit) into a
per-case median-wall table, a quality table and a raw-JSON snapshot.

This is the replacement for ``summarize_vmd.py`` on grids that were run with
``--repeats 3`` (the older summarizer points at ``results/vmd/`` and assumes a
single shot, so it reports unsmoothed raw walls).

Outputs
-------
- ``tests/comparison/results/vmd_pysdkit/summary.md``   (human-readable)
- ``docs/VMD_Native_vs_vmdpy_vs_PySDKit_Results.json``  (raw, machine-readable)

Usage::

    python tests/comparison/summarize_vmd_pysdkit.py
    python tests/comparison/summarize_vmd_pysdkit.py --res tests/comparison/results/vmd_pysdkit
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[1]

IMPLS = ["MD-VMD", "vmdpy", "PySDKit"]
CASES = ["A", "B", "C"]
NS = (256, 1024, 4096, 16384, 65536)


def _load(res: Path) -> tuple[dict, dict]:
    """Return ({ (impl,case,n): [wall_s,...] }, { (impl,case,n): metrics })."""
    walls: dict[tuple, list] = {}
    raw = res / "timing_raw.csv"
    if raw.exists():
        for r in csv.DictReader(raw.open(encoding="utf-8")):
            try:
                key = (r["impl"], r["case"], int(r["n"]))
                walls.setdefault(key, []).append(float(r["wall_s"]))
            except (TypeError, ValueError):
                continue

    metrics: dict[tuple, dict] = {}
    mp = res / "timing_metrics.json"
    if mp.exists():
        for m in json.loads(mp.read_text(encoding="utf-8")):
            metrics[(m["impl"], m["case"], m["n"])] = m
    return walls, metrics


def _fmt(v) -> str:
    return "-" if v is None else f"{v:.4f}"


def _ratio(num, den) -> str:
    if not num or not den:
        return "-"
    return f"{num / den:.1f}x"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", type=Path, default=_HERE / "results" / "vmd_pysdkit")
    args = ap.parse_args()
    res: Path = args.res

    walls, metrics = _load(res)
    if not walls:
        print(f"no data under {res}", file=sys.stderr)
        return 1

    ns_present = sorted({k[2] for k in walls})
    cases_present = [c for c in CASES if any(k[1] == c for k in walls)]

    med = {k: statistics.median(v) for k, v in walls.items()}

    md: list[str] = [
        "# VMD three-way summary (MD-VMD / vmdpy / PySDKit)",
        "",
        "Parity config for all three columns: `alpha=2000`, `tau=0`, `DC=False`, "
        "uniform frequency init, `tol=1e-6`, `max_iter=500`; "
        "K: case A=3, B=3, C=4. Values are the **median of 3 repeats** "
        "(same process, warm-up excluded).",
        "",
    ]
    for case in cases_present:
        md += [
            f"## case {case}",
            "",
            "| n | MD-VMD (s) | vmdpy (s) | PySDKit (s) | vmdpy/MD | PySDKit/MD | PySDKit/vmdpy |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for n in ns_present:
            a, b, c = (med.get((i, case, n)) for i in IMPLS)
            md.append(
                f"| {n} | {_fmt(a)} | {_fmt(b)} | {_fmt(c)} "
                f"| {_ratio(b, a)} | {_ratio(c, a)} | {_ratio(c, b)} |"
            )
        md.append("")
        md += [
            "Quality:",
            "",
            "| impl | rows | recon max abs err | orthogonality idx | mode recovery (best abs corr) |",
            "|---|---:|---:|---:|---|",
        ]
        for impl in IMPLS:
            for n in reversed(ns_present):
                m = metrics.get((impl, case, n))
                if not m:
                    continue
                rec = m.get("mode_recovery") or {}
                corr = ", ".join(
                    f"{k}={v['best_abs_corr']:.4f}" if v.get("best_abs_corr") is not None
                    else f"{k}=n/a"
                    for k, v in rec.items()
                )
                md.append(
                    f"| {impl} (n={n}) | {m.get('n_rows')} "
                    f"| {m.get('recon_max_abs_err'):.2e} "
                    f"| {m.get('orthogonality_index'):.2e} | {corr or '-'} |"
                )
                break
        md.append("")

    out_md = res / "summary.md"
    out_md.write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))

    payload = {
        "impls": IMPLS,
        "parity": {
            "alpha": 2000, "tau": 0.0, "DC": False, "init": "uniform",
            "tol": 1e-6, "max_iter": 500, "K": {"A": 3, "B": 3, "C": 4},
        },
        "median_wall_s": [
            {"impl": i, "case": c, "n": n, "median_s": med.get((i, c, n)),
             "repeats": len(walls.get((i, c, n), []))}
            for c in cases_present for n in ns_present for i in IMPLS
        ],
        "metrics": [
            {"impl": i, "case": c, "n": n, **metrics[(i, c, n)]}
            for c in cases_present for n in ns_present for i in IMPLS
            if (i, c, n) in metrics
        ],
    }
    docs = _REPO / "docs"
    docs.mkdir(exist_ok=True)
    out_json = docs / "VMD_Native_vs_vmdpy_vs_PySDKit_Results.json"
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {out_md}")
    print(f"wrote {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
