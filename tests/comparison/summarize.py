"""
Summarize raw benchmark records into markdown tables + CSV summaries.

Reads ``results/timing_raw.csv`` (+ ``results/timing_metrics.json``) and
``results/memory/*.json`` and writes:

* ``results/summary_timing.md``  - median wall time per (case, n, impl),
  speedup ratios and quality metrics;
* ``results/summary_memory.md``  - per-cell memory-grid status table;
* ``results/timing_median.csv``  - machine-readable median table;
* ``results/memory_flat.csv``    - flattened memory grid records.

Ratios are expressed as "other / MD-EMD", i.e. how many times faster/slower
PyEMD and PySDKit are than the MD-EMD facade on the same cell.
"""

from __future__ import annotations

import csv
import json
import statistics
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_RESULTS = _HERE / "results"

IMPLS = ["MD-EMD", "PyEMD", "PySDKit"]


def _med(vals):
    vals = [v for v in vals if v is not None]
    return statistics.median(vals) if vals else None


def summarize_timing() -> None:
    raw = _RESULTS / "timing_raw.csv"
    if not raw.exists():
        print(f"(no {raw} yet - timing axis not run)")
        return
    rows = list(csv.DictReader(raw.open(encoding="utf-8")))

    cells: dict[tuple, dict] = {}
    for r in rows:
        key = (r["impl"], r["case"], int(r["n"]))
        wall = r.get("wall_s")
        try:
            wall = float(wall)
        except (TypeError, ValueError):
            wall = None
        cells.setdefault(key, []).append(wall)

    metrics = {}
    mpath = _RESULTS / "timing_metrics.json"
    if mpath.exists():
        for m in json.loads(mpath.read_text(encoding="utf-8")):
            metrics[(m["impl"], m["case"], m["n"])] = m

    order = [(impl, c, n)
             for c in ("A", "B", "C")
             for n in (256, 1024, 4096, 16384, 65536)
             for impl in IMPLS]

    lines = ["# Timing summary (regular lengths)", "",
             "Median wall time in seconds (3 repeats; same process/machine, "
             "implementation order round-robined). Ratios: PyEMD / MD-EMD and "
             "PySDKit / MD-EMD (>1 = slower than MD-EMD, <1 = faster).",
             ""]
    for c in ("A", "B", "C"):
        ns = sorted({n for (_, cc, n) in cells if cc == c})
        if not ns:
            continue
        lines.append(f"## Case {c}")
        lines.append("")
        lines.append("| n | impl | wall (s) | ratio vs MD-EMD | n_rows | recon max abs | IO |")
        lines.append("|---|------|---------:|-----------:|-------:|--------------:|---:|")
        for n in ns:
            md_wall = _med(cells.get(("MD-EMD", c, n), []))
            for impl in IMPLS:
                wall = _med(cells.get((impl, c, n), []))
                if wall is None:
                    continue
                ratio = (wall / md_wall) if (impl != "MD-EMD" and md_wall) else 1.0
                met = metrics.get((impl, c, n), {})
                lines.append(
                    f"| {n} | {impl} | {wall:.4f} | "
                    f"{'' if impl == 'MD-EMD' else f'{ratio:.2f}x'} | "
                    f"{met.get('n_rows', '')} | {met.get('recon_max_abs_err', '')} | "
                    f"{met.get('orthogonality_index', '')} |"
                )
        lines.append("")

    ( _RESULTS / "summary_timing.md").write_text("\n".join(lines), encoding="utf-8")

    with open(_RESULTS / "timing_median.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["impl", "case", "n", "median_wall_s", "n_measures"])
        for key in order:
            vals = cells.get(key)
            if not vals:
                continue
            w.writerow([key[0], key[1], key[2],
                        round(_med(vals), 6) if _med(vals) else "", len(vals)])
    print(f"wrote {_RESULTS / 'summary_timing.md'} and timing_median.csv")


def summarize_memory() -> None:
    mem = _RESULTS / "memory"
    flat_path = _RESULTS / "memory_flat.csv"
    if not mem.is_dir():
        print(f"(no {mem} yet - memory axis not run)")
        return
    flat = []
    for impl in IMPLS:
        p = mem / f"{impl}.json"
        if not p.exists():
            continue
        for rec in json.loads(p.read_text(encoding="utf-8")):
            cell = rec.get("cell", {})
            dec = rec.get("decompose") or {}
            inp = rec.get("input") or {}
            res = rec.get("result") or {}
            flat.append({
                "impl": impl,
                "size_bytes": cell.get("size_bytes"),
                "pattern": cell.get("pattern"),
                "status": dec.get("status"),
                "dec_wall_s": dec.get("wall_s"),
                "dec_delta_rss_bytes": dec.get("delta_rss_bytes"),
                "dec_peak_rss_bytes": dec.get("peak_rss_bytes"),
                "input_backing": inp.get("backing"),
                "n_rows": res.get("n_rows"),
                "recon_max_abs_err": res.get("recon_max_abs_err"),
                "error": dec.get("error"),
            })
    with open(flat_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(flat[0].keys()) if flat else ["impl"])
        w.writeheader()
        for row in flat:
            w.writerow(row)

    lines = ["# Memory-grid summary", "",
             "Each cell: fresh subprocess; decomposition budget counted after "
             "the input was fully built. 'deferred' cells were recorded but "
             "not executed (see POSTPONED_3GB.md).", ""]
    for impl in IMPLS:
        p = mem / f"{impl}.json"
        if not p.exists():
            continue
        lines.append(f"## {impl}")
        lines.append("")
        lines.append("| size | pattern | status | wall (s) | peak RSS (MB) | rows | note |")
        lines.append("|----:|---------|--------|---------:|--------------:|-----:|------|")
        for rec in json.loads(p.read_text(encoding="utf-8")):
            cell = rec.get("cell", {})
            dec = rec.get("decompose") or {}
            hb = rec.get("hb") or {}
            peak = dec.get("peak_rss_bytes") or hb.get("peak_rss_bytes")
            err = (dec.get("error") or "").replace("|", "/")
            err = err[:60] + "..." if len(err) > 63 else err
            lines.append(
                f"| {cell.get('size_bytes', 0) / 2**20:.0f}MB | "
                f"{cell.get('pattern')} | {dec.get('status')} | "
                f"{dec.get('wall_s', 0) if dec.get('wall_s') is not None else ''} | "
                f"{(peak or 0) / 2**20:.1f} | "
                f"{(rec.get('result') or {}).get('n_rows', '')} | {err} |")
        lines.append("")
    ( _RESULTS / "summary_memory.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {_RESULTS / 'summary_memory.md'} and memory_flat.csv")


if __name__ == "__main__":
    summarize_timing()
    summarize_memory()
