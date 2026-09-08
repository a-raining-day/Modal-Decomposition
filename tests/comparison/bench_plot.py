"""
Figures for the EMD comparison report.

* ``figs/timing_wall.png``       - wall time vs signal length (log-log), one
  panel per case, three lines (implementations);
* ``figs/timing_ratio.png``      - PySDKit / MD-EMD wall ratio vs length;
* ``figs/memory_delta.png``      - decomposition RSS delta for the ok
  increasing-pattern cells vs data size (log-log).

Run after the benchmarks: ``python tests/comparison/bench_plot.py``
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_RESULTS = _HERE / "results"
_FIGS = _HERE / "figs"

IMPLS = ["MD-EMD", "PyEMD", "PySDKit"]
_COLORS = {"MD-EMD": "#c0392b", "PyEMD": "#2471a3", "PySDKit": "#1e8449"}
_STYLE = {"MD-EMD": "-", "PyEMD": "--", "PySDKit": ":"}


def _load_timing_medians() -> dict:
    med: dict[tuple, float] = {}
    p = _RESULTS / "timing_raw.csv"
    if not p.exists():
        return med
    by_cell: dict[tuple, list[float]] = {}
    for r in csv.DictReader(p.open(encoding="utf-8")):
        try:
            wall = float(r["wall_s"])
        except (TypeError, ValueError):
            continue
        by_cell.setdefault((r["impl"], r["case"], int(r["n"])), []).append(wall)
    for key, vals in by_cell.items():
        vals.sort()
        med[key] = vals[len(vals) // 2]
    return med


def timing_figs() -> None:
    import numpy as np
    from matplotlib import pyplot as plt

    med = _load_timing_medians()
    if not med:
        print("(no timing data yet)")
        return
    _FIGS.mkdir(parents=True, exist_ok=True)

    for case in ("A", "B", "C"):
        ns = sorted({n for (_, c, n) in med if c == case})
        if not ns:
            continue
        plt.figure(figsize=(7, 4.6))
        for impl in IMPLS:
            ys = [med.get((impl, case, n)) for n in ns]
            xs = [n for n, y in zip(ns, ys) if y is not None]
            ys = [y for y in ys if y is not None]
            if xs:
                plt.loglog(xs, ys, _STYLE[impl], color=_COLORS[impl],
                           marker="o", ms=5, label=impl)
        plt.xlabel("signal length n (samples)")
        plt.ylabel("median wall time (s)")
        plt.title(f"case {case} - EMD wall time")
        plt.legend()
        plt.grid(True, which="both", alpha=0.3)
        plt.tight_layout()
        plt.savefig(_FIGS / f"timing_wall_case{case}.png", dpi=150)
        plt.close()

    # ratio vs MD-EMD (case A, representative)
    case = "A"
    ns = sorted({n for (_, c, n) in med if c == case})
    plt.figure(figsize=(7, 4.2))
    for impl in ("PyEMD", "PySDKit"):
        ratios = []
        for n in ns:
            md = med.get(("MD-EMD", case, n))
            o = med.get((impl, case, n))
            if md and o:
                ratios.append(o / md)
        plt.semilogx(ns[:len(ratios)], ratios, _STYLE[impl],
                     color=_COLORS[impl], marker="o", ms=5,
                     label=f"{impl} / MD-EMD")
    plt.axhline(1.0, color="gray", lw=0.8)
    plt.xlabel("signal length n (samples)")
    plt.ylabel("wall-time ratio (other / MD-EMD)")
    plt.title("case A - relative wall time")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(_FIGS / "timing_ratio_caseA.png", dpi=150)
    plt.close()
    print(f"wrote figures to {_FIGS}")


def memory_fig() -> None:
    import numpy as np
    from matplotlib import pyplot as plt

    _FIG = _FIGS / "memory_delta.png"
    rows = []
    for impl in IMPLS:
        p = _RESULTS / "memory" / f"{impl}.json"
        if not p.exists():
            continue
        for rec in json.loads(p.read_text(encoding="utf-8")):
            cell = rec.get("cell", {})
            dec = rec.get("decompose") or {}
            if cell.get("pattern") == "increasing" and dec.get("status") == "ok" \
                    and dec.get("delta_rss_bytes") is not None:
                rows.append((impl, cell["size_bytes"], dec["delta_rss_bytes"]))
    if not rows:
        print("(no memory data yet)")
        return
    _FIGS.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 4.6))
    for impl in IMPLS:
        pts = [(s, d) for (i, s, d) in rows if i == impl]
        if not pts:
            continue
        pts.sort()
        plt.loglog([s for s, _ in pts], [d for _, d in pts], _STYLE[impl],
                   color=_COLORS[impl], marker="o", ms=5, label=impl)
    plt.xlabel("input size (bytes)")
    plt.ylabel("decomposition RSS delta (bytes)")
    plt.title("memory - increasing pattern (ok cells)")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(_FIG, dpi=150)
    plt.close()
    print(f"wrote {_FIG}")


def memory_before_after_fig() -> None:
    """
    Peak RSS of the three implementations on the increasing-pattern cells,
    with the post-optimization MD-EMD curve overlaid (results/memory_after_opt).
    """
    from matplotlib import pyplot as plt

    _FIG = _FIGS / "memory_peak_before_after.png"
    rows = []  # (impl, size_bytes, peak_rss_bytes)
    for impl in IMPLS:
        p = _RESULTS / "memory" / f"{impl}.json"
        if not p.exists():
            continue
        for rec in json.loads(p.read_text(encoding="utf-8")):
            cell = rec.get("cell", {})
            dec = rec.get("decompose") or {}
            if (cell.get("pattern") == "increasing"
                    and dec.get("status") == "ok"
                    and dec.get("peak_rss_bytes") is not None):
                rows.append((impl, cell["size_bytes"], dec["peak_rss_bytes"]))
    after = []
    p2 = _RESULTS / "memory_after_opt" / "MD-EMD.json"
    if p2.exists():
        for rec in json.loads(p2.read_text(encoding="utf-8")):
            cell = rec.get("cell", {})
            dec = rec.get("decompose") or {}
            if (cell.get("pattern") == "increasing"
                    and dec.get("status") == "ok"
                    and dec.get("peak_rss_bytes") is not None):
                after.append((cell["size_bytes"], dec["peak_rss_bytes"]))
    if not rows and not after:
        print("(no memory data yet)")
        return
    _FIGS.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7.2, 4.8))
    for impl in IMPLS:
        pts = sorted({(s, v) for (i, s, v) in rows if i == impl})
        if not pts:
            continue
        label = f"{impl} (before opt)" if impl == "MD-EMD" else impl
        plt.loglog([s for s, _ in pts], [v for _, v in pts], _STYLE[impl],
                   color=_COLORS[impl], marker="o", ms=5, label=label)
    if after:
        after = sorted(set(after))
        plt.loglog([s for s, _ in after], [v for _, v in after], "--",
                   color="#c0392b", marker="x", ms=8, label="MD-EMD (after opt)")
    plt.xlabel("input size (bytes)")
    plt.ylabel("peak RSS (bytes)")
    plt.title("memory - increasing pattern (peak RSS)")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(_FIG, dpi=150)
    plt.close()
    print(f"wrote {_FIG}")


if __name__ == "__main__":
    timing_figs()
    memory_fig()
    memory_before_after_fig()
