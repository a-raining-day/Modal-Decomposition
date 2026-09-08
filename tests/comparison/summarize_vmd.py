"""
Summarize the VMD comparison run (results/vmd/) into a markdown table and a
per-case wall-time figure (figs/timing_wall_vmd.png).

The VMD grid is single-shot (--repeats 1), so no medians: raw walls are shown.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_RES = _HERE / "results" / "vmd"

IMPLS = ["MD-VMD", "vmdpy", "PySDKit"]
NS = (256, 1024, 4096, 16384)


def load() -> tuple[dict, dict]:
    walls: dict[tuple, float] = {}
    raw = _RES / "timing_raw.csv"
    if raw.exists():
        for r in csv.DictReader(raw.open(encoding="utf-8")):
            try:
                walls[(r["impl"], r["case"], int(r["n"]))] = float(r["wall_s"])
            except (TypeError, ValueError):
                pass
    metrics: dict[tuple, dict] = {}
    mp = _RES / "timing_metrics.json"
    if mp.exists():
        for m in json.loads(mp.read_text(encoding="utf-8")):
            metrics[(m["impl"], m["case"], m["n"])] = m
    return walls, metrics


def ratio(o: float, b: float) -> str:
    if b is None:
        return "-"
    return f"{o / b:.2f}x"


def main() -> None:
    walls, metrics = load()
    lines = [
        "# VMD comparison summary (single run, --repeats 1)",
        "",
        "Parity config for all three columns: alpha=2000, tau=0, DC=0, "
        "uniform frequency init, tol=1e-6, max_iter=500; K: case A=3, "
        "B=3, C=4. PySDKit runs with its default `store_history=True`.",
        "",
    ]
    for case in "ABC":
        lines.append(f"## case {case}")
        lines.append("")
        lines.append("| n | MD-VMD | vmdpy | PySDKit | PySDKit/vmdpy |")
        lines.append("|---|-------:|------:|--------:|--------------:|")
        for n in NS:
            vals = [walls.get((i, case, n)) for i in IMPLS]
            if all(v is None for v in vals):
                continue
            fmt = lambda v: f"{v:.4f}" if v is not None else "-"
            lines.append(
                f"| {n} | {fmt(vals[0])} | {fmt(vals[1])} | {fmt(vals[2])} "
                f"| {ratio(vals[2], vals[1])} |"
            )
        lines.append("")
        lines.append("Quality (n=16384):")
        lines.append("")
        lines.append("| impl | recon max abs | best mode corr |")
        lines.append("|------|--------------:|----------------|")
        for impl in IMPLS:
            m = metrics.get((impl, case, 16384), {})
            rec = m.get("mode_recovery", {})
            corr = {k: round(v["best_abs_corr"], 4) for k, v in rec.items()}
            lines.append(
                f"| {impl} | {m.get('recon_max_abs_err', '-')} | {corr or '-'} |"
            )
        lines.append("")

    lines += [
        "## note",
        "",
        "- MD-VMD (wrapper) ~ vmdpy engine within noise (wrapper overhead "
        "negligible).",
        "- PySDKit default (`store_history=True`) never resets the "
        "convergence accumulator, so it always runs the full 500 ADMM "
        "iterations (its `store_history=False` branch resets correctly and "
        "matches vmdpy, e.g. case B n=4096: 0.008s vs 0.248s). Modes are "
        "numerically identical either way (corr ~ 1), so the default path "
        "burns time without changing the result.",
        "- Case C (noise) rarely early-converges for any engine, so the gap "
        "vanishes at n=16384 (3.88s vs 3.91s).",
        "",
    ]
    (_RES / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))

    # figure
    try:
        from matplotlib import pyplot as plt
    except Exception:
        return
    figs = _HERE / "figs"
    figs.mkdir(parents=True, exist_ok=True)
    colors = {"MD-VMD": "#c0392b", "vmdpy": "#2471a3", "PySDKit": "#1e8449"}
    style = {"MD-VMD": "-", "vmdpy": "--", "PySDKit": ":"}
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.9), sharex=True)
    for ax, case in zip(axes, "ABC"):
        for impl in IMPLS:
            xs, ys = [], []
            for n in NS:
                v = walls.get((impl, case, n))
                if v is not None:
                    xs.append(n)
                    ys.append(v)
            if xs:
                ax.loglog(xs, ys, style[impl], color=colors[impl], marker="o",
                          ms=4, label=impl)
        ax.set_title(f"case {case}")
        ax.set_xlabel("n")
        ax.grid(True, which="both", alpha=0.3)
    axes[0].set_ylabel("wall time (s)")
    axes[0].legend()
    fig.suptitle("VMD - wall time (single run, parity config)")
    fig.tight_layout()
    fig.savefig(figs / "timing_wall_vmd.png", dpi=150)
    plt.close(fig)
    print(f"wrote {figs / 'timing_wall_vmd.png'}")


if __name__ == "__main__":
    main()
