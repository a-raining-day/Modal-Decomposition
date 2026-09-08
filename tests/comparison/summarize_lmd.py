"""
Summarize the LMD four-column comparison run (results/lmd/) into a markdown
table and a per-case wall-time figure (figs/timing_wall_lmd.png).
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_RES = _HERE / "results" / (sys.argv[1] if len(sys.argv) > 1 else "lmd")

IMPLS = ["LMD-H(scipy)", "LMD-midpoint", "LMD-H(FHT)", "PySDKit"]
NS = (256, 1024, 4096, 16384)


def main() -> None:
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

    lines = [
        "# LMD comparison summary (single run, --repeats 1)",
        "",
        "Four columns: LMD-H(scipy) = Hilbert-envelope LMD (scipy analytic "
        "signal); LMD-midpoint = shipped extrema-midpoint LMD; "
        "LMD-H(FHT) = Hilbert-envelope LMD via the compiled C (SAO FHT) "
        "kernel; PySDKit = pysdkit.LMD (classical Smith moving-average). "
        "All my variants capped at max_pf=5 (pysdkit default K=5).",
        "",
    ]
    for case in "ABC":
        lines.append(f"## case {case}")
        lines.append("")
        lines.append("| n | LMD-H(scipy) | LMD-midpoint | LMD-H(FHT) | PySDKit |")
        lines.append("|---|-------------:|-------------:|-----------:|--------:|")
        for n in NS:
            vals = [walls.get((i, case, n)) for i in IMPLS]
            fmt = lambda v: f"{v:.4f}" if v is not None else "-"
            lines.append(
                f"| {n} | {fmt(vals[0])} | {fmt(vals[1])} | {fmt(vals[2])} "
                f"| {fmt(vals[3])} |"
            )
        lines.append("")
        lines.append("Quality (n=16384):")
        lines.append("")
        lines.append("| impl | n_rows | recon max abs | best mode corr |")
        lines.append("|------|-------:|--------------:|----------------|")
        for impl in IMPLS:
            m = metrics.get((impl, case, 16384), {})
            rec = m.get("mode_recovery", {})
            corr = {k: round(v["best_abs_corr"], 4) for k, v in rec.items()}
            lines.append(
                f"| {impl} | {m.get('n_rows', '-')} | "
                f"{m.get('recon_max_abs_err', '-')} | {corr or '-'} |"
            )
        lines.append("")

    lines += [
        "## notes",
        "",
        "- Hilbert variants use single-shot demodulation per PF "
        "(a = |H(h)|, PF = h - m_t); the iterated Hilbert sift does NOT "
        "converge on multi-component signals (extrema count 77 -> 201 and "
        "growing on case A) - the historical `compute_envelope` in the old "
        "LMD was dead code for the same reason.",
        "- Compiled C kernel `_fht_native` is active; its envelope matches "
        "scipy to ~1e-14, and standalone microbenchmark at n=16384 shows the "
        "C FHT kernel itself is ~1.9x SLOWER per call than scipy's FFT-based "
        "hilbert (0.36 vs 0.19 ms); inside LMD the difference is hidden "
        "because each PF only needs one Hilbert call.",
        "- PySDKit's moving-average LMD is pure-Python in the inner loops "
        "(per-sample extrema staircase + smoothing), so it scales the worst "
        "with n.",
        "",
    ]
    (_RES / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))

    try:
        from matplotlib import pyplot as plt
    except Exception:
        return
    figs = _HERE / "figs"
    figs.mkdir(parents=True, exist_ok=True)
    colors = {"LMD-H(scipy)": "#8e44ad", "LMD-midpoint": "#c0392b",
              "LMD-H(FHT)": "#16a085", "PySDKit": "#2471a3"}
    style = {"LMD-H(scipy)": "-", "LMD-midpoint": "-",
             "LMD-H(FHT)": ":", "PySDKit": "--"}
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
    axes[0].legend(fontsize=7)
    fig.suptitle("LMD - wall time (single run)")
    fig.tight_layout()
    out_fig = figs / f"timing_wall_lmd_{_RES.name}.png"
    fig.savefig(out_fig, dpi=150)
    plt.close(fig)
    print(f"wrote {out_fig}")


if __name__ == "__main__":
    main()
