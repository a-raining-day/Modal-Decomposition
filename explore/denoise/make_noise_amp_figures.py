"""
Figures for the auxiliary-noise amplitude sweep.

Usage:
    python -m explore.denoise.make_noise_amp_figures
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = ROOT / "docs/explore/denoise/figures"
RESULTS = ROOT / "explore/denoise/results"

AMPS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2)
LEVELS = (0.5, 0.7, 0.9)
CASES = ["two_tone", "trend_tone", "am_fm", "impact", "step", "spike_tone"]


def _load():
    with (RESULTS / "high_noise_noise_amp_summary.json").open("r", encoding="utf-8") as fh:
        return json.load(fh)["summary"]


def _median(values):
    vals = [v for v in values if v is not None]
    return float(np.median(vals)) if vals else float("nan")


def _rows(summary, case=None, level=None, amp=None):
    out = []
    for row in summary:
        if case is not None and row["case"] != case:
            continue
        if level is not None and abs(float(row["noise_level"]) - level) > 1e-12:
            continue
        if amp is not None and row["method"] != f"P-K2-M32-a{amp:g}":
            continue
        if case is None and level is None and amp is None and not row["method"].startswith("P-K2-M32-a"):
            continue
        out.append(row)
    return out


def figure_by_level(summary) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    colors = {0.5: "C0", 0.7: "C1", 0.9: "C2"}
    for level in LEVELS:
        shape = [_median([r["shape_nrmse"] for r in _rows(summary, level=level, amp=a)]) for a in AMPS]
        snr = [_median([r["snr_imp"] for r in _rows(summary, level=level, amp=a)]) for a in AMPS]
        axes[0].plot(AMPS, shape, marker="o", ms=3, color=colors[level], label=f"input noise={level:g}")
        axes[1].plot(AMPS, snr, marker="o", ms=3, color=colors[level], label=f"input noise={level:g}")
        match = level / (1.0 + level**2) ** 0.5
        axes[0].axvline(match, color=colors[level], linestyle=":", lw=0.8, alpha=0.7)
        axes[1].axvline(match, color=colors[level], linestyle=":", lw=0.8, alpha=0.7)
    axes[0].set_xlabel("auxiliary noise_amp (relative to current signal std)")
    axes[0].set_ylabel("shape NRMSE (lower is better)")
    axes[0].set_title("Shape vs auxiliary noise; dotted = input-noise match")
    axes[1].set_xlabel("auxiliary noise_amp")
    axes[1].set_ylabel("SNR improvement (dB)")
    axes[1].set_title("SNR vs auxiliary noise")
    for ax in axes:
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "noise_amp_shape_snr.png", dpi=150)
    plt.close(fig)


def figure_per_case(summary) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for case in CASES:
        shape = [_median([r["shape_nrmse"] for r in _rows(summary, case=case, amp=a)]) for a in AMPS]
        ax.plot(AMPS, shape, marker="o", ms=3, label=case)
    ax.set_xlabel("auxiliary noise_amp")
    ax.set_ylabel("shape NRMSE (lower is better)")
    ax.set_title("Per-shape optimum is different (median over 3 input noise levels)")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "noise_amp_per_case.png", dpi=150)
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    summary = _load()
    figure_by_level(summary)
    figure_per_case(summary)
    print(f"wrote noise-amp figures -> {FIG_DIR}")


if __name__ == "__main__":
    main()
