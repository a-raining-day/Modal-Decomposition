"""
Figures for the joint K x M x amp grid / oracle-cost report.

Usage:
    python -m explore.denoise.make_joint_grid_figures
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

KS = (1, 2, 3)
MS = (16, 32, 64)
AMPS = (0.5, 0.6, 0.8)
CASES = ["two_tone", "trend_tone", "am_fm", "impact", "step", "spike_tone"]


def _load():
    with (RESULTS / "joint_grid_summary.json").open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _shape(result, K, M, amp):
    for row in result["stats"]:
        if row["K"] == K and row["M"] == M and abs(row["amp"] - amp) < 1e-12:
            return row["shape_med"]
    return np.nan


def figure_heatmap(result) -> None:
    fig, axes = plt.subplots(1, len(AMPS), figsize=(12, 4.2), sharey=True)
    for ax, amp in zip(axes, AMPS):
        data = np.array([[_shape(result, K, M, amp) for M in MS] for K in KS])
        im = ax.imshow(data, aspect="auto", cmap="viridis_r", vmin=0.34, vmax=0.44)
        ax.set_xticks(range(len(MS)))
        ax.set_xticklabels([str(m) for m in MS])
        ax.set_yticks(range(len(KS)))
        ax.set_yticklabels([str(k) for k in KS])
        ax.set_xlabel("M")
        ax.set_title(f"noise_amp={amp:g}")
        for i in range(len(KS)):
            for j in range(len(MS)):
                ax.text(j, i, f"{data[i, j]:.3f}", ha="center", va="center",
                        color="white" if data[i, j] > 0.39 else "black", fontsize=8)
    axes[0].set_ylabel("K")
    fig.colorbar(im, ax=axes, label="shape NRMSE")
    fig.suptitle("Joint K x M x amp grid (median over 90 observations; lower is better)")
    fig.savefig(FIG_DIR / "joint_grid_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def figure_cost(result) -> None:
    table = result["case_table"]
    labels = [r["case"] for r in table]
    global_v = [r["global_shape"] for r in table]
    oracle_p = [r["oracle_p_shape"] for r in table]
    loso = [r["loso_shape"] for r in table]
    oracle_c = [r["oracle_baseline_shape"] for r in table]
    x = np.arange(len(labels))
    width = 0.2
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.bar(x - 1.5 * width, global_v, width, label="global fixed", color="C3")
    ax.bar(x - 0.5 * width, oracle_p, width, label="oracle proposed", color="C1")
    ax.bar(x + 0.5 * width, loso, width, label="LOSO fixed", color="C0")
    ax.bar(x + 1.5 * width, oracle_c, width, label="oracle classical", color="C2")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("shape NRMSE (lower is better)")
    ax.set_title("Cost of one fixed configuration vs per-shape oracles")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "joint_grid_cost.png", dpi=150)
    plt.close(fig)


def figure_marginal(result) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(11, 4))
    specs = [("K", (1, 2, 3)), ("M", (16, 32, 64)), ("amp", (0.2, 0.3, 0.5, 0.6, 0.8))]
    for ax, (param, values) in zip(axes, specs):
        y = [result["marginal"][param][str(v)] if str(v) in result["marginal"][param] else result["marginal"][param][v] for v in values]
        ax.plot(range(len(values)), y, marker="o", color="C3")
        ax.set_xticks(range(len(values)))
        ax.set_xticklabels([f"{v:g}" for v in values])
        ax.set_xlabel(param)
        ax.set_ylabel("median shape NRMSE")
        ax.grid(alpha=0.2)
        ax.set_title(f"marginal: {param}")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "joint_grid_marginal.png", dpi=150)
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    result = _load()
    figure_heatmap(result)
    figure_cost(result)
    figure_marginal(result)
    print(f"wrote joint-grid figures -> {FIG_DIR}")


if __name__ == "__main__":
    main()
