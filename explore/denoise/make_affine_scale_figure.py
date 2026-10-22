"""
Figure for the affine-scale ``a`` distribution.

Usage:
    python -m explore.denoise.make_affine_scale_figure
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
FIG_DIR = ROOT / "fig/explore/denoise"
RESULTS = ROOT / "explore/denoise/results"
CASES = ["two_tone", "trend_tone", "step"]
LEVELS = (0.5, 0.7, 0.9)


def main() -> None:
    with (RESULTS / "affine_scale.json").open("r", encoding="utf-8") as fh:
        data = json.load(fh)["rows"]
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(11, 4), sharey=True)
    for ax, case in zip(axes, CASES):
        values = [
            [row["a"] for row in data if row["case"] == case and abs(row["noise_level"] - level) < 1e-12]
            for level in LEVELS
        ]
        ax.boxplot(values, widths=0.5, showfliers=False)
        for i, vals in enumerate(values):
            x = np.full(len(vals), i + 1) + np.linspace(-0.08, 0.08, len(vals))
            ax.scatter(x, vals, s=12, alpha=0.7)
        ax.axhline(1.0, color="black", lw=0.6, linestyle=":")
        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels([f"{level:g}" for level in LEVELS])
        ax.set_xlabel("input noise level")
        ax.set_title(case)
        ax.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("affine scale a (clean ≈ a*y+b)")
    fig.suptitle("Affine amplitude scale needed to map output back to clean amplitude")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "affine_a_distribution.png", dpi=150)
    plt.close(fig)
    print(f"wrote affine_a_distribution.png -> {FIG_DIR}")


if __name__ == "__main__":
    main()
