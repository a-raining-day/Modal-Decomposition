"""Figure for CWRU reliability vs auxiliary-noise amplitude."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "explore/denoise/results"
FIG_DIR = ROOT / "fig/explore/denoise"


def main() -> None:
    with (RESULTS / "reliability_amp_summary.json").open("r", encoding="utf-8") as fh:
        rows = json.load(fh)
    amps = [r["amp"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    axes[0].semilogx(amps, [r["true_R"] for r in rows], marker="o", label="true BPFI 162 Hz")
    axes[0].semilogx(amps, [r["spur_R"] for r in rows], marker="s", label="spurious 221.5 Hz")
    axes[0].set_xlabel("auxiliary noise_amp")
    axes[0].set_ylabel("R = mean / std")
    axes[0].set_title("Reliability ratio")
    axes[0].grid(alpha=0.2, which="both")
    axes[0].legend(fontsize=8)

    axes[1].semilogx(amps, [r["true_SNR"] for r in rows], marker="o", label="true BPFI")
    axes[1].semilogx(amps, [r["spur_SNR"] for r in rows], marker="s", label="spurious 221.5")
    axes[1].set_xlabel("auxiliary noise_amp")
    axes[1].set_ylabel("SNR = mean / median")
    axes[1].set_title("Peak SNR")
    axes[1].grid(alpha=0.2, which="both")
    axes[1].legend(fontsize=8)

    fig.suptitle("CWRU: small auxiliary noise recovers the true BPFI peak; large amp creates 221.5 Hz")
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / "reliability_amp.png", dpi=150)
    plt.close(fig)
    print(f"wrote reliability_amp.png -> {FIG_DIR}")


if __name__ == "__main__":
    main()
