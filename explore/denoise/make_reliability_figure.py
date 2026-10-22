"""
Figure for the reliability-score experiment.

Usage:
    python -m explore.denoise.make_reliability_figure
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio

ROOT = Path(__file__).resolve().parents[2]
for path in (str(ROOT), str(ROOT / "src")):
    if path not in sys.path:
        sys.path.insert(0, path)

from explore.denoise.methods import run_proposed  # noqa: E402
from explore.denoise.run_cwru import BPFI, DATA, FS  # noqa: E402
from explore.denoise.run_reliability import env_spectrum  # noqa: E402

FIG_DIR = ROOT / "fig/explore/denoise"
SPUR = 221.5


def main() -> None:
    data = sio.loadmat(DATA)
    x = np.asarray(data["X105_DE_time"], dtype=np.float64).ravel()
    seg = x[:24000]
    specs = []
    for r in range(20):
        y = run_proposed(seg, {"K": 2, "M": 64, "noise_amp": 0.5}, seed=9000 + r)
        f, spec = env_spectrum(y, FS, (2000.0, 5000.0), (80.0, 250.0))
        specs.append(spec)
    specs = np.asarray(specs)
    mean = specs.mean(axis=0)
    std = specs.std(axis=0, ddof=1)
    ratio = mean / np.maximum(std, 1e-30)

    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    axes[0].semilogy(f, mean, color="C3", lw=1.2, label="mean envelope spectrum")
    axes[0].fill_between(f, np.maximum(mean - std, 1e-6), mean + std,
                         color="C3", alpha=0.2, label="±1 std across 20 runs")
    axes[0].axvline(BPFI, color="black", linestyle=":", lw=0.8, label=f"BPFI {BPFI:.1f} Hz")
    axes[0].axvline(SPUR, color="gray", linestyle="--", lw=0.8, label=f"spurious {SPUR:g} Hz")
    axes[0].set_ylabel("envelope amplitude")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.2, which="both")
    axes[0].set_title("CWRU IR007, K2-M64-a0.5: mean ± std over 20 runs")

    axes[1].semilogy(f, ratio, color="C0", lw=1.2)
    axes[1].axvline(BPFI, color="black", linestyle=":", lw=0.8)
    axes[1].axvline(SPUR, color="gray", linestyle="--", lw=0.8)
    axes[1].set_xlabel("envelope frequency (Hz)")
    axes[1].set_ylabel("mean / std")
    axes[1].grid(alpha=0.2, which="both")
    axes[1].set_title("Reliability ratio; spurious peak is not less stable")
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / "reliability_cwru.png", dpi=150)
    plt.close(fig)
    print(f"wrote reliability_cwru.png -> {FIG_DIR}")


if __name__ == "__main__":
    main()
