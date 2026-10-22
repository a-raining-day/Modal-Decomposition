"""
Figure for the real-EEG sanity check.

Usage:
    python -m explore.denoise.make_real_eeg_figure
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, filtfilt

ROOT = Path(__file__).resolve().parents[2]
for path in (str(ROOT), str(ROOT / "src")):
    if path not in sys.path:
        sys.path.insert(0, path)

from explore.denoise.methods import run_proposed  # noqa: E402
from explore.denoise.run_real_eeg import FS, load_eeg  # noqa: E402

FIG_DIR = ROOT / "fig/explore/denoise"


def main() -> None:
    clean = load_eeg()
    rng = np.random.default_rng(20261017 + 70 * 1_000)
    level = 0.7
    noisy = clean + level * clean.std() * rng.standard_normal(clean.size)
    b, a = butter(4, 15.0 / (FS / 2.0), btype="low")
    butter_y = filtfilt(b, a, noisy)
    proposed = run_proposed(
        noisy,
        {"K": 2, "M": 64, "noise_amp": 0.6},
        seed=20261017 + 70 * 1_000 + 7_000,
    )
    t = np.arange(clean.size) / FS
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(t, clean, color="black", lw=1.2, label="real EEG (reference)")
    ax.plot(t, noisy, color="0.68", lw=0.7, alpha=0.8, label="EEG + noise 0.7")
    ax.plot(t, butter_y, color="C0", lw=1.1, label="Butter15")
    ax.plot(t, proposed, color="C3", lw=1.1, label="P-K2-M64-a0.6")
    ax.set_xlabel("time (s; assumed 100 Hz for display)")
    ax.set_ylabel("amplitude")
    ax.set_title("Real EEG morphology + synthetic high noise: proposed vs Butter15")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / "real_eeg_example.png", dpi=150)
    plt.close(fig)
    print(f"wrote real_eeg_example.png -> {FIG_DIR}")


if __name__ == "__main__":
    main()
