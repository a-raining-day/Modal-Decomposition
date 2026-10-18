"""
Figure for the CWRU bearing demo.

Usage:
    python -m explore.denoise.make_cwru_figure
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, filtfilt, hilbert, savgol_filter

ROOT = Path(__file__).resolve().parents[2]
for path in (str(ROOT), str(ROOT / "src")):
    if path not in sys.path:
        sys.path.insert(0, path)

from explore.denoise.methods import run_proposed  # noqa: E402
from explore.denoise.run_cwru import BPFI, DATA, FS, SEGMENT, load_signal  # noqa: E402

FIG_DIR = ROOT / "docs/explore/denoise/figures"


def env_spectrum(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    b, a = butter(4, [2000.0 / (FS / 2.0), 5000.0 / (FS / 2.0)], btype="band")
    yb = filtfilt(b, a, y)
    env = np.abs(hilbert(yb))
    env = env - env.mean()
    spec = np.abs(np.fft.rfft(env))
    freqs = np.fft.rfftfreq(env.size, 1.0 / FS)
    mask = (freqs >= 20.0) & (freqs <= 300.0)
    return freqs[mask], spec[mask]


def main() -> None:
    x = load_signal()
    seg = x[:SEGMENT]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, level in zip(axes, (0.0, 0.7)):
        rng = np.random.default_rng(20261017)
        noisy = seg if level == 0.0 else seg + level * seg.std() * rng.standard_normal(seg.size)
        b100, a100 = butter(4, 100.0 / (FS / 2.0), btype="low")
        b3000, a3000 = butter(4, 3000.0 / (FS / 2.0), btype="low")
        methods = {
            "raw": noisy,
            "Butter100": filtfilt(b100, a100, noisy),
            "Butter3000": filtfilt(b3000, a3000, noisy),
            "SG11": savgol_filter(noisy, 11, 3, mode="interp"),
            "P-K2-M64-a0.5": run_proposed(
                noisy, {"K": 2, "M": 64, "noise_amp": 0.5}, seed=20261017
            ),
        }
        for name, y in methods.items():
            f, spec = env_spectrum(y)
            med = float(np.median(spec))
            if med > 0:
                ax.semilogy(f, spec / med, lw=1.0, label=name)
        ax.axvline(BPFI, color="black", lw=0.7, linestyle=":", label="BPFI 162 Hz")
        ax.set_xlim(20, 300)
        ax.set_title(f"CWRU IR007, {'real' if level == 0 else f'+noise {level:g}'}")
        ax.set_xlabel("envelope frequency (Hz)")
        ax.grid(alpha=0.2)
        if ax is axes[0]:
            ax.set_ylabel("envelope amplitude / median")
            ax.legend(fontsize=7, ncol=2)
    fig.suptitle("CWRU inner-race 105_0: envelope spectrum after 2-5 kHz demodulation")
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / "cwru_envelope_spectrum.png", dpi=150)
    plt.close(fig)
    print(f"wrote cwru_envelope_spectrum.png -> {FIG_DIR}")


if __name__ == "__main__":
    main()
