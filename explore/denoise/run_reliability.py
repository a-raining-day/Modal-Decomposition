"""
Can a "peak / realization-std" reliability score flag a spurious envelope peak?

Hypothesis (from the user):
    For the proposed method, the spurious CWRU peak at 221.5 Hz should have a
    much larger M-realization variance than the true BPFI peak at ~161.5 Hz.
    Then ``peak / std`` could act as a confidence score and distinguish a real
    fault frequency from a spurious one, even when plain peak SNR ranks the
    spurious peak higher.

Experiment:
* CWRU: 5 non-overlapping 2-s segments; R independent method runs; envelope
  spectrum per run; compare true BPFI (~162.2 Hz) vs spurious 221.5 Hz.
* Synthetic impact: 3 noise levels x 3 observations; R runs; true 12 Hz vs the
  strongest non-12-Hz peak.

Metrics at a frequency f:
    SNR(f) = mean_spectrum(f) / median(mean_spectrum)
    R(f)   = mean_spectrum(f) / std_across_realizations(f)

A useful reliability score should have R(true) > R(spurious) at least when
SNR(spurious) > SNR(true); if R(spurious) is also larger, the score is not
diagnostic.

Usage:
    python -m explore.denoise.run_reliability
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import scipy.io as sio
from scipy.signal import butter, filtfilt, hilbert

ROOT = Path(__file__).resolve().parents[2]
for path in (str(ROOT), str(ROOT / "src")):
    if path not in sys.path:
        sys.path.insert(0, path)

from explore.denoise.methods import run_proposed  # noqa: E402
from explore.denoise.signals import make_observation  # noqa: E402

CWRU = ROOT / "explore/denoise/data/cwru/IR007_105_0.mat"
CWRU_FS = 12000.0
CWRU_SEGMENTS = (0, 24000, 48000, 72000, 96000)
CWRU_SEG_LEN = 24000
BPFI = 5.4152 * (1797.0 / 60.0)
SPUR = 221.5
R = 20

CONFIGS = [
    {"name": "K2-M64-a0.5", "K": 2, "M": 64, "amp": 0.5},
    {"name": "K2-M64-a0.6", "K": 2, "M": 64, "amp": 0.6},
    {"name": "K2-M16-a0.3", "K": 2, "M": 16, "amp": 0.3},
    {"name": "K3-M64-a0.5", "K": 3, "M": 64, "amp": 0.5},
]


def _sanitize(value):
    if isinstance(value, dict):
        return {str(k): _sanitize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize(v) for v in value]
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return value if math.isfinite(value) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def env_spectrum(
    y: np.ndarray, fs: float, band: tuple[float, float], f_range: tuple[float, float]
) -> tuple[np.ndarray, np.ndarray]:
    b, a = butter(4, [band[0] / (fs / 2.0), band[1] / (fs / 2.0)], btype="band")
    yb = filtfilt(b, a, y)
    env = np.abs(hilbert(yb))
    env = env - env.mean()
    spec = np.abs(np.fft.rfft(env))
    freqs = np.fft.rfftfreq(env.size, 1.0 / fs)
    mask = (freqs >= f_range[0]) & (freqs <= f_range[1])
    return freqs[mask], spec[mask]


def peak_in(freqs, mean_spec, center, half):
    mask = np.abs(freqs - center) <= half
    if not np.any(mask):
        return None
    idx = np.flatnonzero(mask)[int(np.argmax(mean_spec[mask]))]
    return int(idx)


def analyze_specs(freqs, specs, true_freq, true_half, spur_freq, spur_half, label):
    specs = np.asarray(specs, dtype=np.float64)
    mean = specs.mean(axis=0)
    std = specs.std(axis=0, ddof=1)
    median = float(np.median(mean))
    i_true = peak_in(freqs, mean, true_freq, true_half)
    i_spur = peak_in(freqs, mean, spur_freq, spur_half)
    if i_true is None or i_spur is None:
        return None
    return {
        "label": label,
        "true_freq": float(freqs[i_true]),
        "true_mean": float(mean[i_true]),
        "true_std": float(std[i_true]),
        "true_R": float(mean[i_true] / std[i_true]) if std[i_true] > 0 else None,
        "true_SNR": float(mean[i_true] / median) if median > 0 else None,
        "spur_freq": float(freqs[i_spur]),
        "spur_mean": float(mean[i_spur]),
        "spur_std": float(std[i_spur]),
        "spur_R": float(mean[i_spur] / std[i_spur]) if std[i_spur] > 0 else None,
        "spur_SNR": float(mean[i_spur] / median) if median > 0 else None,
        "R_correct": bool(mean[i_true] / std[i_true] > mean[i_spur] / std[i_spur])
        if std[i_true] > 0 and std[i_spur] > 0
        else None,
        "SNR_correct": bool(mean[i_true] > mean[i_spur]),
    }


def run_cwru() -> list[dict]:
    data = sio.loadmat(CWRU)
    x = np.asarray(data["X105_DE_time"], dtype=np.float64).ravel()
    out = []
    for cfg in CONFIGS:
        for seg_i, start in enumerate(CWRU_SEGMENTS):
            seg = x[start : start + CWRU_SEG_LEN]
            specs = []
            t0 = time.perf_counter()
            for r in range(R):
                y = run_proposed(
                    seg,
                    {"K": cfg["K"], "M": cfg["M"], "noise_amp": cfg["amp"]},
                    seed=9000 + seg_i * 100 + r,
                )
                f, spec = env_spectrum(y, CWRU_FS, (2000.0, 5000.0), (80.0, 250.0))
                specs.append(spec)
            row = analyze_specs(f, specs, BPFI, 2.0, SPUR, 2.0, f"cwru_seg{seg_i}")
            if row:
                row.update(
                    {
                        "dataset": "cwru",
                        "config": cfg["name"],
                        "segment": seg_i,
                        "runtime_s": time.perf_counter() - t0,
                    }
                )
                out.append(row)
    return out


def run_synthetic() -> list[dict]:
    out = []
    cfgs = CONFIGS[:3]
    for cfg in cfgs:
        for level in (0.5, 0.7, 0.9):
            for obs_i in range(3):
                obs = make_observation("impact", level, seed=20261017 + obs_i * 1000)
                specs = []
                for r in range(R):
                    y = run_proposed(
                        obs.noisy,
                        {"K": cfg["K"], "M": cfg["M"], "noise_amp": cfg["amp"]},
                        seed=11000 + obs_i * 100 + r,
                    )
                    f, spec = env_spectrum(y, 1000.0, (120.0, 240.0), (4.0, 20.0))
                    specs.append(spec)
                specs = np.asarray(specs)
                mean = specs.mean(axis=0)
                # strongest peak excluding the true 12 Hz region
                mask = np.abs(f - 12.0) > 1.0
                idx = np.flatnonzero(mask)[int(np.argmax(mean[mask]))]
                row = analyze_specs(
                    f, specs, true_freq=12.0, true_half=0.5,
                    spur_freq=float(f[idx]), spur_half=0.5,
                    label=f"impact_level{level}_obs{obs_i}",
                )
                if row:
                    row.update(
                        {
                            "dataset": "synthetic_impact",
                            "config": cfg["name"],
                            "level": level,
                            "obs": obs_i,
                        }
                    )
                    out.append(row)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--outdir",
        type=Path,
        default=ROOT / "explore/denoise/results",
    )
    args = parser.parse_args()
    records = run_cwru() + run_synthetic()
    with (args.outdir / "reliability.json").open("w", encoding="utf-8") as fh:
        json.dump(records, fh, indent=2, ensure_ascii=False)
    print(f"wrote {len(records)} records -> {args.outdir / 'reliability.json'}")


if __name__ == "__main__":
    main()
