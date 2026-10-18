"""
Real CWRU bearing-data demo (P2, mechanical data).

Data: ``explore/denoise/data/cwru/IR007_105_0.mat``
  - 12 kHz drive-end acceleration, inner-race fault 0.007 in, 1797 rpm;
  - BPFI ~= 5.4152 * fr, fr = 1797/60 = 29.95 Hz -> 162.2 Hz;
  - five non-overlapping 24,000-sample segments (2 s each).

We compare the proposed method with a *fixed* Butter100 (the baseline used in
the synthetic reports), a wideband Butter3000, SG11, and MorphAvg5.  The
primary metric is the envelope-spectrum SNR at BPFI (after a 2-5 kHz
demodulation band), plus the peak frequency and kurtosis.

Noise levels 0.5/0.7/0.9 add white Gaussian noise to the real CWRU segments;
level 0.0 is the real record with its own noise.

Usage:
    python -m explore.denoise.run_cwru
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import scipy.io as sio
from scipy.signal import butter, filtfilt, hilbert, savgol_filter

ROOT = Path(__file__).resolve().parents[2]
for path in (str(ROOT), str(ROOT / "src")):
    if path not in sys.path:
        sys.path.insert(0, path)

from explore.denoise.methods import run_proposed  # noqa: E402
from explore.denoise.morphology import morph_avg  # noqa: E402

FS = 12000.0
DATA = ROOT / "explore/denoise/data/cwru/IR007_105_0.mat"
SEGMENT = 24000
SEGMENTS = (0, 24000, 48000, 72000, 96000)
LEVELS = (0.0, 0.5, 0.7, 0.9)
BPFI = 5.4152 * (1797.0 / 60.0)  # 162.18 Hz


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


def load_signal() -> np.ndarray:
    data = sio.loadmat(DATA)
    x = np.asarray(data["X105_DE_time"], dtype=np.float64).ravel()
    return x


def envelope_metrics(y: np.ndarray, band=(2000.0, 5000.0), f_range=(80.0, 250.0)) -> dict:
    b, a = butter(4, [band[0] / (FS / 2.0), band[1] / (FS / 2.0)], btype="band")
    yb = filtfilt(b, a, y)
    env = np.abs(hilbert(yb))
    env = env - env.mean()
    spec = np.abs(np.fft.rfft(env))
    freqs = np.fft.rfftfreq(env.size, 1.0 / FS)
    mask = (freqs >= f_range[0]) & (freqs <= f_range[1])
    spec = spec[mask]
    freqs = freqs[mask]
    median = float(np.median(spec))
    peak_idx = int(np.argmax(spec))
    peak_snr = float(spec[peak_idx] / median) if median > 0 else float("inf")
    bpfi_mask = np.abs(freqs - BPFI) <= 2.0
    bpfi_snr = float(spec[bpfi_mask].max() / median) if median > 0 else float("inf")
    var = float(np.var(y))
    kurt = float(np.mean((y - y.mean()) ** 4) / (var * var)) if var > 0 else float("nan")
    return {
        "env_peak_freq": float(freqs[peak_idx]),
        "env_peak_snr": peak_snr,
        "bpfi_freq": BPFI,
        "bpfi_snr": bpfi_snr,
        "kurtosis": kurt,
        "std": float(y.std()),
    }


def methods(y: np.ndarray) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {"raw": np.array(y, copy=True)}
    for cutoff in (100, 3000):
        b, a = butter(4, cutoff / (FS / 2.0), btype="low")
        out[f"Butter{cutoff}"] = filtfilt(b, a, y)
    out["SG11"] = savgol_filter(y, 11, 3, mode="interp")
    out["MorphAvg5"] = morph_avg(y, 5)
    for K, M, amp in ((2, 64, 0.5), (2, 64, 0.6), (1, 64, 0.5)):
        out[f"P-K{K}-M{M}-a{amp}"] = run_proposed(
            y, {"K": K, "M": M, "noise_amp": amp}, seed=0
        )
    return out


def run() -> list[dict]:
    x = load_signal()
    records = []
    for level in LEVELS:
        for seg_i, start in enumerate(SEGMENTS):
            seg = x[start : start + SEGMENT].copy()
            rng = np.random.default_rng(20261017 + seg_i * 100 + int(level * 100))
            noisy = seg if level == 0.0 else seg + level * seg.std() * rng.standard_normal(seg.size)
            for name, y in methods(noisy).items():
                # re-seed proposed methods per segment/level; methods() used seed 0 above,
                # so for reproducibility re-run proposed with a deterministic seed.
                if name.startswith("P-"):
                    K = int(name.split("-")[1][1:])
                    M = int(name.split("-")[2][1:])
                    amp = float(name.split("-a")[1])
                    y = run_proposed(
                        noisy,
                        {"K": K, "M": M, "noise_amp": amp},
                        seed=20261017 + seg_i * 100 + int(level * 100) + 7000,
                    )
                t0 = time.perf_counter()
                metrics = envelope_metrics(y)
                metrics["runtime_ms"] = (time.perf_counter() - t0) * 1000.0
                records.append(
                    {
                        "case": "cwru_IR007_105_0",
                        "noise_level": level,
                        "segment": seg_i,
                        "method": name,
                        "metrics": _sanitize(metrics),
                    }
                )
    return records


def summarize(records: list[dict]) -> list[dict]:
    groups = {}
    for record in records:
        key = (record["method"], record["noise_level"])
        groups.setdefault(key, []).append(record["metrics"])
    rows = []
    for (method, level), metrics in sorted(groups.items()):
        rows.append(
            {
                "method": method,
                "noise_level": level,
                "n": len(metrics),
                "env_peak_freq": float(np.median([m["env_peak_freq"] for m in metrics])),
                "env_peak_snr": float(np.median([m["env_peak_snr"] for m in metrics])),
                "bpfi_snr": float(np.median([m["bpfi_snr"] for m in metrics])),
                "kurtosis": float(np.median([m["kurtosis"] for m in metrics])),
                "std": float(np.median([m["std"] for m in metrics])),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="CWRU bearing demo")
    parser.add_argument(
        "--outdir",
        type=Path,
        default=ROOT / "explore/denoise/results",
    )
    args = parser.parse_args()

    records = run()
    rows = summarize(records)
    with (args.outdir / "cwru.json").open("w", encoding="utf-8") as fh:
        json.dump(records, fh, indent=2, ensure_ascii=False)
    with (args.outdir / "cwru_summary.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["method", "noise_level", "n", "env_peak_freq", "env_peak_snr", "bpfi_snr", "kurtosis", "std"],
        )
        writer.writeheader()
        writer.writerows(rows)
    with (args.outdir / "cwru_summary.json").open("w", encoding="utf-8") as fh:
        json.dump({"bpfi": BPFI, "rows": rows}, fh, indent=2, ensure_ascii=False)

    print(f"BPFI = {BPFI:.2f} Hz; records={len(records)}")
    print("| method | peak SNR @0 | peak SNR @0.5 | peak SNR @0.7 | peak SNR @0.9 | peak freq @0 | BPFI SNR @0 |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    methods_order = [
        "raw", "Butter100", "Butter3000", "SG11", "MorphAvg5",
        "P-K2-M64-a0.5", "P-K2-M64-a0.6", "P-K1-M64-a0.5",
    ]
    for method in methods_order:
        by = {r["noise_level"]: r for r in rows if r["method"] == method}
        vals = [by.get(level, {}).get("env_peak_snr") for level in LEVELS]
        f0 = by.get(0.0, {}).get("env_peak_freq")
        bpfi = by.get(0.0, {}).get("bpfi_snr")
        fmt = lambda v: "-" if v is None else f"{v:.2f}"
        print(
            f"| {method} | {fmt(vals[0])} | {fmt(vals[1])} | {fmt(vals[2])} | {fmt(vals[3])} | "
            f"{fmt(f0)} | {fmt(bpfi)} |"
        )
    print(f"wrote cwru.json/csv -> {args.outdir}")


if __name__ == "__main__":
    main()
