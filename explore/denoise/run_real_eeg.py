"""
Real-data demo on a bundled real EEG trace.

No network access / public CWRU or PhysioNet download was available in this
environment, so this uses the real EEG sample shipped with matplotlib:
``mpl-data/sample_data/eeg.dat`` (4 channels x 800 samples; channel 0 used
here).  The sample does not document a sampling rate; we assume 100 Hz only
for filter cutoffs and display.  The *signal morphology is real*; the added
noise is white Gaussian at 0.5/0.7/0.9 x the EEG std (semi-synthetic test).

This is deliberately a sanity check, not a substitute for a real CWRU /
colored-noise study.

Usage:
    python -m explore.denoise.run_real_eeg
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Callable

import matplotlib
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for path in (str(ROOT), str(ROOT / "src")):
    if path not in sys.path:
        sys.path.insert(0, path)

from explore.denoise.methods import run_proposed  # noqa: E402
from explore.denoise.metrics import evaluate  # noqa: E402
from explore.denoise.morphology import morph_avg  # noqa: E402
from explore.denoise.signals import Observation  # noqa: E402

FS = 100.0
BASE_SEED = 20261017
LEVELS = (0.5, 0.7, 0.9)
REPS = 10


def load_eeg() -> np.ndarray:
    path = Path(matplotlib.get_data_path()) / "sample_data" / "eeg.dat"
    data = np.frombuffer(path.read_bytes(), dtype="<f8")
    # The file is 4 channels x 800 samples, channel-major.
    channels = data.reshape(-1, 4).T
    clean = channels[0].astype(np.float64)
    clean = clean - clean.mean()
    return clean


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


def _timed(fn) -> tuple[np.ndarray, float]:
    t0 = time.perf_counter()
    y = fn()
    return y, (time.perf_counter() - t0) * 1000.0


def build_methods() -> list[tuple[str, Callable[[np.ndarray], np.ndarray]]]:
    from scipy.signal import butter, filtfilt, savgol_filter

    methods: list[tuple[str, Callable[[np.ndarray], np.ndarray]]] = [
        ("raw", lambda x: np.array(x, dtype=np.float64, copy=True))
    ]
    for window in (11, 21):
        methods.append((f"SG{window}", lambda x, w=window: savgol_filter(x, w, 3, mode="interp")))
    for cutoff in (5, 10, 15, 25):
        b, a = butter(4, cutoff / (FS / 2.0), btype="low")
        methods.append((f"Butter{cutoff}", lambda x, b=b, a=a: filtfilt(b, a, x)))
    for size in (5, 11):
        methods.append((f"MorphAvg{size}", lambda x, s=size: morph_avg(x, s)))
    for K, M, amp in ((2, 64, 0.5), (2, 64, 0.6), (2, 32, 0.5)):
        methods.append(
            (
                f"P-K{K}-M{M}-a{amp}",
                lambda x, K=K, M=M, amp=amp: run_proposed(
                    x, {"K": K, "M": M, "noise_amp": amp}, seed=0
                ),
            )
        )
    return methods


def run(clean: np.ndarray, reps: int, out_path: Path) -> list[dict]:
    methods = build_methods()
    records = []
    for level in LEVELS:
        for rep in range(reps):
            seed = BASE_SEED + int(level * 100) * 1_000 + rep
            rng = np.random.default_rng(seed)
            noisy = clean + level * float(clean.std()) * rng.standard_normal(clean.size)
            obs = Observation(
                name="real_eeg",
                clean=clean,
                noisy=noisy,
                components={},
                meta={"kind": "real"},
            )
            for name, fn in methods:
                if name.startswith("P-"):
                    K = int(name.split("-")[1][1:])
                    M = int(name.split("-")[2][1:])
                    amp = float(name.split("-a")[1])
                    y, ms = _timed(
                        lambda K=K, M=M, amp=amp: run_proposed(
                            noisy,
                            {"K": K, "M": M, "noise_amp": amp},
                            seed=seed + 7_000,
                        )
                    )
                else:
                    y, ms = _timed(lambda fn=fn: fn(noisy))
                records.append(
                    {
                        "case": "real_eeg",
                        "noise_level": level,
                        "rep": rep,
                        "method": name,
                        "metrics": _sanitize(evaluate(obs, y, fs=FS)),
                        "runtime_ms": ms,
                    }
                )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(records, fh, indent=2, ensure_ascii=False, allow_nan=False)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Real EEG sanity check")
    parser.add_argument("--reps", type=int, default=REPS)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "explore/denoise/results/real_eeg.json",
    )
    args = parser.parse_args()
    clean = load_eeg()
    records = run(clean, args.reps, args.out)
    print(f"EEG samples={clean.size}, std={clean.std():.4f}; wrote {len(records)} records -> {args.out}")


if __name__ == "__main__":
    main()
