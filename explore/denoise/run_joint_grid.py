"""
Joint K x M x noise_amp grid at high noise, plus per-shape oracle baselines.

Purpose:
1. Test the "low-parameter" narrative: can one fixed (K, M, noise_amp)
   configuration work across shapes, or does each shape need its own setting?
2. Quantify the cost of *not* tuning: compare a global fixed config and
   leave-one-shape-out (LOSO) selection against a per-shape oracle over the
   proposed grid, and against a per-shape oracle over all classical /
   morphological baselines.

Grid:
    K in {1, 2, 3}
    M in {16, 32, 64}
    noise_amp in {0.2, 0.3, 0.5, 0.6, 0.8}

Cases: the six non-null shapes; noise levels 0.5/0.7/0.9; 5 repetitions.
The baseline oracle includes SG11/21, Butter100/150/250, and all
MorphOC/CO/Avg 5/11/21 variants, so it is a strong "tuned filter" reference.

Usage:
    python -m explore.denoise.run_joint_grid --reps 5
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for path in (str(ROOT), str(ROOT / "src")):
    if path not in sys.path:
        sys.path.insert(0, path)

from explore.denoise.methods import run_proposed  # noqa: E402
from explore.denoise.metrics import evaluate  # noqa: E402
from explore.denoise.morphology import morph_methods  # noqa: E402
from explore.denoise.signals import FS, N_DEFAULT, make_observation  # noqa: E402

BASE_SEED = 20261017
CASES = ["two_tone", "trend_tone", "am_fm", "impact", "step", "spike_tone"]
LEVELS = (0.5, 0.7, 0.9)
KS = (1, 2, 3)
MS = (16, 32, 64)
AMPS = (0.2, 0.3, 0.5, 0.6, 0.8)


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


def _baseline_fns() -> list[tuple[str, Callable[[np.ndarray], np.ndarray]]]:
    from scipy.signal import butter, filtfilt, savgol_filter

    fns: list[tuple[str, Callable[[np.ndarray], np.ndarray]]] = [
        ("raw", lambda x: np.array(x, dtype=np.float64, copy=True))
    ]
    for window in (11, 21):
        fns.append(
            (
                f"SG{window}",
                lambda x, w=window: savgol_filter(x, w, 3, mode="interp"),
            )
        )
    for cutoff in (100, 150, 250):
        b, a = butter(4, cutoff / (FS / 2.0), btype="low")
        fns.append((f"Butter{cutoff}", lambda x, b=b, a=a: filtfilt(b, a, x)))
    for name, fn in morph_methods(sizes=(5, 11, 21)).items():
        fns.append((name, lambda x, fn=fn: fn(x)))
    return fns


def run(
    cases: list[str],
    levels: list[float],
    reps: int,
    n: int,
    out_path: Path,
) -> list[dict]:
    baselines = _baseline_fns()
    records: list[dict] = []
    total = len(cases) * len(levels) * reps * (len(baselines) + len(KS) * len(MS) * len(AMPS))
    done = 0

    for rep in range(reps):
        for case_i, case in enumerate(cases):
            for level_i, level in enumerate(levels):
                obs_seed = BASE_SEED + rep * 100_000 + case_i * 1_000 + level_i * 10
                obs = make_observation(case, level, seed=obs_seed, n=n)
                base = {"case": case, "noise_level": level, "rep": rep}

                for name, fn in baselines:
                    y, ms = _timed(lambda fn=fn: fn(obs.noisy))
                    records.append(
                        {
                            **base,
                            "method": name,
                            "params": {},
                            "metrics": _sanitize(evaluate(obs, y, fs=FS)),
                            "runtime_ms": ms,
                        }
                    )
                    done += 1

                for K in KS:
                    for M in MS:
                        for amp in AMPS:
                            name = f"P-K{K}-M{M}-a{amp:g}"
                            y, ms = _timed(
                                lambda K=K, M=M, amp=amp: run_proposed(
                                    obs.noisy,
                                    {"K": K, "M": M, "noise_amp": amp},
                                    seed=obs_seed + 7_000,
                                )
                            )
                            records.append(
                                {
                                    **base,
                                    "method": name,
                                    "params": {"K": K, "M": M, "noise_amp": amp},
                                    "metrics": _sanitize(evaluate(obs, y, fs=FS)),
                                    "runtime_ms": ms,
                                }
                            )
                            done += 1
                if done % 500 == 0 or done == total:
                    print(f"  {done}/{total}", flush=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(records, fh, indent=2, ensure_ascii=False, allow_nan=False)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Joint K x M x amp grid")
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--n", type=int, default=N_DEFAULT)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "explore/denoise/results/joint_grid.json",
    )
    args = parser.parse_args()
    records = run(
        cases=CASES,
        levels=list(LEVELS),
        reps=args.reps,
        n=args.n,
        out_path=args.out,
    )
    print(f"wrote {len(records)} records -> {args.out}")


if __name__ == "__main__":
    main()
