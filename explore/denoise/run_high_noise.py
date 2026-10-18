"""
High-noise comparison: proposed method vs morphological open-close, SG and
Butterworth filters.

This is a focused follow-up to ``run_benchmark.py``.  It uses only high noise
levels (default 0.5/0.7/0.9 x clean std) and adds the morphological family:

* ``MorphOC{s}``  - opening then closing (classical open-close filter)
* ``MorphCO{s}``  - closing then opening
* ``MorphAvg{s}`` - average of OC and CO

SG windows 11/21 and Butterworth cutoffs 100/150/250 Hz are included so the
comparison is not a single arbitrary setting.  The proposed method uses fixed
``noise_amp=0.2`` with K in {1,2} and M in {32,64}.

Usage:
    python -m explore.denoise.run_high_noise --reps 5
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
from scipy.signal import butter, filtfilt, savgol_filter

ROOT = Path(__file__).resolve().parents[2]
for path in (str(ROOT), str(ROOT / "src")):
    if path not in sys.path:
        sys.path.insert(0, path)

from explore.denoise.methods import ideal_lowpass, run_proposed  # noqa: E402
from explore.denoise.metrics import evaluate  # noqa: E402
from explore.denoise.morphology import morph_methods  # noqa: E402
from explore.denoise.signals import (  # noqa: E402
    CASE_ORDER,
    FS,
    N_DEFAULT,
    make_observation,
)

BASE_SEED = 20261017
HIGH_LEVELS = (0.5, 0.7, 0.9)


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


def build_methods() -> list[tuple[str, Callable]]:
    """Return ``(name, fn(x, seed))`` pairs."""
    methods: list[tuple[str, Callable]] = [
        ("raw", lambda x, seed: np.array(x, dtype=np.float64, copy=True))
    ]
    for name, fn in morph_methods(sizes=(5, 11, 21)).items():
        methods.append((name, lambda x, seed, fn=fn: fn(x)))

    for window in (11, 21):
        methods.append(
            (
                f"SG{window}",
                lambda x, seed, w=window: savgol_filter(
                    x, window_length=w, polyorder=3, mode="interp"
                ),
            )
        )

    for cutoff in (100, 150, 250):
        b, a = butter(4, cutoff / (FS / 2.0), btype="low")
        methods.append(
            (
                f"Butter{cutoff}",
                lambda x, seed, b=b, a=a: filtfilt(b, a, x),
            )
        )

    for K, M in [(1, 32), (1, 64), (2, 32), (2, 64)]:
        methods.append(
            (
                f"P-K{K}-M{M}",
                lambda x, seed, K=K, M=M: run_proposed(
                    x,
                    {"K": K, "M": M, "noise_amp": 0.2},
                    seed=seed,
                ),
            )
        )
    return methods


def _timed(fn) -> tuple[np.ndarray, float]:
    t0 = time.perf_counter()
    y = fn()
    return y, (time.perf_counter() - t0) * 1000.0


def run(
    cases: list[str],
    noise_levels: list[float],
    reps: int,
    n: int,
    out_path: Path,
) -> list[dict]:
    methods = build_methods()
    records: list[dict] = []

    for rep in range(reps):
        for case_i, case in enumerate(cases):
            levels = [1.0] if case == "pure_noise" else noise_levels
            for level_i, level in enumerate(levels):
                obs_seed = BASE_SEED + rep * 100_000 + case_i * 1_000 + level_i * 10
                obs = make_observation(case, level, seed=obs_seed, n=n)
                base = {"case": case, "noise_level": level, "rep": rep}

                for name, fn in methods:
                    y, ms = _timed(lambda fn=fn, seed=obs_seed + 7_000: fn(obs.noisy, seed))
                    records.append(
                        {
                            **base,
                            "method": name,
                            "metrics": _sanitize(evaluate(obs, y, fs=FS)),
                            "runtime_ms": ms,
                        }
                    )

                if case != "pure_noise":
                    y, ms = _timed(lambda: ideal_lowpass(obs.noisy, obs.clean, fs=FS))
                    records.append(
                        {
                            **base,
                            "method": "IdealLP",
                            "metrics": _sanitize(evaluate(obs, y, fs=FS)),
                            "runtime_ms": ms,
                        }
                    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(records, fh, indent=2, ensure_ascii=False, allow_nan=False)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="High-noise denoising comparison")
    parser.add_argument("--cases", nargs="*", default=CASE_ORDER, choices=CASE_ORDER)
    parser.add_argument("--noise-levels", default="0.5,0.7,0.9")
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--n", type=int, default=N_DEFAULT)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "explore/denoise/results/high_noise_morph.json",
    )
    args = parser.parse_args()

    levels = [float(part) for part in args.noise_levels.split(",") if part.strip()]
    records = run(
        cases=list(args.cases),
        noise_levels=levels,
        reps=args.reps,
        n=args.n,
        out_path=args.out,
    )
    print(f"wrote {len(records)} records -> {args.out}")


if __name__ == "__main__":
    main()
