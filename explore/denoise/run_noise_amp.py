"""
Auxiliary-noise amplitude sweep at high noise.

Question: at high input noise, does matching/exceeding the input noise std
with the auxiliary noise improve "main shape" extraction?

The proposed method's ``noise_amp`` is relative to the *current signal* std
(which includes the input noise).  Therefore the actual auxiliary noise std is
``noise_amp * std(noisy)``; the ratio to the input-noise std is

    ratio = noise_amp * std(noisy) / (noise_level * std(clean))

For Gaussian noise, ``std(noisy) ~= std(clean)*sqrt(1+level^2)``, so
``noise_amp = level/sqrt(1+level^2)`` matches the input noise, and
``noise_amp = level`` slightly exceeds it.

Usage:
    python -m explore.denoise.run_noise_amp --reps 5
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for path in (str(ROOT), str(ROOT / "src")):
    if path not in sys.path:
        sys.path.insert(0, path)

from explore.denoise.methods import classical_methods, run_proposed  # noqa: E402
from explore.denoise.metrics import evaluate  # noqa: E402
from explore.denoise.morphology import morph_avg  # noqa: E402
from explore.denoise.signals import FS, N_DEFAULT, make_observation  # noqa: E402

BASE_SEED = 20261017
CASES = ["two_tone", "trend_tone", "am_fm", "impact", "step", "spike_tone"]
LEVELS = (0.5, 0.7, 0.9)
AMPS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2)
K, M = 2, 32


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


def _ratio(obs, amp: float, level: float) -> float:
    input_noise = float(level) * float(obs.clean.std())
    if input_noise <= 0.0:
        return float("nan")
    return float(amp * obs.noisy.std() / input_noise)


def run(
    cases: list[str],
    levels: list[float],
    amps: list[float],
    reps: int,
    n: int,
    out_path: Path,
) -> list[dict]:
    records: list[dict] = []

    for rep in range(reps):
        for case_i, case in enumerate(cases):
            for level_i, level in enumerate(levels):
                obs_seed = BASE_SEED + rep * 100_000 + case_i * 1_000 + level_i * 10
                # pass the level through meta so _ratio can read it
                obs = make_observation(case, level, seed=obs_seed, n=n)
                base = {
                    "case": case,
                    "noise_level": level,
                    "rep": rep,
                    "input_noise_std": level * float(obs.clean.std()),
                    "noisy_std": float(obs.noisy.std()),
                }

                # Deterministic context baselines
                for name, y in classical_methods(obs.noisy, fs=FS).items():
                    if name not in ("raw", "SG11", "Butter150"):
                        continue
                    records.append(
                        {
                            **base,
                            "method": name,
                            "params": {},
                            "metrics": _sanitize(evaluate(obs, y, fs=FS)),
                            "runtime_ms": 0.0,
                        }
                    )
                y = morph_avg(obs.noisy, 5)
                records.append(
                    {
                        **base,
                        "method": "MorphAvg5",
                        "params": {"size": 5},
                        "metrics": _sanitize(evaluate(obs, y, fs=FS)),
                        "runtime_ms": 0.0,
                    }
                )

                for amp in amps:
                    y, ms = _timed(
                        lambda amp=amp: run_proposed(
                            obs.noisy,
                            {"K": K, "M": M, "noise_amp": amp},
                            seed=obs_seed + 7_000,
                        )
                    )
                    records.append(
                        {
                            **base,
                            "method": f"P-K{K}-M{M}-a{amp:g}",
                            "params": {"K": K, "M": M, "noise_amp": amp},
                            "aux_ratio": _ratio(obs, amp, level),
                            "metrics": _sanitize(evaluate(obs, y, fs=FS)),
                            "runtime_ms": ms,
                        }
                    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(records, fh, indent=2, ensure_ascii=False, allow_nan=False)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Auxiliary-noise amplitude sweep")
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--n", type=int, default=N_DEFAULT)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "explore/denoise/results/high_noise_noise_amp.json",
    )
    args = parser.parse_args()

    records = run(
        cases=CASES,
        levels=list(LEVELS),
        amps=list(AMPS),
        reps=args.reps,
        n=args.n,
        out_path=args.out,
    )
    print(f"wrote {len(records)} records -> {args.out}")


if __name__ == "__main__":
    main()
