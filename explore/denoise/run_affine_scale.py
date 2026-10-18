"""
Affine-scale distribution for the high-noise main-shape extractor.

For a fixed config (``K=2, M=64, noise_amp=0.6``), fit

    clean ~= a * output + b

on three shapes (two_tone, trend_tone, step), at input noise 0.5/0.7/0.9,
5 repetitions.  The slope ``a`` tells a user how much amplitude rescaling is
needed if they want to use the output directly for a threshold; ``b`` is the
offset.

Usage:
    python -m explore.denoise.run_affine_scale
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for path in (str(ROOT), str(ROOT / "src")):
    if path not in sys.path:
        sys.path.insert(0, path)

from explore.denoise.methods import run_proposed  # noqa: E402
from explore.denoise.signals import make_observation  # noqa: E402

BASE_SEED = 20261017
CASES = ["two_tone", "trend_tone", "step"]
ALL_CASES = ["two_tone", "trend_tone", "am_fm", "impact", "step", "spike_tone"]
LEVELS = (0.5, 0.7, 0.9)
REPS = 5
CONFIG = {"K": 2, "M": 64, "noise_amp": 0.6}


def fit_affine(clean: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    design = np.column_stack([y, np.ones_like(y)])
    coef, *_ = np.linalg.lstsq(design, clean, rcond=None)
    return float(coef[0]), float(coef[1])


def run() -> list[dict]:
    rows = []
    for case in CASES:
        case_i = ALL_CASES.index(case)
        for level_i, level in enumerate(LEVELS):
            for rep in range(REPS):
                seed = BASE_SEED + rep * 100_000 + case_i * 1_000 + level_i * 10
                obs = make_observation(case, level, seed=seed)
                y = run_proposed(obs.noisy, CONFIG, seed=seed + 7_000)
                a, b = fit_affine(obs.clean, y)
                rows.append(
                    {
                        "case": case,
                        "noise_level": level,
                        "rep": rep,
                        "a": a,
                        "b": b,
                        "corr": float(np.corrcoef(obs.clean, y)[0, 1]),
                        "clean_std": float(obs.clean.std()),
                        "output_std": float(y.std()),
                    }
                )
    return rows


def summarize(rows: list[dict]) -> dict:
    groups = defaultdict(list)
    for row in rows:
        groups[row["case"]].append(row["a"])
    out = {}
    for case, values in groups.items():
        values = np.asarray(values, dtype=float)
        out[case] = {
            "n": int(values.size),
            "median": float(np.median(values)),
            "q25": float(np.percentile(values, 25)),
            "q75": float(np.percentile(values, 75)),
            "p5": float(np.percentile(values, 5)),
            "p95": float(np.percentile(values, 95)),
            "min": float(values.min()),
            "max": float(values.max()),
        }
    per_level = defaultdict(lambda: defaultdict(list))
    for row in rows:
        per_level[row["case"]][row["noise_level"]].append(row["a"])
    out["by_level"] = {
        case: {
            str(level): float(np.median(vals))
            for level, vals in levels.items()
        }
        for case, levels in per_level.items()
    }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Affine-scale distribution")
    parser.add_argument(
        "--outdir",
        type=Path,
        default=ROOT / "explore/denoise/results",
    )
    args = parser.parse_args()

    rows = run()
    summary = summarize(rows)
    args.outdir.mkdir(parents=True, exist_ok=True)
    with (args.outdir / "affine_scale.json").open("w", encoding="utf-8") as fh:
        json.dump({"config": CONFIG, "rows": rows, "summary": summary}, fh,
                  indent=2, ensure_ascii=False)
    with (args.outdir / "affine_scale.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["case", "noise_level", "rep", "a", "b", "corr", "clean_std", "output_std"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print("| shape | n | a median | IQR | 5-95% | min-max |")
    print("|---|---:|---:|---|---|---|")
    for case in CASES:
        s = summary[case]
        print(
            f"| {case} | {s['n']} | {s['median']:.3f} | "
            f"[{s['q25']:.3f}, {s['q75']:.3f}] | "
            f"[{s['p5']:.3f}, {s['p95']:.3f}] | "
            f"[{s['min']:.3f}, {s['max']:.3f}] |"
        )
    print("\nBy noise level (median a):")
    for case in CASES:
        print(case, {k: round(v, 3) for k, v in summary["by_level"][case].items()})
    print(f"wrote affine_scale.json/csv -> {args.outdir}")


if __name__ == "__main__":
    main()
