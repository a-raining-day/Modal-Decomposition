"""
Aggregate CWRU reliability vs auxiliary-noise amplitude.

Usage:
    python -m explore.denoise.analyze_reliability_amp
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _median(values):
    vals = [v for v in values if v is not None]
    return float(statistics.median(vals)) if vals else None


def summarize(records: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for row in records:
        groups[float(row["amp"])].append(row)
    rows = []
    for amp, group in sorted(groups.items()):
        rows.append(
            {
                "amp": amp,
                "n": len(group),
                "true_R": _median([r["true_R"] for r in group]),
                "spur_R": _median([r["spur_R"] for r in group]),
                "true_SNR": _median([r["true_SNR"] for r in group]),
                "spur_SNR": _median([r["spur_SNR"] for r in group]),
                "true_std": _median([r["true_std"] for r in group]),
                "spur_std": _median([r["spur_std"] for r in group]),
                "R_correct": sum(bool(r["R_correct"]) for r in group),
                "SNR_correct": sum(bool(r["SNR_correct"]) for r in group),
                "SNR_wrong_R_correct": sum(
                    1 for r in group if (not r["SNR_correct"]) and r["R_correct"]
                ),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "explore/denoise/results/reliability_amp.json",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=ROOT / "explore/denoise/results",
    )
    args = parser.parse_args()
    with args.input.open("r", encoding="utf-8") as fh:
        records = json.load(fh)
    rows = summarize(records)
    with (args.outdir / "reliability_amp_summary.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with (args.outdir / "reliability_amp_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=2, ensure_ascii=False)

    print("| amp | true R | spur R | true SNR | spur SNR | true std | spur std | R correct | SNR correct | SNR wrong but R correct |")
    print("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        print(
            f"| {row['amp']:g} | {row['true_R']:.2f} | {row['spur_R']:.2f} | "
            f"{row['true_SNR']:.2f} | {row['spur_SNR']:.2f} | {row['true_std']:.3f} | "
            f"{row['spur_std']:.3f} | {row['R_correct']}/5 | {row['SNR_correct']}/5 | "
            f"{row['SNR_wrong_R_correct']} |"
        )


if __name__ == "__main__":
    main()
