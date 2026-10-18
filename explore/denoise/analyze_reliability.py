"""
Aggregate the reliability-score experiment.

Usage:
    python -m explore.denoise.analyze_reliability
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


def _load(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def summarize(records: list[dict], dataset: str) -> list[dict]:
    groups = defaultdict(list)
    for row in records:
        if row["dataset"] == dataset:
            groups[row["config"]].append(row)
    rows = []
    for config, group in sorted(groups.items()):
        rows.append(
            {
                "dataset": dataset,
                "config": config,
                "n": len(group),
                "true_R": _median([r["true_R"] for r in group]),
                "spur_R": _median([r["spur_R"] for r in group]),
                "true_SNR": _median([r["true_SNR"] for r in group]),
                "spur_SNR": _median([r["spur_SNR"] for r in group]),
                "R_correct": sum(bool(r["R_correct"]) for r in group),
                "SNR_correct": sum(bool(r["SNR_correct"]) for r in group),
                "SNR_wrong_R_correct": sum(
                    1 for r in group if (not r["SNR_correct"]) and r["R_correct"]
                ),
                "both_wrong": sum(
                    1 for r in group if (not r["SNR_correct"]) and (not r["R_correct"])
                ),
                "true_std": _median([r["true_std"] for r in group]),
                "spur_std": _median([r["spur_std"] for r in group]),
            }
        )
    return rows


def print_table(rows: list[dict], title: str) -> None:
    print(f"\n### {title}\n")
    print("| config | n | true R | spur R | true SNR | spur SNR | R correct | SNR correct | SNR wrong but R correct | both wrong |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        print(
            f"| {row['config']} | {row['n']} | {row['true_R']:.2f} | {row['spur_R']:.2f} | "
            f"{row['true_SNR']:.2f} | {row['spur_SNR']:.2f} | {row['R_correct']} | "
            f"{row['SNR_correct']} | {row['SNR_wrong_R_correct']} | {row['both_wrong']} |"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "explore/denoise/results/reliability.json",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=ROOT / "explore/denoise/results",
    )
    args = parser.parse_args()
    records = _load(args.input)
    rows = summarize(records, "cwru") + summarize(records, "synthetic_impact")
    with (args.outdir / "reliability_summary.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with (args.outdir / "reliability_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=2, ensure_ascii=False)
    print_table([r for r in rows if r["dataset"] == "cwru"], "CWRU (true BPFI 162.2 vs spurious 221.5)")
    print_table([r for r in rows if r["dataset"] == "synthetic_impact"], "Synthetic impact (true 12 Hz vs strongest non-12 peak)")


if __name__ == "__main__":
    main()
