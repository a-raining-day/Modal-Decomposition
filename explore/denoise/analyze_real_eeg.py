"""
Aggregate the real-EEG sanity check.

Outputs:
    explore/denoise/results/real_eeg_summary.csv
    explore/denoise/results/real_eeg_summary.json
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
METHODS = [
    "raw", "SG11", "SG21",
    "Butter5", "Butter10", "Butter15", "Butter25",
    "MorphAvg5", "MorphAvg11",
    "P-K2-M64-a0.5", "P-K2-M64-a0.6", "P-K2-M32-a0.5",
]


def _median(values):
    vals = [v for v in values if v is not None]
    return float(statistics.median(vals)) if vals else None


def _load(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def summarize(records: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for record in records:
        groups[(record["method"], float(record["noise_level"]))].append(record)
    rows = []
    for (method, level), group in sorted(groups.items()):
        rows.append(
            {
                "method": method,
                "noise_level": level,
                "n": len(group),
                "shape_nrmse": _median([r["metrics"].get("shape_nrmse") for r in group]),
                "corr": _median([r["metrics"].get("corr") for r in group]),
                "snr_imp": _median([r["metrics"].get("snr_imp") for r in group]),
                "edge_max": _median([r["metrics"].get("edge_max") for r in group]),
                "runtime_ms": _median([r.get("runtime_ms") for r in group]),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "explore/denoise/results/real_eeg.json",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=ROOT / "explore/denoise/results",
    )
    args = parser.parse_args()

    records = _load(args.input)
    rows = summarize(records)
    with (args.outdir / "real_eeg_summary.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["method", "noise_level", "n", "shape_nrmse", "corr", "snr_imp", "edge_max", "runtime_ms"],
        )
        writer.writeheader()
        writer.writerows(rows)
    with (args.outdir / "real_eeg_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=2, ensure_ascii=False)

    print("| method | shape @0.5 | shape @0.7 | shape @0.9 | overall shape | overall SNR |")
    print("|---|---:|---:|---:|---:|---:|")
    for method in METHODS:
        by_level = {r["noise_level"]: r for r in rows if r["method"] == method}
        all_rows = [r for r in records if r["method"] == method]
        overall_shape = _median([r["metrics"].get("shape_nrmse") for r in all_rows])
        overall_snr = _median([r["metrics"].get("snr_imp") for r in all_rows])
        vals = [by_level.get(level, {}).get("shape_nrmse") for level in (0.5, 0.7, 0.9)]
        fmt = lambda v: "-" if v is None else f"{v:.3f}"
        print(
            f"| {method} | {fmt(vals[0])} | {fmt(vals[1])} | {fmt(vals[2])} | "
            f"{fmt(overall_shape)} | {fmt(overall_snr)} |"
        )
    print(f"wrote real_eeg_summary.csv/json -> {args.outdir}")


if __name__ == "__main__":
    main()
