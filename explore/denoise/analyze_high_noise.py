"""
Aggregate the high-noise morphology comparison.

Outputs:
    explore/denoise/results/high_noise_summary.csv
    explore/denoise/results/high_noise_ranking.csv
    explore/denoise/results/high_noise_summary.json

and optional Markdown tables for the report.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

METHOD_ORDER = [
    "raw",
    "MorphOC5", "MorphOC11", "MorphOC21",
    "MorphCO5", "MorphCO11", "MorphCO21",
    "MorphAvg5", "MorphAvg11", "MorphAvg21",
    "SG11", "SG21",
    "Butter100", "Butter150", "Butter250",
    "P-K1-M32", "P-K1-M64", "P-K2-M32", "P-K2-M64",
    "IdealLP",
]

DEPLOYABLE = [m for m in METHOD_ORDER if m not in ("raw", "IdealLP")]

SUMMARY_METRICS = [
    "rmse_int",
    "snr_imp",
    "corr",
    "shape_nrmse",
    "shape_lp_corr",
    "shape_lp_nrmse",
    "edge_max",
    "trend_lp_rmse",
    "weak_retention",
    "env_peak_snr",
    "env_corr",
    "impact_amp_ratio",
    "edge_width_samples",
    "overshoot",
    "plateau_rmse",
    "spike_leakage",
    "clean_region_rmse",
    "maxabs",
    "output_rms_ratio",
    "spectral_flatness",
]


def _median(values):
    vals = [v for v in values if v is not None]
    return float(statistics.median(vals)) if vals else None


def _mean(values):
    vals = [v for v in values if v is not None]
    return float(statistics.mean(vals)) if vals else None


def _std(values):
    vals = [v for v in values if v is not None]
    return float(statistics.pstdev(vals)) if len(vals) > 1 else 0.0


def _load(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def build_summary(records: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for record in records:
        if "metrics" not in record:
            continue
        groups[(record["case"], float(record["noise_level"]), record["method"])].append(record)
    rows = []
    for (case, level, method), group in sorted(groups.items()):
        row = {"case": case, "noise_level": level, "method": method, "n": len(group)}
        for metric in SUMMARY_METRICS:
            row[metric] = _median([r["metrics"].get(metric) for r in group])
        row["runtime_ms"] = _median([r.get("runtime_ms") for r in group])
        rows.append(row)
    return rows


def _ranks(values: dict[str, float], methods: list[str]) -> dict[str, float]:
    """Average ranks (1 = lowest value) among the given methods."""
    present = [(m, values[m]) for m in methods if values.get(m) is not None]
    present.sort(key=lambda item: item[1])
    ranks: dict[str, float] = {}
    i = 0
    while i < len(present):
        j = i
        while j + 1 < len(present) and abs(present[j + 1][1] - present[i][1]) < 1e-12:
            j += 1
        rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[present[k][0]] = rank
        i = j + 1
    return ranks


def build_ranking(records: list[dict]) -> list[dict]:
    by_obs = defaultdict(dict)
    for record in records:
        if "metrics" not in record:
            continue
        key = (record["case"], float(record["noise_level"]), record["rep"])
        by_obs[key][record["method"]] = record

    ranks = defaultdict(list)
    shape_vals = defaultdict(list)
    snr_vals = defaultdict(list)
    rmse_vals = defaultdict(list)
    edge_vals = defaultdict(list)
    runtime_vals = defaultdict(list)

    for by_method in by_obs.values():
        values = {
            method: record["metrics"].get("shape_nrmse")
            for method, record in by_method.items()
            if record["metrics"].get("shape_nrmse") is not None
        }
        for method, rank in _ranks(values, DEPLOYABLE).items():
            ranks[method].append(rank)
        for method, record in by_method.items():
            if method not in DEPLOYABLE:
                continue
            metrics = record["metrics"]
            shape_vals[method].append(metrics.get("shape_nrmse"))
            snr_vals[method].append(metrics.get("snr_imp"))
            rmse_vals[method].append(metrics.get("rmse_int"))
            edge_vals[method].append(metrics.get("edge_max"))
            if record.get("runtime_ms") is not None:
                runtime_vals[method].append(record["runtime_ms"])

    rows = []
    for method in METHOD_ORDER:
        if method not in DEPLOYABLE:
            continue
        rows.append(
            {
                "method": method,
                "n_obs": len(shape_vals.get(method, [])),
                "mean_rank": _mean(ranks.get(method, [])),
                "median_rank": _median(ranks.get(method, [])),
                "worst_rank": max(ranks[method]) if ranks.get(method) else None,
                "rank_std": _std(ranks.get(method, [])),
                "shape_nrmse_med": _median(shape_vals.get(method, [])),
                "snr_imp_med": _median(snr_vals.get(method, [])),
                "rmse_int_med": _median(rmse_vals.get(method, [])),
                "edge_max_med": _median(edge_vals.get(method, [])),
                "runtime_ms_med": _median(runtime_vals.get(method, [])),
            }
        )
    rows.sort(key=lambda row: (row["mean_rank"] is None, row["mean_rank"]))
    return rows


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _fmt(value, digits=3):
    if value is None:
        return "-"
    if isinstance(value, float):
        if abs(value) >= 1000:
            return f"{value:.2e}"
        return f"{value:.{digits}f}"
    return str(value)


def _row(summary, case, level, method):
    for row in summary:
        if (
            row["case"] == case
            and abs(float(row["noise_level"]) - level) < 1e-12
            and row["method"] == method
        ):
            return row
    return None


def _print_table(summary, case, level, methods, columns):
    print(f"\n**{case}** (noise={level:g})\n")
    header = " | ".join(["method"] + [label for _, label in columns])
    print(f"| {header} |")
    print("|" + "|".join(["---"] + ["---:"] * len(columns)) + "|")
    for method in methods:
        row = _row(summary, case, level, method)
        if row is None:
            continue
        values = " | ".join(_fmt(row.get(key)) for key, _ in columns)
        print(f"| {method} | {values} |")


def print_markdown(records: list[dict]) -> None:
    summary = build_summary(records)
    ranking = build_ranking(records)

    print("\n### Universal-shape ranking (rank by shape NRMSE; lower is better)\n")
    print("| method | mean rank | median rank | worst rank | rank std | shape NRMSE | SNR imp. | edge max |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in ranking:
        print(
            f"| {row['method']} | {_fmt(row['mean_rank'], 2)} | {_fmt(row['median_rank'], 2)} | "
            f"{_fmt(row['worst_rank'], 2)} | {_fmt(row['rank_std'], 2)} | "
            f"{_fmt(row['shape_nrmse_med'])} | {_fmt(row['snr_imp_med'])} | {_fmt(row['edge_max_med'])} |"
        )

    methods = [
        "raw",
        "MorphAvg5",
        "MorphAvg11",
        "MorphAvg21",
        "SG11",
        "SG21",
        "Butter100",
        "Butter150",
        "Butter250",
        "P-K1-M32",
        "P-K2-M32",
        "IdealLP",
    ]
    for case in ["two_tone", "trend_tone", "am_fm", "impact", "step", "spike_tone"]:
        _print_table(
            summary,
            case,
            0.7,
            methods,
            [
                ("shape_nrmse", "shape NRMSE"),
                ("shape_lp_corr", "shape LP corr"),
                ("snr_imp", "SNR imp. (dB)"),
                ("edge_max", "edge max"),
            ],
        )

    print("\n### Morphology size/family sensitivity (median over all non-null obs)\n")
    print("| method | shape NRMSE | SNR imp. | edge max |")
    print("|---|---:|---:|---:|")
    for method in [
        "MorphOC5", "MorphOC11", "MorphOC21",
        "MorphCO5", "MorphCO11", "MorphCO21",
        "MorphAvg5", "MorphAvg11", "MorphAvg21",
    ]:
        row = next((r for r in ranking if r["method"] == method), None)
        if row:
            print(
                f"| {row['method']} | {_fmt(row['shape_nrmse_med'])} | "
                f"{_fmt(row['snr_imp_med'])} | {_fmt(row['edge_max_med'])} |"
            )

    print("\n### Feature metrics at noise=0.7\n")
    _print_table(
        summary,
        "impact",
        0.7,
        ["raw", "MorphAvg5", "MorphAvg11", "SG11", "Butter150", "Butter250", "P-K1-M32", "P-K2-M32", "IdealLP"],
        [
            ("env_peak_snr", "env peak SNR"),
            ("env_corr", "env corr"),
            ("impact_amp_ratio", "impact amp ratio"),
        ],
    )
    _print_table(
        summary,
        "step",
        0.7,
        ["raw", "MorphAvg5", "MorphAvg11", "MorphAvg21", "SG21", "Butter150", "P-K1-M32", "P-K2-M32"],
        [
            ("edge_width_samples", "edge width"),
            ("overshoot", "overshoot"),
            ("plateau_rmse", "plateau RMSE"),
        ],
    )
    _print_table(
        summary,
        "spike_tone",
        0.7,
        ["raw", "MorphAvg5", "MorphAvg11", "SG11", "SG21", "Butter100", "Butter150", "P-K1-M32", "P-K2-M32"],
        [
            ("spike_leakage", "spike leakage"),
            ("clean_region_rmse", "clean-region RMSE"),
        ],
    )
    _print_table(
        summary,
        "trend_tone",
        0.7,
        ["raw", "MorphAvg5", "MorphAvg11", "SG11", "SG21", "Butter100", "Butter150", "P-K1-M32", "P-K2-M32"],
        [
            ("weak_retention", "weak 160 Hz retention"),
            ("trend_lp_rmse", "trend LP RMSE"),
        ],
    )
    _print_table(
        summary,
        "pure_noise",
        1.0,
        ["raw", "MorphAvg5", "MorphAvg11", "SG11", "Butter150", "P-K1-M32", "P-K2-M32"],
        [
            ("output_rms_ratio", "output/input RMS"),
            ("maxabs", "max abs"),
            ("spectral_flatness", "spectral flatness"),
        ],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "explore/denoise/results/high_noise_morph.json",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=ROOT / "explore/denoise/results",
    )
    parser.add_argument("--markdown", action="store_true")
    args = parser.parse_args()

    records = _load(args.input)
    summary = build_summary(records)
    ranking = build_ranking(records)

    fields = ["case", "noise_level", "method", "n"] + SUMMARY_METRICS + ["runtime_ms"]
    write_csv(args.outdir / "high_noise_summary.csv", summary, fields)
    write_csv(
        args.outdir / "high_noise_ranking.csv",
        ranking,
        ["method", "n_obs", "mean_rank", "median_rank", "worst_rank", "rank_std",
         "shape_nrmse_med", "snr_imp_med", "rmse_int_med", "edge_max_med", "runtime_ms_med"],
    )
    with (args.outdir / "high_noise_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(
            {"summary": summary, "ranking": ranking},
            fh,
            indent=2,
            ensure_ascii=False,
        )
    print(f"wrote {len(summary)} summary rows, {len(ranking)} ranking rows -> {args.outdir}")
    if args.markdown:
        print_markdown(records)


if __name__ == "__main__":
    main()
