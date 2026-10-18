"""
Aggregate the denoising benchmark JSON into CSV/JSON summaries and Markdown
tables used by the report.

Usage:
    python -m explore.denoise.analyze_results \
        --input explore/denoise/results/benchmark.json \
        --outdir explore/denoise/results
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
    "MA5",
    "SG11",
    "Median5",
    "Butter150",
    "Wiener11",
    "IdealLP",
    "P-K1-M1",
    "P-K1-M32",
    "P-K2-M1",
    "P-K2-M32",
    "P-K3-M1",
    "P-K3-M32",
    "EEMD-corr",
]

# Metrics exported to summary.csv (all are optional; missing ones are skipped).
SUMMARY_METRICS = [
    "rmse",
    "rmse_int",
    "snr_in",
    "snr_out",
    "snr_imp",
    "corr",
    "shape_nrmse",
    "shape_lp_corr",
    "shape_lp_nrmse",
    "maxabs_err",
    "edge_rmse",
    "edge_max",
    "clean_bw",
    "hf_ratio",
    "lsd",
    "trend_rmse",
    "trend_corr",
    "trend_lp_rmse",
    "trend_lp_corr",
    "weak_retention",
    "env_peak_freq",
    "env_peak_snr",
    "env_corr",
    "kurtosis",
    "impact_amp_ratio",
    "edge_width_samples",
    "overshoot",
    "plateau_rmse",
    "spike_leakage",
    "clean_region_rmse",
    "maxabs",
    "output_rms",
    "output_rms_ratio",
    "spectral_flatness",
]


def _median(values):
    vals = [v for v in values if v is not None]
    return float(statistics.median(vals)) if vals else None


def _load(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _group(records: list[dict], keys: tuple[str, ...]):
    groups = defaultdict(list)
    for record in records:
        if "metrics" not in record:
            continue
        groups[tuple(record[k] for k in keys)].append(record)
    return groups


def build_summary(records: list[dict]) -> list[dict]:
    rows = []
    for (case, level, method), group in sorted(
        _group(records, ("case", "noise_level", "method")).items()
    ):
        row = {
            "case": case,
            "noise_level": level,
            "method": method,
            "n": len(group),
        }
        for metric in SUMMARY_METRICS:
            row[metric] = _median([r["metrics"].get(metric) for r in group])
        row["runtime_ms"] = _median([r.get("runtime_ms") for r in group])
        rows.append(row)
    return rows


def build_relative(records: list[dict], methods: list[str] | None = None) -> list[dict]:
    """Method performance relative to ``raw`` on the same observation."""
    allowed = set(methods) if methods is not None else None
    by_obs = defaultdict(dict)
    for record in records:
        if "metrics" not in record:
            continue
        key = (record["case"], record["noise_level"], record["rep"])
        by_obs[key][record["method"]] = record

    rel: dict[str, list[float]] = defaultdict(list)
    snr: dict[str, list[float]] = defaultdict(list)
    edge: dict[str, list[float]] = defaultdict(list)
    runtime: dict[str, list[float]] = defaultdict(list)
    for by_method in by_obs.values():
        raw = by_method.get("raw")
        if raw is None:
            continue
        raw_rmse = raw["metrics"].get("rmse_int")
        if not raw_rmse or raw_rmse <= 0:
            continue
        for method, record in by_method.items():
            if allowed is not None and method not in allowed:
                continue
            rmse = record["metrics"].get("rmse_int")
            if rmse is not None:
                rel[method].append(float(rmse / raw_rmse))
            imp = record["metrics"].get("snr_imp")
            if imp is not None:
                snr[method].append(float(imp))
            edge_max = record["metrics"].get("edge_max")
            if edge_max is not None:
                edge[method].append(float(edge_max))
            if record.get("runtime_ms") is not None:
                runtime[method].append(float(record["runtime_ms"]))

    rows = []
    for method in (methods if methods is not None else METHOD_ORDER):
        if method not in rel and method not in snr:
            continue
        rows.append(
            {
                "method": method,
                "n_obs": len(rel.get(method, [])),
                "rel_rmse_med": _median(rel.get(method, [])),
                "snr_imp_med": _median(snr.get(method, [])),
                "edge_max_med": _median(edge.get(method, [])),
                "runtime_ms_med": _median(runtime.get(method, [])),
            }
        )
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
        return "—"
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
    main_methods = [m for m in METHOD_ORDER if m != "EEMD-corr"]
    relative = build_relative(records, methods=main_methods)

    print("\n### Overall (median over all observations; EEMD subset excluded)\n")
    print("| method | rel. RMSE | SNR imp. (dB) | edge max | runtime (ms) | n |")
    print("|---|---:|---:|---:|---:|---:|")
    for row in relative:
        print(
            f"| {row['method']} | {_fmt(row['rel_rmse_med'])} | "
            f"{_fmt(row['snr_imp_med'])} | {_fmt(row['edge_max_med'])} | "
            f"{_fmt(row['runtime_ms_med'], 2)} | {row['n_obs']} |"
        )

    show = [
        "raw",
        "SG11",
        "Butter150",
        "Wiener11",
        "IdealLP",
        "P-K1-M32",
        "P-K2-M32",
        "P-K3-M32",
    ]
    for case in [
        "two_tone",
        "trend_tone",
        "am_fm",
        "impact",
        "step",
        "spike_tone",
    ]:
        _print_table(
            summary,
            case,
            0.3,
            show,
            [
                ("rmse_int", "RMSE int"),
                ("snr_imp", "SNR imp. (dB)"),
                ("corr", "corr"),
                ("edge_max", "edge max"),
            ],
        )

    print("\n### Feature metrics\n")
    _print_table(
        summary,
        "trend_tone",
        0.3,
        show,
        [
            ("trend_lp_rmse", "trend LP RMSE"),
            ("weak_retention", "weak 160 Hz retention"),
        ],
    )
    _print_table(
        summary,
        "impact",
        0.3,
        ["raw", "SG11", "Butter150", "Wiener11", "IdealLP", "P-K1-M32", "P-K2-M32", "P-K3-M32", "EEMD-corr"],
        [
            ("env_peak_freq", "env peak (Hz)"),
            ("env_peak_snr", "env peak SNR"),
            ("env_corr", "env corr"),
            ("impact_amp_ratio", "impact amp ratio"),
        ],
    )
    _print_table(
        summary,
        "step",
        0.3,
        show,
        [
            ("edge_width_samples", "edge width (samples)"),
            ("overshoot", "overshoot"),
            ("plateau_rmse", "plateau RMSE"),
        ],
    )
    _print_table(
        summary,
        "spike_tone",
        0.3,
        show,
        [
            ("spike_leakage", "spike leakage"),
            ("clean_region_rmse", "clean-region RMSE"),
            ("maxabs", "max abs"),
        ],
    )
    _print_table(
        summary,
        "pure_noise",
        1.0,
        ["raw", "SG11", "Butter150", "Wiener11", "P-K1-M32", "P-K2-M32", "P-K3-M32"],
        [
            ("output_rms_ratio", "output/input RMS"),
            ("spectral_flatness", "spectral flatness"),
            ("maxabs", "max abs"),
        ],
    )

    print("\n### EEMD subset (noise=0.3)\n")
    for case in ["two_tone", "trend_tone", "impact"]:
        _print_table(
            summary,
            case,
            0.3,
            ["raw", "SG11", "Butter150", "Wiener11", "IdealLP", "P-K1-M32", "P-K2-M32", "EEMD-corr"],
            [
                ("rmse_int", "RMSE int"),
                ("snr_imp", "SNR imp. (dB)"),
                ("env_peak_snr", "env peak SNR"),
            ],
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "explore/denoise/results/benchmark.json",
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
    main_methods = [m for m in METHOD_ORDER if m != "EEMD-corr"]
    relative = build_relative(records, methods=main_methods)

    fields = ["case", "noise_level", "method", "n"] + SUMMARY_METRICS + ["runtime_ms"]
    write_csv(args.outdir / "summary.csv", summary, fields)
    write_csv(
        args.outdir / "ranking.csv",
        relative,
        ["method", "n_obs", "rel_rmse_med", "snr_imp_med", "edge_max_med", "runtime_ms_med"],
    )
    with (args.outdir / "summary.json").open("w", encoding="utf-8") as fh:
        json.dump(
            {"summary": summary, "ranking": relative},
            fh,
            indent=2,
            ensure_ascii=False,
        )
    print(f"wrote {len(summary)} summary rows, {len(relative)} ranking rows -> {args.outdir}")
    if args.markdown:
        print_markdown(records)


if __name__ == "__main__":
    main()
