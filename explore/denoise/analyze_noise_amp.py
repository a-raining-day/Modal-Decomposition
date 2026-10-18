"""
Aggregate the high-noise auxiliary-noise amplitude sweep.

Outputs:
    explore/denoise/results/high_noise_noise_amp_summary.csv
    explore/denoise/results/high_noise_noise_amp_ranking.csv
    explore/denoise/results/high_noise_noise_amp_summary.json
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

BASE_METHODS = ["raw", "SG11", "Butter150", "MorphAvg5"]
AMPS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2)
CASES = ["two_tone", "trend_tone", "am_fm", "impact", "step", "spike_tone"]
LEVELS = (0.5, 0.7, 0.9)


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


def _amp(method: str) -> float | None:
    if not method.startswith("P-K2-M32-a"):
        return None
    try:
        return float(method.split("-a", 1)[1])
    except ValueError:
        return None


def build_summary(records: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for record in records:
        if "metrics" not in record:
            continue
        groups[(record["case"], float(record["noise_level"]), record["method"])].append(record)
    rows = []
    for (case, level, method), group in sorted(groups.items()):
        row = {"case": case, "noise_level": level, "method": method, "n": len(group)}
        for metric in [
            "shape_nrmse", "shape_lp_corr", "snr_imp", "rmse_int", "edge_max",
            "trend_lp_rmse", "weak_retention", "env_peak_snr", "env_corr",
            "impact_amp_ratio", "edge_width_samples", "overshoot", "plateau_rmse",
            "spike_leakage", "clean_region_rmse",
        ]:
            row[metric] = _median([r["metrics"].get(metric) for r in group])
        row["aux_ratio"] = _median([r.get("aux_ratio") for r in group])
        row["runtime_ms"] = _median([r.get("runtime_ms") for r in group])
        rows.append(row)
    return rows


def _ranks(values: dict[str, float]) -> dict[str, float]:
    present = [(m, v) for m, v in values.items() if v is not None]
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
        by_obs[(record["case"], float(record["noise_level"]), record["rep"])][record["method"]] = record
    ranks = defaultdict(list)
    for by_method in by_obs.values():
        values = {
            method: record["metrics"].get("shape_nrmse")
            for method, record in by_method.items()
            if record["metrics"].get("shape_nrmse") is not None
        }
        for method, rank in _ranks(values).items():
            ranks[method].append(rank)

    rows = []
    for method in BASE_METHODS + [f"P-K2-M32-a{a:g}" for a in AMPS]:
        if method not in ranks:
            continue
        rows.append(
            {
                "method": method,
                "amp": _amp(method),
                "mean_rank": _mean(ranks[method]),
                "median_rank": _median(ranks[method]),
                "worst_rank": max(ranks[method]),
                "rank_std": _std(ranks[method]),
                "n_obs": len(ranks[method]),
            }
        )
    rows.sort(key=lambda r: (r["mean_rank"] is None, r["mean_rank"]))
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
        return f"{value:.{digits}f}"
    return str(value)


def _amp_rows(summary: list[dict], amp: float) -> list[dict]:
    return [r for r in summary if r["method"] == f"P-K2-M32-a{amp:g}"]


def _raw_values(records: list[dict], amp: float, metric: str) -> list[float]:
    name = f"P-K2-M32-a{amp:g}"
    return [
        r["metrics"][metric]
        for r in records
        if r.get("method") == name and "metrics" in r and metric in r["metrics"]
    ]


def _raw_metric(
    records: list[dict],
    amp: float,
    metric: str,
    case: str | None = None,
    level: float | None = None,
) -> list[float]:
    name = f"P-K2-M32-a{amp:g}"
    out = []
    for r in records:
        if r.get("method") != name or "metrics" not in r or metric not in r["metrics"]:
            continue
        if case is not None and r["case"] != case:
            continue
        if level is not None and abs(float(r["noise_level"]) - level) > 1e-12:
            continue
        out.append(r["metrics"][metric])
    return out


def print_markdown(records: list[dict]) -> None:
    summary = build_summary(records)
    ranking = build_ranking(records)

    print("\n### Amplitude sweep (median over 90 observations; rank among 15 methods)\n")
    print("| amp | aux/input noise | shape NRMSE | SNR imp. (dB) | edge max | mean rank |")
    print("|---:|---:|---:|---:|---:|---:|")
    rank_map = {r["method"]: r["mean_rank"] for r in ranking}
    for amp in AMPS:
        name = f"P-K2-M32-a{amp:g}"
        ratio = _median([r.get("aux_ratio") for r in records if r.get("method") == name])
        shape = _median(_raw_values(records, amp, "shape_nrmse"))
        snr = _median(_raw_values(records, amp, "snr_imp"))
        edge = _median(_raw_values(records, amp, "edge_max"))
        print(
            f"| {amp:g} | {_fmt(ratio, 2)} | {_fmt(shape)} | {_fmt(snr, 2)} | "
            f"{_fmt(edge)} | {_fmt(rank_map.get(name), 2)} |"
        )

    print("\n### Per-noise-level comparison\n")
    print("| input noise | match amp | shape @0.2 | shape @match | shape @amp=level | best amp | best shape |")
    print("|---:|---:|---:|---:|---:|---:|---:|")
    for level in LEVELS:
        match_amp = level / (1.0 + level**2) ** 0.5
        shape_02 = _median(_raw_metric(records, 0.2, "shape_nrmse", level=level))
        # nearest grid amp to the theoretical match
        grid = min(AMPS, key=lambda a: abs(a - match_amp))
        shape_match = _median(_raw_metric(records, grid, "shape_nrmse", level=level))
        shape_level = _median(_raw_metric(records, level, "shape_nrmse", level=level))
        best_amp = None
        best_shape = None
        for amp in AMPS:
            v = _median(_raw_metric(records, amp, "shape_nrmse", level=level))
            if v is not None and (best_shape is None or v < best_shape):
                best_shape, best_amp = v, amp
        print(
            f"| {level:g} | {match_amp:.2f} ({grid:g}) | {_fmt(shape_02)} | "
            f"{_fmt(shape_match)} | {_fmt(shape_level)} | {best_amp:g} | {_fmt(best_shape)} |"
        )

    print("\n### Per-case best amplitude (median over 3 noise levels x 5 reps)\n")
    print("| case | shape @0.2 | best amp | best shape | gain vs 0.2 |")
    print("|---|---:|---:|---:|---:|")
    for case in CASES:
        shape_02 = _median(_raw_metric(records, 0.2, "shape_nrmse", case=case))
        best_amp = None
        best_shape = None
        for amp in AMPS:
            v = _median(_raw_metric(records, amp, "shape_nrmse", case=case))
            if v is not None and (best_shape is None or v < best_shape):
                best_shape, best_amp = v, amp
        gain = None if (shape_02 is None or best_shape is None) else shape_02 - best_shape
        print(f"| {case} | {_fmt(shape_02)} | {best_amp:g} | {_fmt(best_shape)} | {_fmt(gain)} |")

    print("\n### Auxiliary/input ratio bins (all proposed observations)\n")
    bins = [(0.0, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, 3.0)]
    print("| ratio bin | shape NRMSE | SNR imp. | n |")
    print("|---|---:|---:|---:|")
    for lo, hi in bins:
        rows = [
            r for r in summary
            if r["aux_ratio"] is not None and lo <= r["aux_ratio"] < hi
        ]
        print(
            f"| [{lo:g}, {hi:g}) | {_fmt(_median([r['shape_nrmse'] for r in rows]))} | "
            f"{_fmt(_median([r['snr_imp'] for r in rows]), 2)} | {len(rows)} |"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "explore/denoise/results/high_noise_noise_amp.json",
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

    fields = ["case", "noise_level", "method", "n", "aux_ratio"] + [
        "shape_nrmse", "shape_lp_corr", "snr_imp", "rmse_int", "edge_max",
        "trend_lp_rmse", "weak_retention", "env_peak_snr", "env_corr",
        "impact_amp_ratio", "edge_width_samples", "overshoot", "plateau_rmse",
        "spike_leakage", "clean_region_rmse", "runtime_ms",
    ]
    write_csv(args.outdir / "high_noise_noise_amp_summary.csv", summary, fields)
    write_csv(
        args.outdir / "high_noise_noise_amp_ranking.csv",
        ranking,
        ["method", "amp", "mean_rank", "median_rank", "worst_rank", "rank_std", "n_obs"],
    )
    with (args.outdir / "high_noise_noise_amp_summary.json").open("w", encoding="utf-8") as fh:
        json.dump({"summary": summary, "ranking": ranking}, fh, indent=2, ensure_ascii=False)
    print(f"wrote {len(summary)} summary rows, {len(ranking)} ranking rows -> {args.outdir}")
    if args.markdown:
        print_markdown(records)


if __name__ == "__main__":
    main()
