"""
Aggregate the joint K x M x amp grid and quantify the "low-parameter" claim.

Definitions
-----------
global best:
    one fixed proposed config with the lowest median shape NRMSE over all
    90 observations.
per-shape oracle (proposed):
    for each shape, the best proposed config (in-sample; optimistic).
LOSO:
    for each held-out shape, pick the config that is best on the other five
    shapes; evaluate on the held-out shape.  This is the honest cost of using
    one fixed configuration without tuning per shape.
oracle classical:
    for each shape, the best fixed baseline among SG/Butter/morphology/raw
    (in-sample; a strong "tuned filter" reference).
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]

KS = (1, 2, 3)
MS = (16, 32, 64)
AMPS = (0.2, 0.3, 0.5, 0.6, 0.8)
CASES = ["two_tone", "trend_tone", "am_fm", "impact", "step", "spike_tone"]

PATTERN = re.compile(r"^P-K(\d+)-M(\d+)-a([0-9.]+)$")


def _parse(method: str):
    match = PATTERN.match(method)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2)), float(match.group(3))


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


def _metric_values(records, metric: str, method: str | None = None, case: str | None = None):
    out = []
    for record in records:
        if "metrics" not in record or metric not in record["metrics"]:
            continue
        if method is not None and record["method"] != method:
            continue
        if case is not None and record["case"] != case:
            continue
        out.append(record["metrics"][metric])
    return out


def build_summary(records: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for record in records:
        if "metrics" not in record:
            continue
        groups[(record["case"], float(record["noise_level"]), record["method"])].append(record)
    rows = []
    for (case, level, method), group in sorted(groups.items()):
        row = {"case": case, "noise_level": level, "method": method, "n": len(group)}
        for metric in ["shape_nrmse", "shape_lp_corr", "snr_imp", "rmse_int", "edge_max"]:
            row[metric] = _median([r["metrics"].get(metric) for r in group])
        row["runtime_ms"] = _median([r.get("runtime_ms") for r in group])
        rows.append(row)
    return rows


def config_stats(records: list[dict], methods: list[str]) -> list[dict]:
    rows = []
    for method in methods:
        shape = _metric_values(records, "shape_nrmse", method=method)
        if not shape:
            continue
        rows.append(
            {
                "method": method,
                "K": _parse(method)[0],
                "M": _parse(method)[1],
                "amp": _parse(method)[2],
                "shape_med": _median(shape),
                "snr_med": _median(_metric_values(records, "snr_imp", method=method)),
                "edge_med": _median(_metric_values(records, "edge_max", method=method)),
                "runtime_med": _median(
                    [r.get("runtime_ms") for r in records if r.get("method") == method]
                ),
            }
        )
    return rows


def per_case_config(records: list[dict], configs: list[str], case: str) -> dict[str, float]:
    out = {}
    for method in configs:
        vals = _metric_values(records, "shape_nrmse", method=method, case=case)
        if vals:
            out[method] = _median(vals)
    return out


def _ranks(values: dict[str, float]) -> dict[str, float]:
    present = [(m, v) for m, v in values.items() if v is not None]
    present.sort(key=lambda item: item[1])
    ranks = {}
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


def bootstrap_median_ci(
    records: list[dict], method: str, n_boot: int = 5000, seed: int = 20261017
) -> dict:
    """Cell-stratified bootstrap CI for the median shape NRMSE.

    Cells are (case, noise_level); each cell already has 5 repetitions, so
    resampling cells respects the case/level structure better than resampling
    all 90 observations independently.
    """
    cells = defaultdict(list)
    for record in records:
        if record.get("method") != method or "metrics" not in record:
            continue
        value = record["metrics"].get("shape_nrmse")
        if value is not None:
            cells[(record["case"], float(record["noise_level"]))].append(value)
    cell_medians = [statistics.median(vals) for vals in cells.values() if vals]
    if not cell_medians:
        return {"point": None, "lo": None, "hi": None, "n_cells": 0}
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        sample = rng.choice(cell_medians, size=len(cell_medians), replace=True)
        boots.append(float(np.median(sample)))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {
        "point": float(np.median(cell_medians)),
        "lo": float(lo),
        "hi": float(hi),
        "n_cells": len(cell_medians),
    }


def mean_ranks(records: list[dict], methods: list[str]) -> dict[str, float]:
    by_obs = defaultdict(dict)
    for record in records:
        if "metrics" not in record or "shape_nrmse" not in record["metrics"]:
            continue
        key = (record["case"], float(record["noise_level"]), record["rep"])
        by_obs[key][record["method"]] = record["metrics"]["shape_nrmse"]
    ranks = defaultdict(list)
    for values in by_obs.values():
        for method, rank in _ranks(values).items():
            if method in methods:
                ranks[method].append(rank)
    return {method: _mean(ranks[method]) for method in methods if ranks.get(method)}


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


def analyze(records: list[dict]):
    configs = [m for m in {r["method"] for r in records} if _parse(m)]
    configs.sort(key=lambda m: (_parse(m)[0], _parse(m)[1], _parse(m)[2]))
    baselines = sorted({r["method"] for r in records if not _parse(r["method"])})

    stats = config_stats(records, configs)
    stats.sort(key=lambda r: r["shape_med"])
    rank_all = mean_ranks(records, configs)
    for row in stats:
        row["mean_rank_all"] = rank_all.get(row["method"])

    global_best = stats[0]
    global_name = global_best["method"]
    global_shape = global_best["shape_med"]

    per_case = {case: per_case_config(records, configs, case) for case in CASES}
    oracle_p = {}
    for case in CASES:
        method = min(per_case[case], key=per_case[case].get)
        oracle_p[case] = {"method": method, "shape": per_case[case][method]}

    oracle_baseline = {}
    baseline_case = {}
    for case in CASES:
        values = {
            method: _median(_metric_values(records, "shape_nrmse", method=method, case=case))
            for method in baselines
        }
        values = {m: v for m, v in values.items() if v is not None}
        baseline_case[case] = values
        method = min(values, key=values.get)
        oracle_baseline[case] = {"method": method, "shape": values[method]}

    # LOSO over the tunable classical/morphology baselines (raw is just "do
    # nothing", so it is not selected by the tuning procedure).
    tunable_baselines = [m for m in baselines if m != "raw"]
    baseline_loso = {}
    for case in CASES:
        other = [c for c in CASES if c != case]
        scores = {
            method: _median([baseline_case[c][method] for c in other if method in baseline_case[c]])
            for method in tunable_baselines
        }
        scores = {m: v for m, v in scores.items() if v is not None}
        chosen = min(scores, key=scores.get)
        baseline_loso[case] = {"method": chosen, "shape": baseline_case[case][chosen]}

    # LOSO: select on other cases, evaluate on held-out case
    loso = {}
    for case in CASES:
        other = [c for c in CASES if c != case]
        scores = {}
        for method in configs:
            vals = []
            for c in other:
                if method in per_case[c]:
                    vals.append(per_case[c][method])
            if vals:
                scores[method] = _median(vals)
        chosen = min(scores, key=scores.get)
        loso[case] = {"method": chosen, "shape": per_case[case][chosen]}

    case_table = []
    for case in CASES:
        g = per_case[case].get(global_name)
        o = oracle_p[case]["shape"]
        b = oracle_baseline[case]["shape"]
        l = loso[case]["shape"]
        bl = baseline_loso[case]["shape"]
        case_table.append(
            {
                "case": case,
                "global_shape": g,
                "oracle_p_method": oracle_p[case]["method"],
                "oracle_p_shape": o,
                "loss_vs_oracle_p": None if (g is None or o is None) else g - o,
                "loso_method": loso[case]["method"],
                "loso_shape": l,
                "loss_vs_loso": None if (g is None or l is None) else g - l,
                "baseline_loso_method": baseline_loso[case]["method"],
                "baseline_loso_shape": bl,
                "loss_vs_baseline_loso": None if (g is None or bl is None) else g - bl,
                "oracle_baseline_method": oracle_baseline[case]["method"],
                "oracle_baseline_shape": b,
                "global_minus_oracle_baseline": None if (g is None or b is None) else g - b,
            }
        )

    global_case_mean = _mean([r["global_shape"] for r in case_table])
    oracle_p_mean = _mean([r["oracle_p_shape"] for r in case_table])
    loso_mean = _mean([r["loso_shape"] for r in case_table])
    baseline_mean = _mean([r["oracle_baseline_shape"] for r in case_table])
    baseline_loso_mean = _mean([r["shape"] for r in baseline_loso.values()])
    top3_ci = [
        {"method": row["method"], **bootstrap_median_ci(records, row["method"])}
        for row in stats[:3]
    ]
    cost_summary = {
        "global_aggregate_median": global_shape,
        "global_case_mean": global_case_mean,
        "oracle_p_mean": oracle_p_mean,
        "loso_mean": loso_mean,
        "oracle_baseline_mean": baseline_mean,
        "baseline_loso_mean": baseline_loso_mean,
        "global_minus_baseline_loso": None
        if (global_case_mean is None or baseline_loso_mean is None)
        else global_case_mean - baseline_loso_mean,
        "global_minus_oracle_p": None
        if (global_case_mean is None or oracle_p_mean is None)
        else global_case_mean - oracle_p_mean,
        "global_minus_loso": None
        if (global_case_mean is None or loso_mean is None)
        else global_case_mean - loso_mean,
        "global_minus_oracle_baseline": None
        if (global_case_mean is None or baseline_mean is None)
        else global_case_mean - baseline_mean,
    }

    # plateau: configs within +1% and +5% of the global best shape
    plateau_1pct = [r for r in stats if r["shape_med"] <= global_shape * 1.01]
    plateau_5pct = [r for r in stats if r["shape_med"] <= global_shape * 1.05]

    def _ranges(rows):
        return {
            "n": len(rows),
            "K": sorted({r["K"] for r in rows}),
            "M": sorted({r["M"] for r in rows}),
            "amp": sorted({r["amp"] for r in rows}),
            "shape_max": max((r["shape_med"] for r in rows), default=None),
        }

    marginal = {}
    for param in ("K", "M", "amp"):
        vals = {}
        for value in {"K": KS, "M": MS, "amp": AMPS}[param]:
            rows = [r for r in stats if r[param] == value]
            vals[value] = _median([r["shape_med"] for r in rows])
        marginal[param] = vals

    return {
        "stats": stats,
        "global_best": global_best,
        "case_table": case_table,
        "oracle_p": oracle_p,
        "oracle_baseline": oracle_baseline,
        "baseline_loso": baseline_loso,
        "top3_ci": top3_ci,
        "loso": loso,
        "plateau_1pct": _ranges(plateau_1pct),
        "plateau_5pct": _ranges(plateau_5pct),
        "marginal": marginal,
        "cost_summary": cost_summary,
        "n_configs": len(configs),
        "n_baselines": len(baselines),
    }


def print_markdown(result: dict) -> None:
    stats = result["stats"]
    print("\n### Top proposed configs (global, median over 90 observations)\n")
    print("| config | shape NRMSE | SNR imp. | edge max | mean rank (all methods) |")
    print("|---|---:|---:|---:|---:|")
    for row in stats[:12]:
        print(
            f"| {row['method']} | {_fmt(row['shape_med'])} | {_fmt(row['snr_med'], 2)} | "
            f"{_fmt(row['edge_med'])} | {_fmt(row['mean_rank_all'], 2)} |"
        )

    print("\n### P0: top-3 configs with cell-bootstrap 95% CI\n")
    print("| config | shape (90 obs) | bootstrap median | 95% CI |")
    print("|---|---:|---:|---|")
    stat_map = {row["method"]: row for row in stats}
    for item in result["top3_ci"]:
        row = stat_map[item["method"]]
        print(
            f"| {item['method']} | {_fmt(row['shape_med'])} | {_fmt(item['point'])} | "
            f"[{_fmt(item['lo'])}, {_fmt(item['hi'])}] |"
        )

    print("\n### P0: LOSO over classical/morphology baselines\n")
    print("| case | LOSO baseline | shape | oracle baseline | shape | LOSO - oracle |")
    print("|---|---|---:|---|---:|---:|")
    for row in result["case_table"]:
        loss = None
        if row["baseline_loso_shape"] is not None and row["oracle_baseline_shape"] is not None:
            loss = row["baseline_loso_shape"] - row["oracle_baseline_shape"]
        print(
            f"| {row['case']} | {row['baseline_loso_method']} | {_fmt(row['baseline_loso_shape'])} | "
            f"{row['oracle_baseline_method']} | {_fmt(row['oracle_baseline_shape'])} | {_fmt(loss)} |"
        )
    print(f"\n- classical LOSO mean shape: **{_fmt(result['cost_summary']['baseline_loso_mean'])}**")
    print(f"- classical oracle mean shape: **{_fmt(result['cost_summary']['oracle_baseline_mean'])}**")

    print("\n### Per-shape: global config vs oracle proposed vs LOSO vs oracle baseline\n")
    print("| case | global shape | oracle P (config) | oracle P shape | loss | LOSO shape | oracle baseline (method) | oracle C shape | global - oracleC |")
    print("|---|---:|---|---:|---:|---:|---|---:|---:|")
    for row in result["case_table"]:
        print(
            f"| {row['case']} | {_fmt(row['global_shape'])} | "
            f"{row['oracle_p_method']} | {_fmt(row['oracle_p_shape'])} | "
            f"{_fmt(row['loss_vs_oracle_p'])} | {_fmt(row['loso_shape'])} | "
            f"{row['oracle_baseline_method']} | {_fmt(row['oracle_baseline_shape'])} | "
            f"{_fmt(row['global_minus_oracle_baseline'])} |"
        )

    print("\n### Cost of one fixed configuration\n")
    cost = result["cost_summary"]
    print(f"- global best `{result['global_best']['method']}`: aggregate median **{_fmt(cost['global_aggregate_median'])}**, per-case mean **{_fmt(cost['global_case_mean'])}**")
    print(f"- per-shape oracle proposed: mean shape **{_fmt(cost['oracle_p_mean'])}**")
    print(f"- LOSO fixed config: mean shape **{_fmt(cost['loso_mean'])}**")
    print(f"- per-shape oracle classical/morph: mean shape **{_fmt(cost['oracle_baseline_mean'])}**")
    print(f"- LOSO classical/morph: mean shape **{_fmt(cost['baseline_loso_mean'])}**")
    print(f"- global per-case mean vs oracle proposed: **{_fmt(cost['global_minus_oracle_p'])}**")
    print(f"- global per-case mean vs oracle classical: **{_fmt(cost['global_minus_oracle_baseline'])}**")

    print("\n### Parameter plateau\n")
    for label, key in (("within +1%", "plateau_1pct"), ("within +5%", "plateau_5pct")):
        p = result[key]
        print(
            f"- {label}: {p['n']} configs; K={p['K']}, M={p['M']}, amp={p['amp']}; "
            f"worst shape {_fmt(p['shape_max'])}"
        )

    print("\n### Marginal medians (config-level medians)\n")
    print("| param | values (shape NRMSE) |")
    print("|---|---|")
    for param, vals in result["marginal"].items():
        body = ", ".join(f"{k:g}: {_fmt(v)}" for k, v in vals.items())
        print(f"| {param} | {body} |")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "explore/denoise/results/joint_grid.json",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=ROOT / "explore/denoise/results",
    )
    parser.add_argument("--markdown", action="store_true")
    args = parser.parse_args()

    records = _load(args.input)
    result = analyze(records)

    write_csv(
        args.outdir / "joint_grid_configs.csv",
        result["stats"],
        ["method", "K", "M", "amp", "shape_med", "snr_med", "edge_med", "runtime_med", "mean_rank_all"],
    )
    write_csv(
        args.outdir / "joint_grid_cases.csv",
        result["case_table"],
        [
            "case", "global_shape", "oracle_p_method", "oracle_p_shape", "loss_vs_oracle_p",
            "loso_method", "loso_shape", "loss_vs_loso",
            "baseline_loso_method", "baseline_loso_shape", "loss_vs_baseline_loso",
            "oracle_baseline_method", "oracle_baseline_shape", "global_minus_oracle_baseline",
        ],
    )
    with (args.outdir / "joint_grid_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, ensure_ascii=False)
    print(
        f"wrote {len(result['stats'])} configs, {len(result['case_table'])} case rows "
        f"-> {args.outdir}"
    )
    if args.markdown:
        print_markdown(result)


if __name__ == "__main__":
    main()
