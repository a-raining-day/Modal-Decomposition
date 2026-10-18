"""
Run the denoising benchmark.

Examples
--------
Full main grid (7 cases x 3 noise levels x 6 proposed configs x 5 reps):

    python -m explore.denoise.run_benchmark --reps 5

Quick smoke test:

    python -m explore.denoise.run_benchmark --cases two_tone trend_tone \
        --reps 1 --out explore/denoise/results/smoke.json
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

from explore.denoise.methods import (  # noqa: E402
    classical_method_fns,
    eemd_corr,
    ideal_lowpass,
    proposed_configs,
    proposed_sensitivity_configs,
    run_proposed,
)
from explore.denoise.metrics import evaluate  # noqa: E402
from explore.denoise.signals import (  # noqa: E402
    CASE_ORDER,
    FS,
    N_DEFAULT,
    make_observation,
)

BASE_SEED = 20261017


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


def _parse_list(text: str) -> list[str]:
    return [part.strip() for part in text.split(",") if part.strip()]


def _parse_floats(text: str) -> list[float]:
    return [float(part) for part in _parse_list(text)]


def _run_and_time(fn) -> tuple[np.ndarray, float]:
    t0 = time.perf_counter()
    out = fn()
    return out, (time.perf_counter() - t0) * 1000.0


def run(
    cases: list[str],
    noise_levels: list[float],
    reps: int,
    n: int,
    out_path: Path,
    include_eemd: bool,
    eemd_cases: list[str],
    eemd_levels: list[float],
    include_sensitivity: bool,
    sensitivity_cases: list[str],
    sensitivity_levels: list[float],
    sensitivity_reps: int,
) -> list[dict]:
    records: list[dict] = []
    main_configs = proposed_configs()
    sensitivity_configs = proposed_sensitivity_configs()

    for rep in range(reps):
        for case_i, case in enumerate(cases):
            levels = [1.0] if case == "pure_noise" else noise_levels
            for level_i, level in enumerate(levels):
                obs_seed = BASE_SEED + rep * 100_000 + case_i * 1_000 + level_i * 10
                obs = make_observation(case, level, seed=obs_seed, n=n)
                base_params = {"noise_level": level, "rep": rep, "case": case}

                # Deterministic baselines
                for name, fn in classical_method_fns(fs=FS).items():
                    y, ms = _run_and_time(lambda fn=fn: fn(obs.noisy))
                    records.append(
                        {
                            **base_params,
                            "method": name,
                            "params": {},
                            "metrics": _sanitize(evaluate(obs, y, fs=FS)),
                            "runtime_ms": ms,
                        }
                    )

                # Oracle low-pass (unfair by design: uses clean bandwidth)
                if case != "pure_noise":
                    y, ms = _run_and_time(lambda: ideal_lowpass(obs.noisy, obs.clean, fs=FS))
                    records.append(
                        {
                            **base_params,
                            "method": "IdealLP",
                            "params": {"note": "oracle cutoff from clean bandwidth"},
                            "metrics": _sanitize(evaluate(obs, y, fs=FS)),
                            "runtime_ms": ms,
                        }
                    )

                # Proposed grid
                for cfg in main_configs:
                    y, ms = _run_and_time(
                        lambda cfg=cfg: run_proposed(obs.noisy, cfg, seed=obs_seed + 7_000)
                    )
                    records.append(
                        {
                            **base_params,
                            "method": cfg["name"],
                            "params": {"K": cfg["K"], "M": cfg["M"], "noise_amp": cfg["noise_amp"]},
                            "metrics": _sanitize(evaluate(obs, y, fs=FS)),
                            "runtime_ms": ms,
                        }
                    )

                # EEMD reference on a small subset
                if (
                    include_eemd
                    and case in eemd_cases
                    and any(abs(level - x) < 1e-12 for x in eemd_levels)
                ):
                    try:
                        y, ms = _run_and_time(
                            lambda: eemd_corr(obs.noisy, seed=obs_seed + 11_000, M=10)
                        )
                        records.append(
                            {
                                **base_params,
                                "method": "EEMD-corr",
                                "params": {"M": 10, "noise_amp": 0.2, "corr_thr": 0.2},
                                "metrics": _sanitize(evaluate(obs, y, fs=FS)),
                                "runtime_ms": ms,
                            }
                        )
                    except Exception as exc:  # keep the benchmark going
                        records.append(
                            {
                                **base_params,
                                "method": "EEMD-corr",
                                "params": {"M": 10},
                                "error": f"{type(exc).__name__}: {exc}",
                            }
                        )

    # Auxiliary-noise sensitivity on a smaller grid
    if include_sensitivity:
        for rep in range(sensitivity_reps):
            for case_i, case in enumerate(sensitivity_cases):
                for level_i, level in enumerate(sensitivity_levels):
                    obs_seed = BASE_SEED + 500_000 + rep * 10_000 + case_i * 1_000 + level_i * 10
                    obs = make_observation(case, level, seed=obs_seed, n=n)
                    for cfg in sensitivity_configs:
                        y, ms = _run_and_time(
                            lambda cfg=cfg: run_proposed(obs.noisy, cfg, seed=obs_seed + 7_000)
                        )
                        records.append(
                            {
                                "case": case,
                                "noise_level": level,
                                "rep": rep,
                                "method": cfg["name"],
                                "params": {
                                    "K": cfg["K"],
                                    "M": cfg["M"],
                                    "noise_amp": cfg["noise_amp"],
                                },
                                "metrics": _sanitize(evaluate(obs, y, fs=FS)),
                                "runtime_ms": ms,
                            }
                        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(records, fh, indent=2, ensure_ascii=False, allow_nan=False)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Denoising benchmark")
    parser.add_argument("--cases", nargs="*", default=CASE_ORDER, choices=CASE_ORDER)
    parser.add_argument("--noise-levels", default="0.1,0.3,0.6")
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--n", type=int, default=N_DEFAULT)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "explore/denoise/results/benchmark.json",
    )
    parser.add_argument("--include-eemd", action="store_true")
    parser.add_argument("--eemd-cases", default="two_tone,trend_tone,impact")
    parser.add_argument("--eemd-levels", default="0.3")
    parser.add_argument("--include-sensitivity", action="store_true")
    parser.add_argument("--sensitivity-cases", default="trend_tone,impact")
    parser.add_argument("--sensitivity-levels", default="0.3")
    parser.add_argument("--sensitivity-reps", type=int, default=3)
    args = parser.parse_args()

    cases = list(args.cases)
    noise_levels = _parse_floats(args.noise_levels)
    eemd_cases = _parse_list(args.eemd_cases)
    eemd_levels = _parse_floats(args.eemd_levels)
    sensitivity_cases = _parse_list(args.sensitivity_cases)
    sensitivity_levels = _parse_floats(args.sensitivity_levels)

    records = run(
        cases=cases,
        noise_levels=noise_levels,
        reps=args.reps,
        n=args.n,
        out_path=args.out,
        include_eemd=args.include_eemd,
        eemd_cases=eemd_cases,
        eemd_levels=eemd_levels,
        include_sensitivity=args.include_sensitivity,
        sensitivity_cases=sensitivity_cases,
        sensitivity_levels=sensitivity_levels,
        sensitivity_reps=args.sensitivity_reps,
    )
    n_ok = sum(1 for r in records if "metrics" in r)
    print(f"wrote {len(records)} records ({n_ok} with metrics) -> {args.out}")


if __name__ == "__main__":
    main()
