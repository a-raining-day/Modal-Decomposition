"""
Regular-length EMD benchmark: wall time, RSS delta and decomposition quality
for the three implementations on deterministic signals.

One cell is (implementation, case, length). Every cell runs ``--repeats``
times with round-robin implementation order so drift between repeats hits all
implementations equally; imports / warm-ups are excluded from measurements.
Raw per-run rows stream to ``results/timing_raw.csv`` (crash-resilient), and
quality metrics (computed once per cell on the first repeat) stream to
``results/timing_metrics.json``.

CLI
---
    python tests/comparison/bench_timing.py                      # full grid
    python tests/comparison/bench_timing.py --cases A B          # subset
    python tests/comparison/bench_timing.py --ns 4096 16384      # lengths
    python tests/comparison/bench_timing.py --cases C --noise-cap 65536

``--noise-cap`` limits the white-noise case (C) to at most that many samples;
noise is the most expensive input, so the cap keeps default runs bounded.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import threading
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[1]
for _p in (str(_HERE), str(_REPO / "src"), str(_REPO / "ref")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import psutil

from quality import analyze
from signals import build_case, case_desc, default_ns
from workers import (
    CASE_K,
    EFD_IMPL_ORDER,
    FMD_IMPL_ORDER,
    IMPL_ORDER as EMD_IMPL_ORDER,
    LMD_IMPL_ORDER,
    VMD_IMPL_ORDER,
    run_efd_stack,
    run_fmd_stack,
    run_lmd_stack,
    run_stack,
    run_vmd_stack,
    versions,
)

DEFAULT_RESULTS = _HERE / "results"


class _RssSampler:
    """Sampled RSS around one measured call (1 ms interval, like the memory
    pipeline's PeakMonitor)."""

    def __init__(self, interval_s: float = 0.001):
        self.interval_s = interval_s

    def measure(self, fn):
        baseline = int(psutil.Process().memory_info().rss)
        samples: list[int] = []
        stop = threading.Event()

        def _sample():
            while not stop.is_set():
                samples.append(int(psutil.Process().memory_info().rss))
                stop.wait(self.interval_s)

        th = threading.Thread(target=_sample, daemon=True)
        start = time.perf_counter()
        th.start()
        try:
            result = fn()
        finally:
            wall_s = time.perf_counter() - start
            stop.set()
            th.join()
            samples.append(int(psutil.Process().memory_info().rss))
            samples.append(int(psutil.Process().memory_info().rss))
        peak = max(samples) if samples else baseline
        return result, {
            "wall_s": wall_s,
            "baseline_rss_bytes": baseline,
            "peak_rss_bytes": peak,
            "delta_rss_bytes": peak - baseline,
            "n_samples": len(samples),
        }


_CSV_FIELDS = [
    "impl", "case", "n", "repeat",
    "wall_s", "delta_rss_bytes", "peak_rss_bytes",
    "n_rows", "recon_max_abs_err", "orthogonality_index",
    "note",
]


def _row(cell: dict, timing: dict, metrics: dict, note: str = "") -> dict:
    return {
        "impl": cell["impl"],
        "case": cell["case"],
        "n": cell["n"],
        "repeat": cell["repeat"],
        "wall_s": round(timing["wall_s"], 6),
        "delta_rss_bytes": timing["delta_rss_bytes"],
        "peak_rss_bytes": timing["peak_rss_bytes"],
        "n_rows": metrics["n_rows"],
        "recon_max_abs_err": metrics["recon_max_abs_err"],
        "orthogonality_index": metrics["orthogonality_index"],
        "note": note,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cases", nargs="+", default=["A", "B", "C"],
                    help="case names (A/B/C); default all three")
    ap.add_argument("--ns", nargs="+", type=int, default=None,
                    help="sample counts; default %(default)s")
    ap.add_argument("--noise-cap", type=int, default=16384,
                    help="max samples for the white-noise case C (default 16384)")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--per-run-cap", type=float, default=0.0,
                    help="abort one run when it exceeds this many seconds (0 = off)")
    ap.add_argument("--method", choices=["EMD", "VMD", "LMD", "FMD", "EFD"],
                    default="EMD",
                    help="which decomposition method to benchmark")
    ap.add_argument("--impls", nargs="+", default=None,
                    help="subset of implementations to run")
    ap.add_argument("--out", type=Path, default=None,
                    help="results dir; defaults to results/<method>")
    args = ap.parse_args()

    method = args.method
    if method == "EMD":
        impl_order = EMD_IMPL_ORDER
    elif method == "VMD":
        impl_order = VMD_IMPL_ORDER
    elif method == "LMD":
        impl_order = LMD_IMPL_ORDER
    elif method == "FMD":
        impl_order = FMD_IMPL_ORDER
    else:
        impl_order = EFD_IMPL_ORDER
    if args.impls:
        impl_order = [i for i in impl_order if i in args.impls]
    out = Path(args.out) if args.out is not None else DEFAULT_RESULTS / method.lower()
    out.mkdir(parents=True, exist_ok=True)
    ns_default = default_ns()
    ns = args.ns if args.ns is not None else ns_default
    repeats = max(1, args.repeats)
    noise_cap = args.noise_cap

    csv_path = out / "timing_raw.csv"
    new_file = not csv_path.exists()
    fh = open(csv_path, "a", newline="", encoding="utf-8")
    writer = __import__("csv").DictWriter(fh, fieldnames=_CSV_FIELDS)
    if new_file:
        writer.writeheader()

    metrics_path = out / "timing_metrics.json"
    metrics_all: list[dict] = []
    if metrics_path.exists():
        metrics_all = json.loads(metrics_path.read_text(encoding="utf-8"))

    env_path = out / "env.json"
    if not env_path.exists():
        env_path.write_text(
            json.dumps({"bench_timing_env": versions()}, indent=2), encoding="utf-8"
        )

    sampler = _RssSampler()
    warm = {impl: False for impl in impl_order}

    def run_one(impl: str, S, case: str):
        if method == "VMD":
            return run_vmd_stack(impl, S, K=CASE_K[case])
        if method == "LMD":
            return run_lmd_stack(impl, S)
        if method == "FMD":
            return run_fmd_stack(impl, S)
        if method == "EFD":
            return run_efd_stack(impl, S)
        return run_stack(impl, S)

    def warm_up():
        S0, _ = build_case("A", 256)
        for impl in impl_order:
            if warm[impl]:
                continue
            run_one(impl, S0, "A")  # imports + lazy engine init land here
            warm[impl] = True

    warm_up()
    print(f"method={method} impls={impl_order} writing raw rows to {csv_path} "
          f"(repeats={repeats})")

    for case in args.cases:
        grid = [n for n in ns if case != "C" or n <= noise_cap]
        if not grid:
            print(f"case {case}: no lengths (noise cap {noise_cap}) - skipped")
            continue
        print(f"== case {case} ({case_desc(case)}), n = {grid}")
        for n in grid:
            S, modes = build_case(case, n)
            S_orig_sum = float(np.sum(S))
            metrics_done = False
            for rep in range(repeats):
                for impl in impl_order:
                    cell = {"impl": impl, "case": case, "n": n, "repeat": rep}
                    try:
                        stack, timing = sampler.measure(
                            lambda: run_one(impl, S, case)
                        )
                    except Exception as exc:
                        rec = _row(cell, {"wall_s": float("nan"),
                                          "delta_rss_bytes": None,
                                          "peak_rss_bytes": None},
                                   {"n_rows": None, "recon_max_abs_err": None,
                                    "orthogonality_index": None},
                                   note=f"error: {type(exc).__name__}: {exc}")
                        writer.writerow(rec)
                        fh.flush()
                        print(f"  {impl:7s} {case} n={n:<6d} rep{rep} ERROR "
                              f"{type(exc).__name__}")
                        break
                    if timing["wall_s"] > args.per_run_cap > 0:
                        rec = _row(cell, timing,
                                   {"n_rows": None, "recon_max_abs_err": None,
                                    "orthogonality_index": None},
                                   note=f"over per-run cap {args.per_run_cap}s")
                        writer.writerow(rec)
                        fh.flush()
                        print(f"  {impl:7s} {case} n={n:<6d} rep{rep} OVER CAP")
                        break

                    metrics = analyze(S, stack, modes)
                    rec = _row(cell, timing, metrics)
                    writer.writerow(rec)
                    fh.flush()
                    if not metrics_done:
                        metrics_all.append({
                            "impl": impl, "case": case, "n": n,
                            **metrics,
                        })
                    print(f"  {impl:7s} {case} n={n:<6d} rep{rep}: "
                          f"{timing['wall_s']:8.4f}s  rows={metrics['n_rows']}")

                    if not np.isclose(float(np.sum(S)), S_orig_sum, rtol=0, atol=1e-9):
                        raise RuntimeError(
                            f"{impl} modified the input signal (case {case}, n={n})"
                        )
                    del stack
                    gc.collect()
                metrics_done = True

    fh.close()
    metrics_path.write_text(
        json.dumps(metrics_all, indent=2), encoding="utf-8"
    )
    print("done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
