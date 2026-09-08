"""
Memory-axis driver: run the three EMD implementations over a data-size grid,
each cell in a fresh subprocess (``case_worker.py``) with a wall budget.

Cell grid (all float64, mirroring the ``tests/test_memory`` conventions):

    implementation  x  data size          x  pattern
    {MD-EMD,           {1MB 20MB 100MB       {increasing, random}
     PyEMD,             500MB 1GB}
     PySDKit}

Unlike the timing axis, cells here use *inputs as large as real deployments*:
the decomposition budget counts only after the input is fully built (parent
waits for the child's phase-1 report), so timeouts measure decomposition
throughput at a given data size, and the child's RSS heartbeat leaves a memory
growth trace even for cells killed by the budget.

Deferred cells (recorded, not executed -- see POSTPONED_3GB.md):

* every size >= 2 GB (the 3 GB cells of the tests/test_memory matrix): the
  earlier matrix already shows pure-Python sifting cannot finish them in any
  reasonable budget; they only cost disk and machine time;
* ``random`` noise at size >= 512 MB on this 16 GB machine: a single sifting
  step on >= 512 MB of float64 noise needs multi-GB working arrays; the whole
  machine starts swapping before any comparative information appears.

Pass ``--no-defer`` to execute them anyway (e.g. on a larger machine).

CLI
---
    python tests/comparison/bench_memory.py                      # full grid
    python tests/comparison/bench_memory.py --sizes 1MB 20MB
    python tests/comparison/bench_memory.py --sizes 1GB --patterns increasing
    python tests/comparison/bench_memory.py --impls PyEMD PySDKit --budget 60
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[1]
for _p in (str(_HERE), str(_REPO / "src"), str(_REPO / "ref")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from workers import IMPL_ORDER, versions

CASE_WORKER = _HERE / "case_worker.py"
DEFAULT_OUT = _HERE / "results" / "memory"

# Defaults follow the tests/test_memory matrix (float64 cell only).
DEFAULT_SIZES = ["1MB", "20MB", "100MB", "500MB", "1GB"]
DEFAULT_PATTERNS = ["increasing", "random"]
DEFER_GB_LIMIT = 2 * 1024 ** 3        # sizes >= this are always deferred
DEFER_RANDOM_FROM = 512 * 1024 ** 2   # random >= this is deferred unless --no-defer

_CSV_FIELDS = [
    "impl", "size_bytes", "n_samples", "pattern", "dtype", "status",
    "dec_wall_s", "dec_baseline_rss_bytes", "dec_peak_rss_bytes",
    "dec_delta_rss_bytes", "hb_n",
    "input_backing", "input_build_wall_s", "input_build_delta_rss_bytes",
    "n_rows", "recon_max_abs_err", "error",
]


def _parse_size(token: str) -> int:
    t = str(token).strip().lower()
    if t.endswith("mb"):
        return int(float(t[:-2]) * 1024 ** 2)
    if t.endswith("gb"):
        return int(float(t[:-2]) * 1024 ** 3)
    if t.endswith("kb"):
        return int(float(t[:-2]) * 1024)
    return int(t)


def _defer_reason(cfg: dict) -> str | None:
    if cfg["size_bytes"] >= DEFER_GB_LIMIT:
        return (f"size >= {DEFER_GB_LIMIT // 2**30}GB is postponed - see "
                f"POSTPONED_3GB.md (earlier matrix: 3GB cells all timeout)")
    if cfg["pattern"] == "random" and cfg["size_bytes"] >= DEFER_RANDOM_FROM:
        return (f"random noise at >= {DEFER_RANDOM_FROM // 2**20}MB is postponed "
                f"on this 16GB machine - see POSTPONED_3GB.md (pass --no-defer "
                f"on a larger machine)")
    return None


def _run_cell(cfg: dict, budget_s: float, build_timeout_s: float) -> dict:
    runs = Path(cfg["out_dir"]) / "_runs"
    runs.mkdir(parents=True, exist_ok=True)
    key = f"{cfg['impl']}_{cfg['size_bytes']}_{cfg['pattern']}_{cfg['dtype']}"
    cfg_path = runs / f"{key}.cfg.json"
    report_path = runs / f"{key}.report.json"
    hb_path = report_path.with_name(report_path.stem + ".hb.log")
    phase1_path = report_path.with_name(report_path.stem + ".phase1.json")
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")
    for p in (report_path, hb_path, phase1_path):
        p.unlink(missing_ok=True)

    try:
        proc = subprocess.Popen(
            [sys.executable, str(CASE_WORKER), str(cfg_path), str(report_path)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    except Exception as exc:
        return {"cell": cfg, "decompose": {"status": "error",
                "error": f"spawn failed: {type(exc).__name__}: {exc}"}}

    t_start = time.monotonic()
    t_phase1 = None
    decomp_deadline: float | None = None
    timed_out = False
    killed_phase = None
    while True:
        proc.poll()
        if proc.returncode is not None:
            break
        now = time.monotonic()
        if t_phase1 is None:
            if phase1_path.is_file():
                t_phase1 = now
                decomp_deadline = now + budget_s
            elif now - t_start > build_timeout_s:
                killed_phase = "input-build"
                proc.kill()
                break
        elif now > decomp_deadline:
            killed_phase = "decompose"
            proc.kill()
            break
        time.sleep(0.05)

    proc.communicate(timeout=30)

    record: dict = {"cell": cfg}
    if report_path.is_file():
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        record["input"] = payload.get("input")
        record["decompose"] = payload.get("decompose")
        record["result"] = payload.get("result")
    else:
        partial = {}
        if phase1_path.is_file():
            partial = json.loads(phase1_path.read_text(encoding="utf-8"))
            if partial.get("phase") == "input":
                record["input"] = partial.get("input")
        status = "timeout" if killed_phase else (
            "crash" if proc.returncode != 0 else "error")
        record["decompose"] = {
            "status": status,
            "error": (f"killed after {killed_phase} budget"
                      if killed_phase else f"exit code {proc.returncode}"),
        }
        if proc.returncode == 0 and not killed_phase:
            record["decompose"] = {
                "status": "error",
                "error": "child exited before writing the report",
            }

    # RSS heartbeat summary for cells that did start decomposing.
    hb_summary = None
    if hb_path.is_file():
        try:
            lines = [ln for ln in hb_path.read_text(encoding="utf-8").splitlines()
                     if ln and not ln.startswith("#")]
            if lines:
                vals = [int(ln.split(",")[1]) for ln in lines]
                first = float(lines[0].split(",")[0])
                last = float(lines[-1].split(",")[0])
                hb_summary = {
                    "n_samples": len(vals),
                    "first_t_monotonic": first,
                    "last_t_monotonic": last,
                    "window_s": round(last - first, 3),
                    "first_rss_bytes": vals[0],
                    "last_rss_bytes": vals[-1],
                    "peak_rss_bytes": max(vals),
                }
        except Exception as exc:  # pragma: no cover
            hb_summary = {"error": f"{type(exc).__name__}: {exc}"}
    record["hb"] = hb_summary
    record["_killed_phase"] = killed_phase

    # Remove backing files left by killed children.
    inp = record.get("input") or {}
    backing = inp.get("backing_file")
    if backing and Path(backing).exists():
        try:
            Path(backing).unlink()
        except OSError:
            pass
    return record


def _flatten(record: dict) -> dict:
    cell = record.get("cell", {})
    dec = record.get("decompose") or {}
    inp = record.get("input") or {}
    res = record.get("result") or {}
    return {
        "impl": cell.get("impl"),
        "size_bytes": cell.get("size_bytes"),
        "n_samples": inp.get("n_samples"),
        "pattern": cell.get("pattern"),
        "dtype": cell.get("dtype"),
        "status": dec.get("status"),
        "dec_wall_s": dec.get("wall_s"),
        "dec_baseline_rss_bytes": dec.get("baseline_rss_bytes"),
        "dec_peak_rss_bytes": dec.get("peak_rss_bytes"),
        "dec_delta_rss_bytes": dec.get("delta_rss_bytes"),
        "hb_n": (record.get("hb") or {}).get("n_samples"),
        "input_backing": inp.get("backing"),
        "input_build_wall_s": inp.get("build_wall_s"),
        "input_build_delta_rss_bytes": inp.get("build_peak_delta_rss_bytes"),
        "n_rows": res.get("n_rows"),
        "recon_max_abs_err": res.get("recon_max_abs_err"),
        "error": dec.get("error"),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--impls", nargs="+", default=IMPL_ORDER)
    ap.add_argument("--sizes", nargs="+", default=DEFAULT_SIZES,
                    help="data sizes as bytes or 1MB/1GB tokens")
    ap.add_argument("--patterns", nargs="+", default=DEFAULT_PATTERNS,
                    choices=["increasing", "random"])
    ap.add_argument("--dtype", default="float64")
    ap.add_argument("--budget", type=float, default=20.0,
                    help="decomposition wall budget (s), counted after the "
                         "input is fully built")
    ap.add_argument("--build-timeout", type=float, default=300.0,
                    help="cap for the input-build phase (s)")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--no-defer", action="store_true",
                    help="execute cells that are normally recorded as deferred")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    env_path = out / "env.json"
    if not env_path.exists():
        env_path.write_text(json.dumps({"bench_memory_env": versions()},
                                       indent=2), encoding="utf-8")

    collected: dict[str, list[dict]] = {impl: [] for impl in args.impls}
    for impl in args.impls:
        rec_path = out / f"{impl}.json"
        if rec_path.exists():
            collected[impl] = json.loads(rec_path.read_text(encoding="utf-8"))

    seen: set[tuple] = set()
    for impl_recs in collected.values():
        for r in impl_recs:
            c = r.get("cell") or {}
            seen.add((c.get("impl"), c.get("size_bytes"), c.get("pattern")))

    cells = [
        {"impl": impl, "size_bytes": _parse_size(sz), "pattern": pat,
         "dtype": args.dtype, "out_dir": str(out)}
        for impl in args.impls
        for sz in args.sizes
        for pat in args.patterns
    ]

    for cfg in cells:
        key = (cfg["impl"], cfg["size_bytes"], cfg["pattern"])
        if args.skip_existing and key in seen:
            print(f"[skip-existing] {cfg['impl']} {cfg['size_bytes']} {cfg['pattern']}")
            continue

        reason = _defer_reason(cfg)
        if reason and not args.no_defer:
            rec = {"cell": cfg,
                   "decompose": {"status": "deferred", "error": reason}}
            print(f"[deferred] {cfg['impl']:7s} {cfg['size_bytes'] / 2**20:6.0f}MB "
                  f"{cfg['pattern']:<10s} {reason}")
        else:
            if reason and args.no_defer:
                print(f"[--no-defer] running deferred cell "
                      f"{cfg['impl']} {cfg['size_bytes']} {cfg['pattern']}")
            rec = _run_cell(cfg, budget_s=args.budget,
                            build_timeout_s=args.build_timeout)
            dec = rec.get("decompose") or {}
            hb = rec.get("hb") or {}
            peak = dec.get("peak_rss_bytes") or (hb.get("peak_rss_bytes"))
            print(
                f"{cfg['impl']:7s} {cfg['size_bytes'] / 2**20:6.0f}MB "
                f"{cfg['pattern']:<10s} {str(dec.get('status')):<9s} "
                f"wall={dec.get('wall_s', 0):7.2f}s "
                f"peak_rss={((peak or 0) / 2**20):9.1f}MB "
                f"rows={str((rec.get('result') or {}).get('n_rows'))}"
            )

        rec["cell"] = cfg
        collected.setdefault(cfg["impl"], []).append(rec)

    for impl, records in collected.items():
        json_path = out / f"{impl}.json"
        json_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
        with open(out / f"{impl}.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=_CSV_FIELDS)
            w.writeheader()
            for r in records:
                w.writerow(_flatten(r))
        print(f"saved {json_path} ({len(records)} records)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
