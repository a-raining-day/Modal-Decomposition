"""
Memory matrix runner for decomposition methods.

Axis grid (fully pluggable over every registered method; only EMD is run as
the representative):

    monotonicity  x  data size      x  dtype       x  safe
    {increasing,      {1MB 20MB 100MB  {F16,          {0, 1, 2}
     random}           500MB 1GB 3GB}    F32, F64}

Each case runs inside a fresh subprocess (``case_worker.py``) under RSS
sampling so that timeouts and MemoryErrors of genuinely unbounded cells
(e.g. EMD on multi-GB noise) can be recorded safely together with the
input-pipeline memory facts (input build RSS, backing, per-size/dtype
behaviour). The decomposition runs once per (size, pattern, dtype); the
residual monotonic scan is then measured at every ``safe`` level.

Per-method results are persisted to
``result_for_each_decomposition/<method>.json`` and ``<method>.csv``.

CLI
---
    python tests/test_memory/run_matrix.py --methods EMD                # all axes
    python tests/test_memory/run_matrix.py --methods EMD --sizes 1MB 20MB --dtypes float64
    python tests/test_memory/run_matrix.py --list-methods
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable, Optional

# --- path bootstrap so the script also runs directly (python run_matrix.py) ---
_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[1]
for _p in (str(_HERE), str(_REPO), str(_REPO / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
# ----------------------------------------------------------------------------

from memory_probe import DEFAULT_DTYPES, DEFAULT_SIZES_BYTES, PATTERNS
from method_workers import available_methods, make_worker

__all__ = ["run_matrix", "save_results", "DEFAULT_RESULT_DIR"]

DEFAULT_RESULT_DIR = _HERE / "result_for_each_decomposition"

CASE_WORKER = _HERE / "case_worker.py"
DEFAULT_SAFES = (0, 1, 2)
DEFAULT_CHUNK_SIZE = 1_048_576


def _parse_size(token: str) -> int:
    """Parse ``"1MB"`` / ``"3GB"`` / plain byte count into bytes."""
    token = str(token).strip().lower()
    if token.endswith("mb"):
        return int(float(token[:-2]) * 1024 ** 2)
    if token.endswith("gb"):
        return int(float(token[:-2]) * 1024 ** 3)
    if token.endswith("kb"):
        return int(float(token[:-2]) * 1024)
    return int(token)


def _run_child(cfg: dict, budget_s: float) -> dict:
    """
    Execute one (method, size, pattern, dtype) cell in a fresh subprocess.

    Returns the report dict, or a ``timeout`` / ``crash`` record built from
    the phase-1 report when the child could not finish in time.
    """
    out_dir = Path(cfg["out_dir"])
    runs = out_dir / "_runs"
    runs.mkdir(parents=True, exist_ok=True)

    key = f"{cfg['method']}_{cfg['size_bytes']}_{cfg['pattern']}_{cfg['dtype']}"
    cfg_path = runs / f"{key}.cfg.json"
    report_path = runs / f"{key}.report.json"
    phase_path = runs / f"{key}.report.phase1.json"
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")
    for stale in (report_path, phase_path):
        stale.unlink(missing_ok=True)

    try:
        proc = subprocess.Popen(
            [sys.executable, str(CASE_WORKER), str(cfg_path), str(report_path)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
    except Exception as exc:
        return {"cell": cfg, "decompose": {"status": "error",
                "error": f"spawn failed: {type(exc).__name__}: {exc}"}}

    deadline = time.monotonic() + budget_s
    timed_out = False
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            break
        time.sleep(0.1)
    else:
        timed_out = True
        proc.kill()

    out, err = proc.communicate(timeout=30)

    if report_path.is_file():
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        payload["cell"] = cfg
        payload["_budget_s"] = budget_s
        return payload

    # Child never wrote the final report: recover what we can.
    partial = {}
    if phase_path.is_file():
        partial = json.loads(phase_path.read_text(encoding="utf-8"))
    record = {
        "cell": cfg,
        "input": partial.get("input"),
        "decompose": {
            "status": "timeout" if timed_out else "crash",
            "error": ("killed after budget" if timed_out
                      else f"exit code {proc.returncode}"),
        },
        "_budget_s": budget_s,
    }
    if err.strip():
        record["decompose"]["stderr_tail"] = err.strip()[-800:]
    return record


def _as_safe_records(report: dict) -> list[dict]:
    """
    Expand one cell report into one record per ``safe`` level (the decompose
    measurement is shared; the scan differs).
    """
    cell = report.get("cell", {})
    base = {
        "method": cell.get("method"),
        "n_samples": (report.get("input") or {}).get("n_samples"),
        "size_bytes": cell.get("size_bytes"),
        "dtype": cell.get("dtype"),
        "pattern": cell.get("pattern"),
        "chunk_size": cell.get("chunk_size"),
        "input": report.get("input"),
        "decompose": report.get("decompose"),
        "result": report.get("result"),
        "status": (report.get("decompose") or {}).get("status"),
    }
    scans = report.get("scans") or []
    if scans:
        return [{**base, "scan": scan} for scan in scans]
    # No scan (timeout/crash/no residual): still emit one record per safe level.
    return [{**base, "scan": {"safe": s, "error": "no scan available"}}
            for s in cell.get("safes", DEFAULT_SAFES)]


def run_matrix(
    methods: Iterable[str],
    sizes: Iterable[int] = DEFAULT_SIZES_BYTES,
    patterns: Iterable[str] = PATTERNS,
    dtypes: Iterable[str] = DEFAULT_DTYPES,
    safes: Iterable[int] = DEFAULT_SAFES,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    budget_s: float = 20.0,
    out_dir: Path = DEFAULT_RESULT_DIR,
    save: bool = True,
) -> dict[str, list[dict]]:
    """
    Run the matrix for every method and (by default) persist one result file
    per method. Returns ``{method: [per-safe records]}``.
    """
    out_dir = Path(out_dir)
    if save:
        out_dir.mkdir(parents=True, exist_ok=True)

    collected: dict[str, list[dict]] = {}
    for method in methods:
        make_worker(method)  # fail early on unknown names
        records: list[dict] = []
        for size in sizes:
            for pattern in patterns:
                for dtype in dtypes:
                    cfg = {
                        "method": method,
                        "size_bytes": int(size),
                        "pattern": pattern,
                        "dtype": dtype,
                        "safes": list(safes),
                        "chunk_size": int(chunk_size),
                        "out_dir": str(out_dir),
                    }
                    report = _run_child(cfg, budget_s)
                    records.extend(_as_safe_records(report))
                    status = (report.get("decompose") or {}).get("status")
                    print(
                        f"[{method}] {int(size) / 2 ** 20:7.0f}MB "
                        f"{dtype:<7s} {pattern:<10s} -> {status}"
                    )
        collected[method] = records
        if save:
            save_results(method, records, out_dir)
    return collected


_CSV_FIELDS = [
    "method", "n_samples", "size_bytes", "dtype", "pattern", "chunk_size",
    "scan.safe", "status",
    "input.backing", "input.nbytes", "input.build_wall_s",
    "input.build_peak_delta_rss_bytes",
    "decompose.wall_s", "decompose.delta_rss_bytes", "decompose.peak_rss_bytes",
    "result.n_imfs", "result.res.nbytes",
    "scan.result", "scan.delta_rss_bytes", "scan.wall_s",
    "error",
]


def _flatten(record: dict) -> dict:
    flat: dict = {}
    for key in _CSV_FIELDS:
        if "." in key:
            section, field = key.split(".", 1)
            value = (record.get(section) or {}).get(field, "")
        elif key == "error":
            dec = record.get("decompose") or {}
            scan = record.get("scan") or {}
            value = dec.get("error") or scan.get("error") or ""
        else:
            value = record.get(key, "")
        flat[key] = value
    return flat


def save_results(method: str, records: list[dict], out_dir: Path = DEFAULT_RESULT_DIR) -> None:
    """Persist per-safe records for one method as JSON and CSV."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / f"{method}.json", "w", encoding="utf-8") as fh:
        json.dump(records, fh, indent=2, ensure_ascii=False)

    with open(out_dir / f"{method}.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_CSV_FIELDS)
        writer.writeheader()
        for record in records:
            writer.writerow(_flatten(record))


def _print_summary(collected: dict[str, list[dict]]) -> None:
    for method, records in collected.items():
        print(f"\n=== {method}: {len(records)} records ===")
        seen: dict[tuple, dict] = {}
        for r in records:
            key = (r["size_bytes"], r["dtype"], r["pattern"])
            seen.setdefault(key, r)
        for r in seen.values():
            dec = r.get("decompose") or {}
            inp = r.get("input") or {}
            res = r.get("result") or {}
            err = dec.get("error", "")
            print(
                f"{r['size_bytes'] / 2 ** 20:7.0f}MB {r['dtype']:<7s} "
                f"{r['pattern']:<10s} {dec.get('status', '?'):<8s} "
                f"wall={dec.get('wall_s', 0):7.2f}s "
                f"peak_delta={dec.get('delta_rss_bytes', 0) / 2 ** 20:8.2f}MB "
                f"input={inp.get('backing', '?')} "
                f"n_imfs={res.get('n_imfs', '?')}"
                f"{err and ' | ' + err or ''}"
            )


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", nargs="+", default=["EMD"])
    parser.add_argument("--sizes", nargs="+", default=None,
                        help="e.g. 1MB 20MB 100MB 500MB 1GB 3GB")
    parser.add_argument("--patterns", nargs="+", default=list(PATTERNS),
                        choices=list(PATTERNS))
    parser.add_argument("--dtypes", nargs="+", default=list(DEFAULT_DTYPES),
                        choices=list(DEFAULT_DTYPES))
    parser.add_argument("--safes", nargs="+", type=int, default=list(DEFAULT_SAFES),
                        choices=[0, 1, 2])
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--budget", type=float, default=20.0,
                        help="wall-clock budget (s) per decomposition cell")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--list-methods", action="store_true")
    args = parser.parse_args(argv)

    if args.list_methods:
        print("\n".join(available_methods()))
        return

    sizes = [_parse_size(s) for s in args.sizes] if args.sizes else DEFAULT_SIZES_BYTES
    collected = run_matrix(
        methods=args.methods,
        sizes=sizes,
        patterns=args.patterns,
        dtypes=args.dtypes,
        safes=args.safes,
        chunk_size=args.chunk_size,
        budget_s=args.budget,
        out_dir=args.out_dir,
    )
    _print_summary(collected)


if __name__ == "__main__":
    main()
