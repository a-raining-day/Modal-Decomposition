"""
Foreground, resumable driver for the full EMD memory matrix.

Runs the (size x pattern x dtype) grid cell by cell through
``run_matrix._run_child`` (each cell = one budgeted subprocess), appends every
completed cell to a JSONL cache, and finally merges the cache into the
per-method result files. Cells already present in the cache are skipped, so
the driver can be interrupted and re-run on the remaining sizes without
repeating finished cells.

Usage (one call per size chunk keeps each foreground call short)::

    python tests/test_memory/driver_emd.py --sizes 1MB 20MB --budget 20
    python tests/test_memory/driver_emd.py --sizes 100MB 500MB --budget 20
    python tests/test_memory/driver_emd.py --sizes 1GB 3GB --budget 20
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# --- path bootstrap ---
_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[1]
for _p in (str(_HERE), str(_REPO), str(_REPO / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
# ----------------------

from memory_probe import DEFAULT_DTYPES, PATTERNS  # noqa: E402
from run_matrix import DEFAULT_RESULT_DIR, _as_safe_records, _run_child, save_results  # noqa: E402


def _parse_size(token: str) -> int:
    token = str(token).strip().lower()
    if token.endswith("gb"):
        return int(float(token[:-2]) * 1024 ** 3)
    if token.endswith("mb"):
        return int(float(token[:-2]) * 1024 ** 2)
    return int(token)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", required=True)
    parser.add_argument("--budget", type=float, default=20.0)
    parser.add_argument("--method", default="EMD")
    parser.add_argument("--dtypes", nargs="+", default=list(DEFAULT_DTYPES),
                        choices=list(DEFAULT_DTYPES))
    parser.add_argument("--force", action="store_true",
                        help="re-run cells already present in the cache")
    parser.add_argument("--cache", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_RESULT_DIR)
    args = parser.parse_args(argv)

    cache_path = args.cache or (args.out_dir / "_runs" / f"{args.method}.cells.jsonl")
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    # Existing cache entries, keyed by cell.
    done: dict[str, list[dict]] = {}
    if cache_path.is_file():
        for line in cache_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                entry = json.loads(line)
                done[entry["key"]] = entry["records"]

    sizes = [_parse_size(s) for s in args.sizes]
    print(f"[driver] method={args.method} sizes={sizes} budget={args.budget}s "
          f"cache={cache_path}", flush=True)

    for size in sizes:
        for pattern in PATTERNS:
            for dtype in args.dtypes:
                key = f"{size}_{pattern}_{dtype}"
                if key in done and not args.force:
                    print(f"[skip ] {key} (cached)", flush=True)
                    continue

                cfg = {
                    "method": args.method,
                    "size_bytes": int(size),
                    "pattern": pattern,
                    "dtype": dtype,
                    "safes": [0, 1, 2],
                    "chunk_size": 1_048_576,
                    "out_dir": str(args.out_dir),
                }
                report = _run_child(cfg, args.budget)
                records = _as_safe_records(report)
                with open(cache_path, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps({"key": key, "records": records}) + "\n")
                done[key] = records

                dec = report.get("decompose") or {}
                inp = report.get("input") or {}
                print(
                    f"[done ] {int(size) / 2 ** 20:6.0f}MB {dtype:<7s} "
                    f"{pattern:<10s} -> {dec.get('status', '?')} "
                    f"wall={dec.get('wall_s', 0):6.2f}s "
                    f"peak_delta={dec.get('delta_rss_bytes', 0) / 2 ** 20:8.2f}MB "
                    f"backing={inp.get('backing', '?')}",
                    flush=True,
                )

    # Merge the whole cache and persist the per-method result files.
    merged: list[dict] = []
    for line in cache_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            merged.extend(json.loads(line)["records"])
    save_results(args.method, merged, args.out_dir)
    print(f"[driver] merged {len(merged)} per-safe records -> "
          f"{args.out_dir / (args.method + '.json')}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
