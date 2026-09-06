"""
One measured decomposition case, executed in a fresh subprocess.

The parent (``run_matrix.py``) spawns this script with
``[config.json, report.json]`` and enforces a wall-clock budget by killing the
process. Running each case in a fresh interpreter keeps RSS baselines clean
and makes timeouts / MemoryErrors safe to record.

A ``*.phase1.json`` report is written right after the input stage, so the
parent can still recover the input-pipeline memory facts if this process is
killed during the (possibly unbounded) decomposition.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[1]
for _p in (str(_HERE), str(_REPO), str(_REPO / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np

from memory_probe import PeakMonitor, array_info, monotonic_pattern, dtype_of
from method_workers import make_worker

from Modal_Decomposition.Utils import is_monotonic, set_memmap_ratio

_MEASURE_INTERVAL_S = 0.001


def _report_path(cfg) -> Path:
    return Path(cfg["report_path"])


def _write(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _residual_target(result):
    target = getattr(result, "Res", None)
    if target is None:
        try:
            target = getattr(result, "IMFs")[-1]
        except Exception:
            target = None
    return target


def main() -> int:
    cfg_path, report_path = sys.argv[1], sys.argv[2]
    cfg = json.loads(Path(cfg_path).read_text())
    report = Path(report_path)
    _write(report.with_name(report.stem + ".phase1.json"), {"phase": "starting"})

    set_memmap_ratio(0.6)  # package-default memory policy inside the child

    dt = dtype_of(cfg["dtype"])
    n = int(cfg["size_bytes"]) // dt.itemsize

    # --- phase 1: build the input signal (streamed for large sizes) ---
    build = PeakMonitor(interval_s=_MEASURE_INTERVAL_S)
    S = build.measure(
        lambda: monotonic_pattern(cfg["pattern"], cfg["size_bytes"], cfg["dtype"])
    )
    input_block = {
        "pattern": cfg["pattern"],
        "dtype": cfg["dtype"],
        "n_samples": int(n),
        "size_bytes": int(cfg["size_bytes"]),
        **array_info(S),
        "build_wall_s": round(build.wall_s, 6),
        "build_peak_delta_rss_bytes": int(build.delta),
    }
    _write(report.with_name(report.stem + ".phase1.json"), {
        "phase": "input",
        "input": input_block,
        "rss_after_input_bytes": int(__import__("memory_probe").process_rss()),
    })

    worker = make_worker(cfg["method"])

    # Warm up lazy backends (PyEMD & co.) outside the measured window so the
    # per-case RSS delta reflects the decomposition itself, not module imports.
    try:
        worker(np.linspace(-1.0, 1.0, 256))
    except Exception:
        pass  # best effort; the real run reports genuine failures

    # --- phase 2: decomposition (the parent enforces the time budget) ---
    dec = PeakMonitor(interval_s=_MEASURE_INTERVAL_S)
    result = None
    dec_error = None
    try:
        result = dec.measure(lambda: worker(S))
    except MemoryError as exc:
        dec_error = f"MemoryError: {exc}"
    except Exception as exc:  # missing optional backend, invalid params, ...
        dec_error = f"{type(exc).__name__}: {exc}"

    decompose_block = {
        "status": "ok" if dec_error is None else "error",
        "wall_s": round(dec.wall_s, 6),
        "baseline_rss_bytes": int(dec.baseline),
        "peak_rss_bytes": int(dec.peak),
        "delta_rss_bytes": int(dec.delta),
    }
    if dec_error is not None:
        decompose_block["error"] = dec_error

    # --- phase 3: residual monotonic scans for every requested safe level ---
    scans = []
    target = _residual_target(result) if result is not None else None
    if target is not None and getattr(target, "size", 0) > 1:
        for safe in cfg["safes"]:
            scan = PeakMonitor(interval_s=_MEASURE_INTERVAL_S)
            scan_value = None
            scan_error = None
            try:
                scan_value = scan.measure(
                    lambda: bool(
                        is_monotonic(target, chunk_size=cfg["chunk_size"], safe=safe)
                    )
                )
            except Exception as exc:
                scan_error = f"{type(exc).__name__}: {exc}"
            scans.append({
                "safe": int(safe),
                "chunk_size": int(cfg["chunk_size"]),
                "result": scan_value if scan_error is None else None,
                "wall_s": round(scan.wall_s, 6),
                "delta_rss_bytes": int(scan.delta) if scan_error is None else None,
                "error": scan_error,
            })
    elif result is not None:
        for safe in cfg["safes"]:
            scans.append({
                "safe": int(safe), "chunk_size": int(cfg["chunk_size"]),
                "result": None, "wall_s": 0.0, "delta_rss_bytes": None,
                "error": "no residual / too short to scan",
            })

    report_payload = {
        "method": cfg["method"],
        "input": input_block,
        "decompose": decompose_block,
        "result": {
            "n_imfs": None if result is None else int(result.n_imfs),
            "imfs": array_info(getattr(result, "IMFs", None)),
            "res": array_info(getattr(result, "Res", None)),
        },
        "scans": scans,
    }
    _write(report, report_payload)
    return 0


if __name__ == "__main__":
    sys.exit(main())
