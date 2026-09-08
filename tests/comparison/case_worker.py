"""
One measured memory/decomposition case, executed in a fresh subprocess.

The parent (``bench_memory.py``) spawns this script with
``[config.json, report.json]``, waits for the phase-1 (input built) report,
then enforces the decomposition budget by killing the process. Running every
cell in a fresh interpreter keeps RSS baselines clean and makes timeouts and
MemoryErrors safe to record.

During the decomposition a heartbeat thread appends ``(elapsed, rss)`` samples
to ``<report>.hb.log`` every 0.02 s, so even a cell killed mid-decomposition
leaves an RSS growth trace behind (the parent records a summary of it).
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[1]
for _p in (str(_HERE), str(_REPO / "src"), str(_REPO / "ref"),
           str(_REPO / "tests" / "test_memory")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np

from memory_probe import PeakMonitor, array_info, dtype_of, monotonic_pattern, process_rss
from quality import analyze
from workers import run_stack

HB_INTERVAL_S = 0.02
METRICS_N_LIMIT = 2_000_000  # skip full quality metrics above this length


class Heartbeat:
    """Append (elapsed, rss) samples to a file while a callable runs."""

    def __init__(self, path: Path, interval_s: float = HB_INTERVAL_S):
        self.path = path
        self.interval_s = interval_s
        self.samples: list[tuple[float, int]] = []
        self.baseline: int = 0
        self.wall_s: float = 0.0
        self._stop = threading.Event()
        self._fh = None

    @property
    def peak(self) -> int:
        return max((rss for _, rss in self.samples), default=self.baseline)

    def measure(self, fn):
        self.baseline = process_rss()
        self._fh = open(self.path, "w", encoding="utf-8")
        self._fh.write(f"# t_s,rss_bytes\n")

        def _sample():
            while not self._stop.is_set():
                t0 = time.monotonic()
                rss = process_rss()
                self.samples.append((t0, rss))
                self._fh.write(f"{t0:.3f},{rss}\n")
                self._fh.flush()
                self._stop.wait(self.interval_s)

        th = threading.Thread(target=_sample, daemon=True)
        start = time.monotonic()
        th.start()
        try:
            result = fn()
        finally:
            self.wall_s = time.monotonic() - start
            self._stop.set()
            th.join()
            if self._fh is not None:
                self._fh.close()
        return result


def _write(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _backing_path(S) -> str | None:
    """Path of the memmap backing file of ``S`` (or None for ndarrays)."""
    filename = getattr(getattr(S, "filename", None), "__str__", None)
    return str(filename()) if filename else None


def _cleanup_input(S) -> None:
    """Best-effort removal of the backing file of a memmap input.

    Must run only after the last read of ``S``: closing the mapping while the
    decomposition still needs it would break it, and on Windows an open
    mapping blocks deletion anyway (parents clean up files left behind by
    killed children).
    """
    path = _backing_path(S)
    if not path:
        return
    try:
        S._mmap.close()  # type: ignore[attr-defined]
    except Exception:
        pass
    try:
        os.remove(path)
    except OSError:
        pass


def main() -> int:
    cfg_path, report_path = sys.argv[1], sys.argv[2]
    cfg = json.loads(Path(cfg_path).read_text(encoding="utf-8"))
    report = Path(report_path)
    hb_path = report.with_name(report.stem + ".hb.log")
    _write(report.with_name(report.stem + ".phase1.json"), {"phase": "starting"})

    dt = dtype_of(cfg["dtype"])
    n = int(cfg["size_bytes"]) // dt.itemsize

    # --- phase 1: build the input signal (streamed to memmap when large) ---
    build = PeakMonitor(interval_s=0.01)
    try:
        S = build.measure(
            lambda: monotonic_pattern(cfg["pattern"], cfg["size_bytes"], cfg["dtype"])
        )
    except Exception as exc:
        _write(report.with_name(report.stem + ".phase1.json"),
               {"phase": "input-error", "error": f"{type(exc).__name__}: {exc}"})
        return 2

    input_block = {
        "pattern": cfg["pattern"],
        "dtype": cfg["dtype"],
        "n_samples": int(n),
        "size_bytes": int(cfg["size_bytes"]),
        **array_info(S),
        "backing_file": _backing_path(S),
        "build_wall_s": round(build.wall_s, 6),
        "build_peak_delta_rss_bytes": int(build.delta),
    }
    _write(report.with_name(report.stem + ".phase1.json"), {
        "phase": "input",
        "input": input_block,
        "rss_after_input_bytes": int(process_rss()),
    })

    # Warm up lazy spline_kind imports outside the measured window.
    try:
        run_stack(cfg["impl"], np.linspace(-1.0, 1.0, 256))
    except Exception:
        pass  # the real run reports genuine failures

    # --- phase 2: decomposition (parent enforces the time budget) ---
    dec = Heartbeat(hb_path, interval_s=HB_INTERVAL_S)
    stack = None
    dec_error = None
    try:
        stack = dec.measure(lambda: run_stack(cfg["impl"], S))
    except MemoryError as exc:
        dec_error = f"MemoryError: {exc}"
    except Exception as exc:
        dec_error = f"{type(exc).__name__}: {exc}"

    decompose_block = {
        "status": "ok" if dec_error is None else "error",
        "wall_s": round(dec.wall_s, 6),
        "baseline_rss_bytes": int(dec.baseline),
        "peak_rss_bytes": int(dec.peak),
        "delta_rss_bytes": int(dec.peak - dec.baseline),
        "hb_n_samples": len(dec.samples),
    }
    if dec_error is not None:
        decompose_block["error"] = dec_error

    result_block = {"n_rows": None}
    if stack is not None:
        rows = np.asarray(stack, dtype=np.float64)
        result_block["n_rows"] = int(rows.shape[0])
        result_block["rows"] = array_info(rows)
        if n <= METRICS_N_LIMIT:
            metrics = analyze(np.asarray(S, dtype=np.float64), rows, {})
            result_block.update({
                "recon_max_abs_err": metrics["recon_max_abs_err"],
                "orthogonality_index": metrics["orthogonality_index"],
            })
        else:
            # whole-array recon/IO scans on hundreds of MB cost seconds; keep
            # them out of the memory record for the largest cells.
            result_block.update({
                "recon_max_abs_err": None,
                "orthogonality_index": None,
                "note": f"metrics skipped above {METRICS_N_LIMIT} samples",
            })
        del rows
    elif dec_error is not None:
        pass

    _cleanup_input(S)
    try:
        del S
    except Exception:
        pass

    _write(report, {
        "impl": cfg["impl"],
        "input": input_block,
        "decompose": decompose_block,
        "result": result_block,
    })
    return 0


if __name__ == "__main__":
    sys.exit(main())
