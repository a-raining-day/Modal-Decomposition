"""
EMD representative memory tests for the pluggable matrix pipeline.

Each case (method x size x monotonic pattern x dtype x safe) runs in a fresh
subprocess under RSS sampling; the matrix axes and their defaults mirror the
requested test spec. EMD is the executed representative; every registered
method can be inserted through the same runner.
"""

from __future__ import annotations

import pytest

try:
    import PyEMD  # noqa: F401  (backend of the EMD method)
except ImportError:  # pragma: no cover - exercised only in minimal envs
    pytest.skip("PyEMD is not installed; EMD representative test skipped",
                allow_module_level=True)

from run_matrix import (  # noqa: E402
    DEFAULT_RESULT_DIR,
    run_matrix,
    save_results,
)
from memory_probe import DEFAULT_DTYPES, DEFAULT_SIZES_BYTES, PATTERNS  # noqa: E402

# A fast smoke grid for pytest (the CLI runner executes the full requested
# axis grid: 6 sizes x 2 patterns x 3 dtypes x 3 safe levels).
_SMOKE_SIZES = [256 * 1024]          # 0.25 MB, float64 -> 32k samples
_SMOKE_DTYPES = ["float64"]
_SMOKE_PATTERNS = ["increasing", "random"]
_SMOKE_SAFES = [0, 1, 2]
_BUDGET_S = 30.0


def _run_smoke():
    return run_matrix(
        methods=["EMD"],
        sizes=_SMOKE_SIZES,
        patterns=_SMOKE_PATTERNS,
        dtypes=_SMOKE_DTYPES,
        safes=_SMOKE_SAFES,
        budget_s=_BUDGET_S,
        out_dir=DEFAULT_RESULT_DIR,
        save=False,
    )["EMD"]


def test_emd_smoke_records_per_safe():
    records = _run_smoke()
    cells = 1 * len(_SMOKE_PATTERNS) * len(_SMOKE_DTYPES)
    assert len(records) == cells * len(_SMOKE_SAFES)

    for record in records:
        assert record["method"] == "EMD"
        assert record["dtype"] == "float64"
        assert record["size_bytes"] == _SMOKE_SIZES[0]
        dec = record["decompose"]
        assert dec["status"] == "ok"
        assert dec["wall_s"] >= 0 and dec["delta_rss_bytes"] >= 0
        assert record["input"]["backing"] in ("ndarray", "memmap")

        if record["pattern"] == "random":
            assert record["result"]["n_imfs"] >= 1
        else:
            # a monotone ramp has no interior extrema
            assert record["result"]["n_imfs"] == 0

        scan = record["scan"]
        assert scan["safe"] in (0, 1, 2)
        assert scan["result"] is not None
        if record["pattern"] in ("increasing",):
            assert scan["result"] is True  # residual of a ramp stays monotone
        else:
            assert scan["result"] in (True, False)


def test_emd_results_persisted_and_loadable():
    records = _run_smoke()
    # Persist into a scratch subfolder so the pytest smoke never clobbers the
    # full matrix results produced by the CLI runner.
    scratch = DEFAULT_RESULT_DIR / "_pytest_scratch"
    save_results("EMD", records, scratch)

    json_path = scratch / "EMD.json"
    csv_path = scratch / "EMD.csv"
    assert json_path.is_file() and csv_path.is_file()

    import csv as _csv
    import json as _json

    try:
        with open(json_path, encoding="utf-8") as fh:
            loaded = _json.load(fh)
        assert isinstance(loaded, list) and len(loaded) == len(records)
        with open(csv_path, newline="", encoding="utf-8") as fh:
            rows = list(_csv.DictReader(fh))
        assert len(rows) == len(records)
        header = set(rows[0])
        assert {"dtype", "pattern", "scan.safe", "status",
                "decompose.delta_rss_bytes"} <= header
    finally:
        import shutil
        shutil.rmtree(scratch, ignore_errors=True)


def test_matrix_axis_defaults_match_spec():
    # The requested axis grid is wired in as the runner defaults.
    assert list(PATTERNS) == ["increasing", "random"]
    assert list(DEFAULT_DTYPES) == ["float16", "float32", "float64"]
    assert [b // 2 ** 20 for b in DEFAULT_SIZES_BYTES] == [1, 20, 100, 500, 1024, 3072]
    assert DEFAULT_SIZES_BYTES == (
        1 * 2 ** 20, 20 * 2 ** 20, 100 * 2 ** 20, 500 * 2 ** 20,
        1 * 2 ** 30, 3 * 2 ** 30,
    )


def test_pipeline_insertable_for_any_registered_method():
    from method_workers import available_methods, make_worker

    methods = available_methods()
    assert "EMD" in methods
    for name in methods:
        assert callable(make_worker(name))
