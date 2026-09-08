"""
SSA on multi-tone / wideband / non-stationary signals.

Three deterministic cases (N=2048, fs=1000 Hz, full decomposition):

* ``multitone``    - three pure tones (37/113/231 Hz) + tiny noise;
* ``wideband``     - white noise (flat spectrum, no dominant mode);
* ``nonstationary``- abrupt frequency jump at mid-signal (37 Hz -> 90 Hz)
                     with light AM: non-stationary in frequency and
                     amplitude, but piecewise narrowband so SSA separates
                     the two segments into sine/cosine pairs.

Every case asserts exact reconstruction for stride in {1, 4, 16}, plus a
case-specific quality check (tone tracking / energy spread / segment
tracking). The measured numbers are written to ``tests/ssa/results/``
(``ssa_signal_results.json`` and ``ssa_signal_results.md``) so the
behaviour is recorded alongside the tests.
"""

import json
import time
from pathlib import Path

import numpy as np
import pytest

from Modal_Decomposition import Class

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"

N = 2048
FS = 1000.0
STRIDES = (1, 4, 16)


def _multitone():
    t = np.arange(N) / FS
    return (
        np.sin(2 * np.pi * 37.0 * t)
        + 0.8 * np.sin(2 * np.pi * 113.0 * t)
        + 0.5 * np.sin(2 * np.pi * 231.0 * t)
        + 0.02 * np.random.default_rng(0).standard_normal(N)
    ), {
        "tone_37": np.sin(2 * np.pi * 37.0 * t),
        "tone_113": 0.8 * np.sin(2 * np.pi * 113.0 * t),
        "tone_231": 0.5 * np.sin(2 * np.pi * 231.0 * t),
    }


def _wideband():
    return np.random.default_rng(1).standard_normal(N), {}


def _nonstationary():
    t = np.arange(N) / FS
    half = N // 2
    tone1 = np.sin(2 * np.pi * 37.0 * t)
    tone2 = 0.8 * np.sin(2 * np.pi * 90.0 * t)
    s = np.where(t < t[half], tone1, tone2) * (1.0 + 0.2 * np.cos(2 * np.pi * 4.0 * t))
    return s, {
        "seg_37hz": tone1 * (t < t[half]),
        "seg_90hz": tone2 * (t >= t[half]),
    }


CASES = {
    "multitone": (_multitone, "tone tracking via grouped pairs"),
    "wideband": (_wideband, "energy spread (no dominant mode)"),
    "nonstationary": (_nonstationary, "segment tracking via grouped pairs"),
}


def _measure(case: str, stride: int):
    builder, _ = CASES[case]
    s, modes = builder()
    t0 = time.perf_counter()
    r = Class.SSA(stride=stride).decompose(s)
    wall = time.perf_counter() - t0
    recon = float(np.max(np.abs(r.reconstruct() - s)))

    first_corr = None
    if modes:
        name, mode = next(iter(modes.items()))
        first_corr = float(np.corrcoef(r.IMFs[0], mode)[0, 1])

    metrics = {
        "case": case,
        "stride": stride,
        "n": N,
        "window_size": r.config.window_size,
        "n_components": int(r.IMFs.shape[0]),
        "recon_max_abs": recon,
        "first_rc_corr": first_corr,
        "wall_s": round(wall, 6),
    }
    if case == "wideband":
        # energy share of the first component (white noise spreads energy)
        metrics["first_rc_energy_share"] = float(
            np.sum(r.IMFs[0] ** 2) / np.sum(s ** 2)
        )
    return s, r, metrics


@pytest.mark.parametrize("case", list(CASES))
def test_ssa_signal_reconstructs_for_all_strides(case):
    for stride in STRIDES:
        s, r, m = _measure(case, stride)
        assert np.max(np.abs(r.reconstruct() - s)) < 1e-9, (
            f"{case} stride={stride}: recon {m['recon_max_abs']}"
        )


def test_ssa_multitone_grouped_pairs_track_tones():
    s, modes = _multitone()
    # each tone occupies one sine/cosine pair: group [0,1], [2,3], [4,5]
    r = Class.SSA(stride=1, groups=[[0, 1], [2, 3], [4, 5]]).decompose(s)
    names = ["tone_37", "tone_113", "tone_231"]
    assert r.IMFs.shape[0] == 3
    for row, name in zip(r.IMFs, names):
        corr = float(np.corrcoef(row, modes[name])[0, 1])
        assert abs(corr) > 0.95, f"{name}: grouped corr {corr:.3f}"


def test_ssa_wideband_energy_spread():
    s, _, m = _measure("wideband", 1)
    # flat spectrum -> no component may dominate the energy
    assert m["first_rc_energy_share"] < 0.5


def test_ssa_nonstationary_pairs_track_segments():
    s, modes = _nonstationary()
    # each segment occupies one sine/cosine pair: [0,1] -> 37 Hz part,
    # [2,3] -> 90 Hz part
    r = Class.SSA(stride=1, groups=[[0, 1], [2, 3]]).decompose(s)
    assert r.IMFs.shape[0] == 2
    for row, name in zip(r.IMFs, ("seg_37hz", "seg_90hz")):
        corr = float(np.corrcoef(row, modes[name])[0, 1])
        assert corr > 0.9, f"{name}: grouped corr {corr:.3f}"


def test_ssa_signal_results_recorded():
    """Write the measured numbers to tests/ssa/results/ (JSON + Markdown)."""
    RESULTS.mkdir(parents=True, exist_ok=True)

    records = []
    for case in CASES:
        for stride in STRIDES:
            _, _, m = _measure(case, stride)
            records.append(m)

    (RESULTS / "ssa_signal_results.json").write_text(
        json.dumps(records, indent=2), encoding="utf-8"
    )

    lines = [
        "# SSA on multi-tone / wideband / non-stationary signals",
        "",
        f"N={N}, fs={FS} Hz, full decomposition; measured by "
        f"test_ssa_signal.py::test_ssa_signal_results_recorded.",
        "",
        "| case | stride | n_comp | recon max abs | first-RC metric | wall (s) |",
        "|------|-------:|-------:|--------------:|-----------------|---------:|",
    ]
    for m in records:
        metric = (
            f"corr {m['first_rc_corr']:.4f}"
            if m["first_rc_corr"] is not None
            else f"energy {m['first_rc_energy_share']:.4f}"
        )
        lines.append(
            f"| {m['case']} | {m['stride']} | {m['n_components']} | "
            f"{m['recon_max_abs']:.2e} | {metric} | {m['wall_s']:.4f} |"
        )
    lines += [
        "",
        "Notes: reconstruction is exact (~1e-14) for every case/stride; "
        "multitone tones are recovered by grouped sine/cosine pairs "
        "(corr > 0.95); the wideband (white-noise) case spreads energy "
        "across components; the non-stationary case (37 -> 90 Hz jump at "
        "mid-signal) is separated into two pairs tracking each segment "
        "(corr > 0.9).",
        "",
    ]
    (RESULTS / "ssa_signal_results.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    assert (RESULTS / "ssa_signal_results.json").exists()
    assert (RESULTS / "ssa_signal_results.md").exists()
