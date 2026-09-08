"""
Evaluate the four Envelope.envelope() modes on synthetic AM signals with a
known ground-truth envelope, to pick the best default mode.

Modes: Hilbert (|analytic|), Lowpass (rectify + Butterworth), IQ (quadrature
demodulation at fc), PeakInterpolation (spline through peaks).
Cases: baseline AM, wideband envelope (fm close to fc), noisy AM (20 dB SNR),
deep/slow AM. Metrics: RMSE (edges trimmed), correlation, per-call runtime.

Run:  .venv\\Scripts\\python.exe tests\\comparison\\verify_envelope_modes.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[1]
for _p in (str(_REPO / "src"), str(_REPO / "src" / "Modal_Decomposition" / "Utils")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np

from Modal_Decomposition.Utils.Envelope import envelope

FS = 10000.0
FC = 1000.0
CF = 200.0
TRIM = 40

MODES = ["Hilbert", "Lowpass", "IQ", "PeakInterpolation"]


def make_case(kind: str, rng: np.random.Generator):
    t = np.arange(0, 0.2, 1 / FS)
    if kind == "baseline":
        m, fm = 0.8, 50.0
        noise = 0.0
    elif kind == "wideband":
        m, fm = 0.5, 200.0
        noise = 0.0
    elif kind == "noisy":
        m, fm = 0.8, 50.0
        noise = 0.05
    else:  # deep_slow
        m, fm = 0.2, 10.0
        noise = 0.0
    carrier = np.cos(2 * np.pi * FC * t)
    S = (1 + m * np.cos(2 * np.pi * fm * t)) * carrier
    if noise > 0:
        S = S + noise * rng.standard_normal(S.size)
    true_env = 1 + m * np.cos(2 * np.pi * fm * t)
    return S, true_env


def main() -> int:
    rng = np.random.default_rng(20260906)
    cases = ["baseline", "wideband", "noisy", "deep_slow"]
    results = {c: {} for c in cases}

    for case in cases:
        S, true = make_case(case, rng)
        sl = slice(TRIM, -TRIM)
        for mode in MODES:
            try:
                t0 = time.perf_counter()
                env = envelope(S, method=mode, fs=FS, fc=FC, cf=CF)
                wall = time.perf_counter() - t0
                rmse = float(np.sqrt(np.mean((env[sl] - true[sl]) ** 2)))
                corr = float(np.corrcoef(env[sl], true[sl])[0, 1])
                results[case][mode] = (rmse, corr, wall)
            except Exception as exc:
                results[case][mode] = f"{type(exc).__name__}: {exc}"

    print(f"{'case':<10s} | " + " | ".join(f"{m:<18s}" for m in MODES))
    for case in cases:
        row = []
        for mode in MODES:
            v = results[case][mode]
            if isinstance(v, tuple):
                rmse, corr, wall = v
                row.append(f"RMSE {rmse:8.4f} r {corr:5.3f} {wall*1e3:6.2f}ms")
            else:
                row.append(str(v)[:18])
        print(f"{case:<10s} | " + " | ".join(row))

    # aggregate: mean RMSE over the cases where the mode succeeded
    print("\nmean RMSE / mean corr / mean ms (successful cases):")
    for mode in MODES:
        rmses, corrs, walls = [], [], []
        for case in cases:
            v = results[case][mode]
            if isinstance(v, tuple):
                rmses.append(v[0])
                corrs.append(v[1])
                walls.append(v[2])
        if rmses:
            print(f"  {mode:<18s} rmse={np.mean(rmses):.4f} corr={np.mean(corrs):.3f} "
                  f"{np.mean(walls)*1e3:.2f} ms/call")
        else:
            print(f"  {mode:<18s} FAILED everywhere")
    return 0


if __name__ == "__main__":
    sys.exit(main())
