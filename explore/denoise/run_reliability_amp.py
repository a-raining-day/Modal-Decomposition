"""
CWRU reliability vs auxiliary-noise amplitude.

The earlier reliability test used noise_amp=0.5/0.6, where the spurious
221.5 Hz peak dominated.  This sweep checks whether smaller auxiliary noise
(0.01/0.05/0.1/0.2/0.3) makes the true BPFI peak correct/stable.

Usage:
    python -m explore.denoise.run_reliability_amp
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.io as sio

ROOT = Path(__file__).resolve().parents[2]
for path in (str(ROOT), str(ROOT / "src")):
    if path not in sys.path:
        sys.path.insert(0, path)

from explore.denoise.methods import run_proposed  # noqa: E402
from explore.denoise.run_reliability import (  # noqa: E402
    BPFI,
    CWRU,
    CWRU_FS,
    CWRU_SEGMENTS,
    CWRU_SEG_LEN,
    SPUR,
    analyze_specs,
    env_spectrum,
)

AMPS = (0.01, 0.05, 0.1, 0.2, 0.3, 0.5)
R = 20
K = 2
M = 64


def run() -> list[dict]:
    data = sio.loadmat(CWRU)
    x = np.asarray(data["X105_DE_time"], dtype=np.float64).ravel()
    records = []
    for amp in AMPS:
        for seg_i, start in enumerate(CWRU_SEGMENTS):
            seg = x[start : start + CWRU_SEG_LEN]
            specs = []
            t0 = time.perf_counter()
            for r in range(R):
                y = run_proposed(
                    seg,
                    {"K": K, "M": M, "noise_amp": amp},
                    seed=9000 + seg_i * 100 + r,
                )
                f, spec = env_spectrum(y, CWRU_FS, (2000.0, 5000.0), (80.0, 250.0))
                specs.append(spec)
            row = analyze_specs(f, specs, BPFI, 2.0, SPUR, 2.0, f"seg{seg_i}")
            if row:
                row.update(
                    {
                        "amp": amp,
                        "segment": seg_i,
                        "runtime_s": time.perf_counter() - t0,
                    }
                )
                records.append(row)
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--outdir",
        type=Path,
        default=ROOT / "explore/denoise/results",
    )
    args = parser.parse_args()
    records = run()
    with (args.outdir / "reliability_amp.json").open("w", encoding="utf-8") as fh:
        json.dump(records, fh, indent=2, ensure_ascii=False)
    print(f"wrote {len(records)} records -> {args.outdir / 'reliability_amp.json'}")


if __name__ == "__main__":
    main()
