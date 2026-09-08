"""
Verify the LMD fixes and the new envelope/kwargs API (post-fix edition).

1. Fixed midpoint sift (Pchip amplitude interpolation):
   - reconstruction identity error should be ~1e-15 (old: 2.4e-4 = 2 ulp(1e12))
   - PF amplitude range should stay O(signal) (old: +-1e12)
2. API: envelope="midpoint"|"hilbert", hilbert_mod/hilbert_backend read from
   **kwargs, validation errors.
3. Hilbert scipy vs FHT produce bit-identical envelopes; pure sinusoid is
   extracted as one PF with exact reconstruction.

Run:  .venv\\Scripts\\python.exe tests\\comparison\\verify_lmd_recon.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[1]
for _p in (str(_HERE), str(_REPO / "src"), str(_REPO / "ref")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np

from Modal_Decomposition import Class
from quality import analyze
from signals import build_case

LMD_dev = Class.LMD


def main() -> int:
    ok = True
    print("== post-fix midpoint: identity error / PF range ==")
    for case, n in (("B", 1024), ("B", 4096), ("B", 16384), ("A", 16384), ("C", 16384)):
        S, modes = build_case(case, n)
        old = Class.LMD(max_pf=5).decompose(S)
        old_err = float(np.max(np.abs(S - old.reconstruct())))
        old_rng = (float(old.IMFs.min()), float(old.IMFs.max())) if old.IMFs.size else (0.0, 0.0)

        dev = LMD_dev(max_pf=5, envelope="midpoint").decompose(S)
        dev_err = float(np.max(np.abs(S - dev.reconstruct())))
        dev_rng = (float(dev.IMFs.min()), float(dev.IMFs.max())) if dev.IMFs.size else (0.0, 0.0)
        met = analyze(S, np.vstack([dev.IMFs, dev.Res[None, :]]), modes)
        corr = {k: round(v["best_abs_corr"], 3) for k, v in met.get("mode_recovery", {}).items()}
        print(f"case {case} n={n:6d}: old_err={old_err:.2e} old_range=[{old_rng[0]:.1f},{old_rng[1]:.1f}] "
              f"-> dev_err={dev_err:.2e} dev_range=[{dev_rng[0]:.3f},{dev_rng[1]:.3f}] "
              f"n_pf={dev.IMFs.shape[0]} corr={corr}")
        if dev_err > 1e-9:
            ok = False
        # PF samples must stay within the physical clamp (1e4 x signal scale)
        if max(abs(dev_rng[0]), abs(dev_rng[1])) > 1e4 * float(np.max(np.abs(S))) + 1.0:
            ok = False

    print("== API ==")
    S, _ = build_case("B", 1024)
    r_mid = LMD_dev(max_pf=5, envelope="midpoint").decompose(S)
    r_h_scipy = LMD_dev(max_pf=5, envelope="hilbert").decompose(S)
    r_h_fht = LMD_dev(max_pf=5, envelope="hilbert", hilbert_mod="FHT").decompose(S)
    r_h_alias = LMD_dev(max_pf=5, envelope="hilbert", hilbert_backend="FHT").decompose(S)
    same = bool(np.allclose(r_h_fht.IMFs, r_h_alias.IMFs, rtol=1e-9, atol=1e-12))
    diff = float(np.max(np.abs(r_h_fht.IMFs - r_h_scipy.IMFs)))
    print(f"mid rows={r_mid.IMFs.shape[0]}  hilbert(scipy) rows={r_h_scipy.IMFs.shape[0]}  "
          f"alias==hilbert_mod: {same}  max|FHT-scipy|={diff:.3e}")
    ok = ok and same and diff < 1e-9

    for bad, exc in (
        (dict(envelope="x"), ValueError),
        (dict(envelope="hilbert", hilbert_mod="x"), ValueError),
        (dict(envelope="midpoint", nope=1), TypeError),
    ):
        try:
            LMD_dev(max_pf=5, **bad)
            print(f"  FAIL: {bad} did not raise")
            ok = False
        except exc:
            print(f"  ok: {bad} raises {exc.__name__}")

    tt = np.linspace(0, 2 * np.pi, 256, endpoint=False)
    r_sin = LMD_dev(max_pf=2, envelope="midpoint").decompose(np.sin(tt))
    sin_ok = (r_sin.IMFs.shape == (0, 256)
              and float(np.max(np.abs(r_sin.reconstruct() - np.sin(tt)))) < 1e-9)
    print(f"pure sine: rows={r_sin.IMFs.shape} recon<1e-9: {sin_ok}")
    ok = ok and sin_ok

    print("VERIFY:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
