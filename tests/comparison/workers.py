"""
Workers for the three EMD implementations under comparison.

Each worker takes one 1-D float64 signal and returns the component stack as
``float64`` rows, with the residue (trend) appended as the last row whenever
the implementation reports one:

+-----------------+------------------------------------------------------+
| key             | engine                                                   |
+-----------------+------------------------------------------------------+
| ``MD-EMD``      | ``Modal_Decomposition.Class.EMD`` — **原生实现**      |
|                 | (原 EMD_new; 2026-09 转正, 取代旧 PyEMD 包装版)。     |
|                 | 现行默认 = CubicSpline + sd_thr=0.01 + faster=False   |
|                 | (质量档: SD 收敛外加 |zc-ext|<=1 窄带门)。           |
|                 | 高速对照 = ``EMD(faster=True)``                       |
| ``PyEMD``       | ``PyEMD.EMD`` direct (the legacy alias shipped with  |
|                 | EMD-signal 1.9.0) — cubic/nbsym=2 默认                |
| ``PySDKit``     | ``pysdkit.EMD`` (ref/pysdkit, v0.5.0; independent    |
|                 | port of the PyEMD sifter)                            |
+-----------------+------------------------------------------------------+

The three are independent engines today (the library EMD is no longer a
PyEMD wrapper). External columns run their shipped defaults (cubic, nbsym=2);
the MD column runs the shipped library default (quality branch, see
``docs/EMD_faster_Branch_Comparison_Report.md``). The module only imports
numpy; backends are imported lazily so that the subprocess memory cells can
keep import cost out of the measured windows.
(Note: the pre-2026-09 benchmark conclusions in REPORT_EMD_MD_vs_PySDKit.md
describe the PyEMD-wrapper era; its raw data is archived under
``results/_legacy_pyemd_wrapper/``.)
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np

IMPL_ORDER = ["MD-EMD", "PyEMD", "PySDKit"]

# ---------------------------------------------------------------------------
# VMD (second representative method, same three-column layout):
#   MD-VMD  = Modal_Decomposition.Class.VMD (facade over the vmdpy engine)
#   vmdpy   = vmdpy.VMD direct (the engine inside MD-VMD; baseline)
#   PySDKit = pysdkit.VMD (independent port of the vmdpy ADMM solver)
# Parity config for the comparison: alpha=2000, tau=0, DC=0, uniform init,
# tol=1e-6 (vmdpy typical / PySDKit default), max_iter=500 (both engines).
# ---------------------------------------------------------------------------
VMD_IMPL_ORDER = ["MD-VMD", "vmdpy", "PySDKit"]

VMD_ENGINE_NOTES = {
    "MD-VMD": "Modal_Decomposition.VMD.decompose -> vmdpy.VMD (ADMM solver)",
    "vmdpy": "vmdpy.VMD direct - the engine shared with MD-VMD",
    "PySDKit": "pysdkit.VMD.fit_transform - independent port of vmdpy "
               "(ref/pysdkit/_vmd/vmd_c.py)",
}

# Number of modes per benchmark case (see signals.py).
CASE_K = {"A": 3, "B": 3, "C": 4}

# ---------------------------------------------------------------------------
# LMD (four-column layout, as requested):
#   LMD-H(scipy)   = LMD_Hilbert(hilbert_mod="Scipy") - old Hilbert-envelope
#                    variant (analytic signal via scipy)
#   LMD-midpoint   = Modal_Decomposition.Class.LMD - the shipped
#                    extrema-midpoint envelope
#   LMD-H(FHT)     = LMD_Hilbert(hilbert_mod="FHT") - Hilbert envelope via the
#                    compiled third-party C (SAO FHT) kernel
#   PySDKit        = pysdkit.LMD - classical Smith moving-average LMD
# Parity: max 5 PFs on all my variants (pysdkit default K=5).
# ---------------------------------------------------------------------------
LMD_IMPL_ORDER = ["LMD-H(scipy)", "LMD-midpoint", "LMD-H(FHT)", "PySDKit"]

# ---------------------------------------------------------------------------
# FMD / EFD (two-column layouts: my implementation vs pysdkit's).
#   FMD: MD-FMD = Class.FMD (bounded config: K=3, max_iter=10, num_hand=3,
#        seed=0; K=-1 auto mode currently hangs on noisy signals - see
#        tests/comparison notes). PySDKit-FMD = pysdkit.FMD (mode_num=3,
#        fs=1000 to match the benchmark signals).
#   EFD: MD-EFD = Class.EFD(max_IMFs=3); PySDKit-EFD = pysdkit.EFD(max_imfs=3).
# Both pysdkit FMD/EFD return modes WITHOUT a residue row; the workers append
# ``res = S - sum(modes)`` so the reconstruction metrics stay comparable.
# ---------------------------------------------------------------------------
FMD_IMPL_ORDER = ["MD-FMD", "PySDKit-FMD"]
EFD_IMPL_ORDER = ["MD-EFD", "PySDKit-EFD"]

FMD_ENGINE_NOTES = {
    "MD-FMD": "Class.FMD(K=3, max_iter=10, num_hand=3, seed=0): adaptive FIR "
              "deconvolution (Miao 2022); K=-1 auto mode hangs on noisy "
              "signals (see tests/comparison notes)",
    "PySDKit-FMD": "pysdkit.FMD(mode_num=3, fs=1000): MATLAB FMD port, "
                   "Hanning FIR bank + IMCKD refinement",
}
EFD_ENGINE_NOTES = {
    "MD-EFD": "Class.EFD(max_IMFs=3): argrelmax spectrum segmentation + "
              "zero-phase filter bank",
    "PySDKit-EFD": "pysdkit.EFD(max_imfs=3): MATLAB EFD.m port (Segm_tec + "
                   "mirror-extended ideal bandpass)",
}

LMD_ENGINE_NOTES = {
    "LMD-H(scipy)": "Class.LMD(envelope='hilbert', hilbert_mod='Scipy') - "
                    "single-shot Hilbert demodulation",
    "LMD-midpoint": "Class.LMD - extrema-midpoint envelope (Pchip amplitude)",
    "LMD-H(FHT)": "Class.LMD(envelope='hilbert', hilbert_mod='FHT') - "
                  "compiled C FHT kernel (SAO)",
    "PySDKit": "pysdkit.LMD: classical Smith moving-average LMD",
}

ENGINE_NOTES = {
    "MD-EMD": "Modal_Decomposition.EMD - native sifting implementation "
              "(ex-EMD_new, since 2026-09; formerly a PyEMD wrapper). "
              "Shipped default: CubicSpline + sd_thr=0.01 + faster=False "
              "(quality branch with the |zc-ext|<=1 narrowband gate)",
    "PyEMD": "PyEMD.EMD.emd direct - cubic/nbsym=2 defaults",
    "PySDKit": "pysdkit.EMD.fit_transform - independent port of the PyEMD "
               "sifter (ref/pysdkit/_emd/emd.py)",
}

# Mapping of the threshold / config names on the PyEMD side to the pysdkit
# side. Both ship the same defaults (verified against source):
#   std_thr=0.2  svar_thr=0.001  total_power_thr=0.005  range_thr=0.001
#   energy_ratio_thr=0.2  MAX_ITERATION=1000  spline_kind='cubic'  nbsym=2
PARAM_PARITY = {
    "spline_kind": "spline_kind",
    "nbsym": "nbsym",
    "std_thr": "std_thr",
    "svar_thr": "svar_thr",
    "total_power_thr": "total_power_thr",
    "range_thr": "range_thr",
    "energy_ratio_thr": "energy_ratio_thr",
    "MAX_ITERATION": "max_iteration",
}


def _load(impl: str):
    """Lazily import the spline_kind module for ``impl`` (cached)."""
    if impl == "MD-EMD":
        import Modal_Decomposition as _MD
        return _MD
    if impl == "PyEMD":
        import PyEMD as _PyEMD
        return _PyEMD
    if impl == "PySDKit":
        import pysdkit as _pysdkit
        return _pysdkit
    raise ValueError(f"unknown implementation {impl!r}")


def run_stack(impl: str, S: np.ndarray, max_imf: int = -1) -> np.ndarray:
    """
    Decompose ``S`` with implementation ``impl`` and return the component
    stack (rows = components, last row = residue when present).

    A fresh instance is created per call: PyEMD/pysdkit cache their last
    result on the instance, and keeping instances would pin large arrays
    between repeats.
    """
    backend = _load(impl)

    if impl == "MD-EMD":
        result = backend.Class.EMD(max_imf=int(max_imf)).decompose(S)
        rows = np.asarray(result.IMFs, dtype=np.float64)
        res = np.asarray(result.Res, dtype=np.float64).reshape(1, -1)
        return np.ascontiguousarray(np.vstack([rows, res]))

    if impl == "PyEMD":
        emd = backend.EMD()
        return np.ascontiguousarray(
            np.asarray(emd.emd(S, max_imf=int(max_imf)), dtype=np.float64)
        )

    # PySDKit
    emd = backend.EMD(max_imfs=int(max_imf))
    return np.ascontiguousarray(
        np.asarray(emd.fit_transform(S), dtype=np.float64)
    )


def run_vmd_stack(impl: str, S: np.ndarray, K: int = 3, tol: float = 1e-6) -> np.ndarray:
    """
    Decompose ``S`` with VMD implementation ``impl`` and return the mode
    stack (rows = K modes, no residual row).

    All three columns use the same ADMM configuration (alpha=2000, tau=0,
    DC=0, uniform frequency init, tol=1e-6, max_iter=500).
    """
    S = np.asarray(S, dtype=np.float64)

    if impl == "MD-VMD":
        import Modal_Decomposition as _MD
        result = _MD.Class.VMD(
            alpha=2000, tau=0.0, K=int(K), DC=0, init=1, tol=tol
        ).decompose(S)
        return np.ascontiguousarray(np.asarray(result.IMFs, dtype=np.float64))

    if impl == "vmdpy":
        from vmdpy import VMD as _vmdpy
        u, _, _ = _vmdpy(S, 2000, 0.0, int(K), 0, 1, tol)
        return np.ascontiguousarray(np.asarray(u, dtype=np.float64))

    # PySDKit
    import pysdkit as _pysdkit
    vmd = _pysdkit.VMD(
        alpha=2000, K=int(K), tau=0.0, init="uniform",
        DC=False, max_iter=500, tol=tol,
    )
    return np.ascontiguousarray(np.asarray(vmd.fit_transform(S), dtype=np.float64))


def run_lmd_stack(impl: str, S: np.ndarray, max_pf: int = 5) -> np.ndarray:
    """
    Decompose ``S`` with LMD implementation ``impl`` and return the component
    stack (rows = product functions, last row = residue).
    """
    S = np.asarray(S, dtype=np.float64)

    if impl in ("LMD-H(scipy)", "LMD-H(FHT)"):
        import Modal_Decomposition as _MD
        mod = "Scipy" if impl == "LMD-H(scipy)" else "FHT"
        result = _MD.Class.LMD(
            max_pf=max_pf, envelope="hilbert", hilbert_mod=mod
        ).decompose(S)
        rows = np.asarray(result.IMFs, dtype=np.float64)
        res = np.asarray(result.Res, dtype=np.float64).reshape(1, -1)
        return np.ascontiguousarray(np.vstack([rows, res]))

    if impl == "LMD-midpoint":
        import Modal_Decomposition as _MD
        result = _MD.Class.LMD(max_pf=max_pf).decompose(S)
        rows = np.asarray(result.IMFs, dtype=np.float64)
        res = np.asarray(result.Res, dtype=np.float64).reshape(1, -1)
        return np.ascontiguousarray(np.vstack([rows, res]))

    # PySDKit (fit_transform already appends the residue as the last row)
    import pysdkit as _pysdkit
    lmd = _pysdkit.LMD(K=int(max_pf))
    return np.ascontiguousarray(
        np.asarray(lmd.fit_transform(S), dtype=np.float64)
    )


def run_fmd_stack(impl: str, S: np.ndarray) -> np.ndarray:
    """Decompose ``S`` with FMD implementation ``impl`` (rows = modes, last
    row = residue; pysdkit returns modes only, so the residue is computed)."""
    S = np.asarray(S, dtype=np.float64)

    if impl == "MD-FMD":
        import Modal_Decomposition as _MD
        result = _MD.Class.FMD(
            K=3, max_iter=10, num_hand=3, seed=0
        ).decompose(S)
        rows = np.asarray(result.IMFs, dtype=np.float64)
        res = np.asarray(result.Res, dtype=np.float64).reshape(1, -1)
        return np.ascontiguousarray(np.vstack([rows, res]))

    # PySDKit-FMD
    import pysdkit as _pysdkit
    fmd = _pysdkit.FMD(fs=1000.0, mode_num=3, cut_num=7, filter_size=30,
                       max_iter_num=20)
    imfs = np.asarray(fmd.fit_transform(S), dtype=np.float64)
    res = S - imfs.sum(axis=0)
    return np.ascontiguousarray(np.vstack([imfs, res.reshape(1, -1)]))


def run_efd_stack(impl: str, S: np.ndarray) -> np.ndarray:
    """Decompose ``S`` with EFD implementation ``impl`` (rows = modes, last
    row = residue; pysdkit returns modes only, so the residue is computed)."""
    S = np.asarray(S, dtype=np.float64)

    if impl == "MD-EFD":
        import Modal_Decomposition as _MD
        result = _MD.Class.EFD(max_IMFs=3).decompose(S)
        rows = np.asarray(result.IMFs, dtype=np.float64)
        res = np.asarray(result.Res, dtype=np.float64).reshape(1, -1)
        return np.ascontiguousarray(np.vstack([rows, res]))

    # PySDKit-EFD
    import pysdkit as _pysdkit
    efd = _pysdkit.EFD(max_imfs=3)
    imfs = np.asarray(efd.fit_transform(S), dtype=np.float64)
    res = S - imfs.sum(axis=0)
    return np.ascontiguousarray(np.vstack([imfs, res.reshape(1, -1)]))


def versions() -> dict:
    """Runtime versions for the results/env.json record."""
    import importlib.metadata as im
    import platform
    import sys

    out = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "emd_signal": _pkg_version("emd-signal", "EMD-signal"),
    }
    for name, attr in (("scipy", "scipy"),):
        try:
            mod = __import__(name)
            out[name] = getattr(mod, "__version__", "?")
        except Exception as exc:  # pragma: no cover
            out[name] = f"<import failed: {exc}>"
    # spline_kind package versions (imports are lazy elsewhere; fine here)
    try:
        out["PyEMD"] = __import__("PyEMD").__version__
    except Exception as exc:
        out["PyEMD"] = f"<import failed: {exc}>"
    try:
        out["pysdkit"] = __import__("pysdkit").__version__
    except Exception as exc:
        out["pysdkit"] = f"<import failed: {exc}>"
    return out


def _pkg_version(dist: str, fallback_dist: str) -> str:
    import importlib.metadata as im
    try:
        return im.version(dist)
    except Exception:
        try:
            return im.version(fallback_dist)
        except Exception:
            return "?"
