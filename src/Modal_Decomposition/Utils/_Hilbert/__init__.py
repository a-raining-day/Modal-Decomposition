"""
Private implementation package of the Hilbert / FHT backends — this module
holds the spline_kind dispatcher itself.

History / layout
----------------
The former public ``Utils/Hilbert/`` package was renamed to ``Utils/_Hilbert/``
so that a public *module* could take over the name (Python forbids a module and
a package with the same name in one directory: the package always wins and the
module file would be silently shadowed). The public surface is:

* ``Modal_Decomposition.Utils.Hilbert`` — thin wrapper module that re-exports
  ``hilbert`` (this file) plus the FHT primitives; it contains no logic and no
  cache code;
* ``Modal_Decomposition.Utils.get_hilbert()`` — the cache-introduction point:
  first access lazily imports ``Utils.Hilbert`` and registers it in the
  process-wide import cache under ``"Modal_Decomposition.Utils.Hilbert"``.

Implementation details in this package:

* ``_fft``            — **default** ``"FFT"`` backend: analytic signal from the
                        real half-spectrum (``rfft`` + ``irfft``); exact for
                        any length, ~0.8x the cost of ``scipy.signal.hilbert``;
* ``_fht``            — ``"FHT"`` backend: pure-NumPy FHT / Hilbert reference
                        implementation with an automatic override by the
                        compiled ``_fht_native`` Cython kernel (SAO "am" C
                        code, see setup.py). **Power-of-two lengths only** —
                        the kernel zero-pads, which changes the analytic
                        signal (see docs/test/Hilbert_Backend_Report.md §5.2);
* ``_scipy_backend``  — thin ``scipy.signal.hilbert`` wrapper;
* ``_fir`` / ``_kramers_kronig`` / ``_wave`` / ``_forth_order_wave`` —
  reserved empty placeholders (0 bytes);
* ``_fht_native``     — the optional compiled accelerator (``.pyx``/``.c``/
                        ``.pyd`` live in this directory).

Measured cost / accuracy of the three backends:
``docs/test/Hilbert_Backend_Report.md`` (raw numbers:
``docs/test/Hilbert_Backend_Results.json``).

Import the public API through ``..Hilbert`` instead of reaching into this
package.
"""

import numpy as np
from enum import Enum
from typing import Literal

from ..Check import Check

from . import (
    _fft,             # "FFT" 后端: rfft/irfft 实半谱法
    _fht,             # "FHT" 后端: SAO 编译内核 + 纯 NumPy 参考实现
    _fir,             # noqa: F401  (reserved empty placeholder)
    _kramers_kronig,  # noqa: F401  (reserved empty placeholder)
    _scipy_backend,   # "scipy" 后端: scipy.signal.hilbert
    _wave,            # noqa: F401  (reserved empty placeholder)
)

__all__ = ["hilbert", "HILBERT_MODS", "HILBERT_MODS_ARGS"]

HILBERT_MODS_ARGS = Literal[
    "scipy",
    "Fourth-order Wave",
    "FHT",
    "SB-Hilbert",
    "Kramers-Kronig",
    "FIR",
    "HST",
    "FFT",
    "Optional-order FIR",
]
class HILBERT_MODS(Enum):
    Scipy = "scipy"
    FW = "Fourth-order Wave"
    FHT = "FHT"
    SBH = "SB-Hilbert"
    KK = "Kramers-Kronig"
    FIR = "FIR"
    HST = "HST"
    FFT = "FFT"
    Oo_FIR = "Optional-order FIR"

def hilbert(
    S: np.ndarray,
    mod: HILBERT_MODS_ARGS | HILBERT_MODS = "FFT",
    **kwargs
) -> np.ndarray[..., np.complexfloating]:
    """
    Hilbert transform of a real signal with a selectable spline_kind.

    Parameters
    ----------
    S : np.ndarray
        1-D real signal.
    mod : str
        Backend name. Implemented: ``"FFT"`` (default; real half-spectrum
        ``rfft``/``irfft``, exact for **any** length and the fastest of the
        three), ``"FHT"`` (compiled SAO C kernel; **power-of-two lengths
        only**, peak memory halved) and ``"scipy"`` (``scipy.signal.hilbert``).
        The remaining declared names raise ``ValueError``.
    **kwargs
        Passed to the selected backend. Recognised: ``matlab_phase``
        (bool, ``"FFT"`` / ``"FHT"`` only; default ``True`` = MATLAB/SciPy
        convention) and ``verbose`` (``"scipy"`` only). Unknown keys are
        ignored by ``"scipy"`` and raise ``TypeError`` for ``"FFT"``/``"FHT"``.

    Returns
    -------
    np.ndarray
        Analytic signal (complex128, same length as ``S``). ``"FFT"`` and
        ``"FHT"`` default to ``matlab_phase=True``, i.e. the same phase
        convention as ``scipy.signal.hilbert``.

    Raises
    ------
    ValueError
        For an unknown spline_kind name, or for a declared-but-unimplemented
        one (the docstring used to promise ``NotImplementedError`` — see
        ``docs/test/Hilbert_Backend_Report.md`` §8).
    """
    mod = Check.validate_mod(mod, HILBERT_MODS)

    match mod:
        case HILBERT_MODS.Scipy:
            return _scipy_backend._hilbert(S, **kwargs)

        case HILBERT_MODS.FHT:
            return _fht._hilbert(S, **kwargs)

        case HILBERT_MODS.FFT:
            return _fft._hilbert(S, **kwargs)

        case _:
            raise ValueError(f"Unknown Hilbert spline_kind {mod!r}; expected one of {HILBERT_MODS._member_names_}")
