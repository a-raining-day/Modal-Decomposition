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

* ``_fht``            — pure-NumPy FHT / Hilbert reference implementation with
                        an automatic override by the compiled ``_fht_native``
                        Cython kernel (SAO "am" C code, see setup.py);
* ``_scipy_backend``  — thin ``scipy.signal.hilbert`` wrapper;
* ``_fft`` / ``_fir`` / ``_kramers_kronig`` / ``_wave`` /
  ``_forth_order_wave`` — reserved empty placeholders (0 bytes);
* ``_fht_native``     — the optional compiled accelerator (``.pyx``/``.c``/
                        ``.pyd`` live in this directory).

Import the public API through ``..Hilbert`` instead of reaching into this
package.
"""

from typing import Literal

import numpy as np

from ...Base import HILBERT_BACKEND
from . import (
    _fft,             # noqa: F401  (reserved empty placeholder)
    _fht,
    _fir,             # noqa: F401  (reserved empty placeholder)
    _kramers_kronig,  # noqa: F401  (reserved empty placeholder)
    _scipy_backend,
    _wave,            # noqa: F401  (reserved empty placeholder)
)

__all__ = ["hilbert"]


def hilbert(
    S: np.ndarray,
    mod: Literal[
        "Scipy",
        "Fourth-order Wave",
        "FHT",
        "SB-Hilbert",
        "Kramers-Kronig",
        "FIR",
        "HST",
        "FFT",
        "Optional-order FIR",
    ] = "Scipy",
    **kwargs
) -> np.ndarray:
    """
    Hilbert transform of a real signal with a selectable spline_kind.

    Parameters
    ----------
    S : np.ndarray
        1-D real signal.
    mod : str
        Backend name. ``"Scipy"`` (default) or ``"FHT"`` are implemented;
        the remaining declared names raise ``NotImplementedError``.

    Returns
    -------
    np.ndarray
        Analytic signal (complex128, same length as ``S``). With the default
        ``matlab_phase=False`` convention (see ``_Hilbert._fht``) the
        imaginary part follows the +90° SAO phase convention; the magnitude
        agrees with ``scipy.signal.hilbert`` up to the conjugate phase.

    Raises
    ------
    NotImplementedError
        For a declared spline_kind without an implementation.
    ValueError
        For an unknown spline_kind name.
    """
    match mod:
        case "Scipy":
            return _scipy_backend._hilbert(S, **kwargs)

        case "FHT":
            return _fht._hilbert(S, **kwargs)

        case _:
            raise ValueError(
                f"Unknown Hilbert spline_kind {mod!r}; expected one of {sorted(HILBERT_BACKEND)}"
            )
