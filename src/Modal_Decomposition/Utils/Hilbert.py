"""
Hilbert / FHT backends — thin wrapper module.

This module is a pure re-export wrapper over the private implementation
package ``Utils/_Hilbert`` (its ``__init__.py`` holds the ``hilbert``
dispatcher). It contains no logic and no cache code of its own.

Layered access (cache introduced at the *getter* layer only):

    Utils.get_hilbert()          # 惰性 import + 注册进 Base.Cache.cache,
                                 #   返回 cache.get("Modal_Decomposition.Utils.Hilbert")
        └─ Utils.Hilbert (本文件) # 薄封装: 只 re-export
             └─ Utils._Hilbert/__init__.py   # 分发逻辑所在

Public surface
--------------
``hilbert(S, mod="Scipy")``
    Dispatch to a Hilbert-transform backend by name.
    Implemented backends: ``"Scipy"`` (``scipy.signal.hilbert``) and
    ``"FHT"`` (fast Hartley transform based; the compiled ``_fht_native``
    Cython kernel — third-party SAO C code — is used automatically when
    present, otherwise the pure-NumPy reference in ``_Hilbert._fht``).
    Declared-but-unimplemented backends (``"Fourth-order Wave"``,
    ``"SB-Hilbert"``, ``"Kramers-Kronig"``, ``"FIR"``, ``"HST"``, ``"FFT"``,
    ``"Optional-order FIR"``) raise ``NotImplementedError``; unknown names
    raise ``ValueError``.

Also re-exported: ``fht_forward`` / ``fht_inverse`` / ``hilbert_transform``
(pure/accelerated FHT primitives).
"""

from ._Hilbert import hilbert  # noqa: F401  (thin re-export of the dispatcher)
from ._Hilbert._fht import fht_forward as fht_forward  # noqa: F401  (re-export)
from ._Hilbert._fht import fht_inverse as fht_inverse  # noqa: F401  (re-export)
from ._Hilbert._fht import hilbert_transform as hilbert_transform  # noqa: F401  (re-export)

__all__ = [
    "hilbert",
    "fht_forward",
    "fht_inverse",
    "hilbert_transform",
]
