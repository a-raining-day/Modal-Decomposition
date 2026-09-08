"""
Hilbert / FHT backends — thin wrapper module.

本模块只承载 **Hilbert 变换族**: ``hilbert(S, mod)`` 后端分发与 FHT 原语
(``fht_forward`` / ``fht_inverse`` / ``hilbert_transform``)。

注意: **包络 (envelope) 不在本模块** —— 幅度包络提取见
``Utils.Envelope`` (工具, 含 Hilbert/检波/样条等策略), 模态分解迭代中的
极值样条包络见 ``Utils.Spline``。三者为三个不同概念, 不要混用。

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
