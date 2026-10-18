"""
Hilbert backends — thin wrapper module.

本模块只承载 **Hilbert 变换族**: ``hilbert(S, mod)`` 后端分发与 FHT 原语
(``fht_forward`` / ``fht_inverse`` / ``hilbert_transform``)。

注意: **包络 (envelope) 不在本模块** —— 幅度包络提取见
``Utils.Envelope`` (工具, 含 Hilbert/检波/样条等策略), 模态分解迭代中的
极值样条包络见 ``Utils.Spline``。三者为三个不同概念, 不要混用。

Layered access (cache introduced at the *getter* layer only):

    Utils.Hilbert (本文件)       # 薄封装: 只 re-export
        └─ Utils._Hilbert/__init__.py   # 分发逻辑所在
             ├─ _fft.py          # "FFT" 后端 (默认): rfft/irfft 实半谱
             ├─ _fht.py          # "FHT" 后端: 自适应 (SAO 内核 / 实半谱)
             └─ _scipy_backend.py# "scipy" 后端

Public surface
--------------
``hilbert(S, mod="FFT")``
    按名字分发 Hilbert 后端。已实现三个:

    * ``"FFT"`` (**默认**) — 实半谱法 (``rfft`` + ``irfft``，两次实变换)；
      任意长度精确，约为 ``scipy.signal.hilbert`` 的 **0.59–0.99×**；
    * ``"FHT"`` — 自适应：长度为 2 的幂且 ≤ 512 时走 SAO 编译内核
      (``_fht_native``，该区间最快)，其余走实半谱；**任意长度精确**，
      全尺寸为 scipy 的 **0.48–0.89×**；
    * ``"scipy"`` — 直接 ``scipy.signal.hilbert`` (基线)。

    ``fht_forward`` / ``fht_inverse`` / ``hilbert_transform`` 仍是**原样的
    SAO 语义**（长度非 2 的幂时零填充到 2 的幂），供需要纯 FHT 的调用方使用。
"""

from ._Hilbert import hilbert, HILBERT_MODS, HILBERT_MODS_ARGS  # noqa: F401  (thin re-export of the dispatcher)
from ._Hilbert._fht import fht_forward as fht_forward  # noqa: F401  (re-export)
from ._Hilbert._fht import fht_inverse as fht_inverse  # noqa: F401  (re-export)
from ._Hilbert._fht import hilbert_transform as hilbert_transform  # noqa: F401  (re-export)

__all__ = [
    "hilbert",
    "fht_forward",
    "fht_inverse",
    "hilbert_transform",

    "HILBERT_MODS",
    "HILBERT_MODS_ARGS"
]
