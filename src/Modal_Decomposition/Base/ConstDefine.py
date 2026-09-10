"""
Size constants shared by the whole package.

所有涉及分块 / 内存策略的大小常量统一从本字典取用
(见 ``Utils.Chunk`` / ``Utils.Check`` / ``Utils.Memory`` /
``Utils.Monotonicity``), 避免各模块各自书写魔数。
"""

__all__ = ["SIZE", "SPLINE_KIND", "DEFAULT_NUMPY_TYPE", "CACHE_KEY", "HILBERT_BACKEND"]

import numpy as np


SIZE = \
{
    "1KB": 1024,            # 2**10 bytes
    "1MB": 1024 ** 2,       # 2**20 bytes
    "1GB": 1024 ** 3,       # 2**30 bytes
}

SPLINE_KIND = \
[
    "UnivariateSpline",
    "CubicSpline",
    "PCHIP",
    "Akima",
]

DEFAULT_NUMPY_TYPE = np.float64

CACHE_KEY = \
{
    "scipy": \
        {
            "signal": "scipy.signal",
            "interpolate": "scipy.interpolate",
        }
}

HILBERT_BACKEND = \
[
    "Scipy",
    "Fourth-order Wave",
    "FHT",
    "SB-Hilbert",
    "Kramers-Kronig",
    "FIR",
    "HST",
    "FFT",
    "Optional-order FIR",
]