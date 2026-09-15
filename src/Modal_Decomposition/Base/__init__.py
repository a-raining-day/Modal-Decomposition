"""
Base package: contracts, metadata tables, the config base class, and the
import cache.
"""

from .Cache import Cache, cache
from .ClassDefine import Decomposer, DecompositionResult
from .ConfigDefine import Config
from .ConstDefine import (
    SIZE,
    SPLINE_KIND,
    DEFAULT_NUMPY_TYPE,
    CACHE_KEY,
    HILBERT_BACKEND,
    BIG_ARRAY,
    FFT_BACKEND,
    FFT_BACKEND_SMALL,
    FFT_BACKEND_BIG,
    FFT_BACKEND_LIST,
    FFT_BACKEND_ALIAS,
    FFT_PIP_PACKAGE,
    FFT_THREAD_MIN_ELEMS,
    FFT_TILED_MIN_ELEMS,
    MIN_CHUNK_ELEMS,
    ADAPT_MIN_BYTES,
    DEFAULT_FILL_CHUNK_ELEMS,
    VMD_MIN_SAMPLES,
    VMD_UHAT_INFO_LIMIT,
    VMD_PEAK_INIT_LIMIT,
    VMD_CHUNK_WORK_BYTES,
)
from .TextDefine import Name, Reference

__all__ = [
    "Decomposer",
    "DecompositionResult",
    "Config",
    "Name",
    "Reference",
    "Cache",
    "SIZE",
    "SPLINE_KIND",
    "DEFAULT_NUMPY_TYPE",
    "CACHE_KEY",
    "HILBERT_BACKEND",
    "BIG_ARRAY",
    "FFT_BACKEND",
    "FFT_BACKEND_SMALL",
    "FFT_BACKEND_BIG",
    "FFT_BACKEND_LIST",
    "FFT_BACKEND_ALIAS",
    "FFT_PIP_PACKAGE",
    "FFT_THREAD_MIN_ELEMS",
    "FFT_TILED_MIN_ELEMS",
    "VMD_MIN_SAMPLES",
    "VMD_UHAT_INFO_LIMIT",
    "VMD_PEAK_INIT_LIMIT",
    "VMD_CHUNK_WORK_BYTES",
]
