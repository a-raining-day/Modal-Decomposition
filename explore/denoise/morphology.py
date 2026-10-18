"""
Morphological 1-D filters used in the high-noise comparison.

``MorphOC`` is the classical opening-then-closing filter, ``MorphCO`` is
closing-then-opening, and ``MorphAvg`` is their average (a symmetric
open-close filter).  All use a flat structuring element and reflect boundary
handling so that the comparison is not dominated by zero-padding edges.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.ndimage import grey_closing, grey_opening

__all__ = [
    "morph_open",
    "morph_close",
    "morph_oc",
    "morph_co",
    "morph_avg",
    "morph_methods",
]


def _as_float(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.float64)


def morph_open(x: np.ndarray, size: int) -> np.ndarray:
    return grey_opening(_as_float(x), size=size, mode="reflect")


def morph_close(x: np.ndarray, size: int) -> np.ndarray:
    return grey_closing(_as_float(x), size=size, mode="reflect")


def morph_oc(x: np.ndarray, size: int) -> np.ndarray:
    """Opening followed by closing (classical open-close filter)."""
    return grey_closing(grey_opening(_as_float(x), size=size, mode="reflect"),
                        size=size, mode="reflect")


def morph_co(x: np.ndarray, size: int) -> np.ndarray:
    """Closing followed by opening."""
    return grey_opening(grey_closing(_as_float(x), size=size, mode="reflect"),
                        size=size, mode="reflect")


def morph_avg(x: np.ndarray, size: int) -> np.ndarray:
    """Average of OC and CO; symmetric and less directional."""
    return 0.5 * (morph_oc(x, size) + morph_co(x, size))


def morph_methods(sizes: tuple[int, ...] = (5, 11, 21)) -> dict[str, Callable]:
    """Return all morphological filters keyed by display name."""
    out: dict[str, Callable] = {}
    for size in sizes:
        out[f"MorphOC{size}"] = lambda x, s=size: morph_oc(x, s)
        out[f"MorphCO{size}"] = lambda x, s=size: morph_co(x, s)
        out[f"MorphAvg{size}"] = lambda x, s=size: morph_avg(x, s)
    return out
