import math
import numpy as np

from .Daubechies_polynomial import daubechies_arr

__all__ = ["empirical_wavelet_arr"]

def empirical_wavelet_core_sin(x: int | float | np.ndarray, band_factor: float, w: int | float) -> int | float | np.ndarray:
    return np.sin(math.pi / 2 * daubechies_arr((np.abs(x) - (1 - band_factor) * w) / (2 * band_factor * w)))

def empirical_wavelet_core_cos(x: int | float | np.ndarray, band_factor: float, w: int | float) -> int | float | np.ndarray:
    return np.cos(math.pi / 2 * daubechies_arr((np.abs(x) - (1 - band_factor) * w) / (2 * band_factor * w)))

def empirical_wavelet_arr(x: int | float | np.ndarray, band_factor: float, w0, w1) -> int | float | np.ndarray:
    down = 1 - band_factor
    up = 1 + band_factor

    cand = \
    [
        x < down * w0,
        (down * w0 < x) & (x <= up * w0),
        (up * w0 <= x) & (x <= down * w1),
        (down * w1 < x) & (x <= up * w1),
        up * w1 < x
    ]

    func = \
    [
        0,
        empirical_wavelet_core_sin(x, band_factor, w0),
        1,
        empirical_wavelet_core_cos(x, band_factor, w1),
        0
    ]

    return np.select(cand, func)