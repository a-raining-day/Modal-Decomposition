import numpy as np

__all__ = ["daubechies", "daubechies_arr"]

def daubechies_core(x: int | float | np.ndarray) -> int | float | np.ndarray:
    return x ** 4 * (35 - 84 * x + 70 * x ** 2 - 20 * x ** 3)

def daubechies(x: int | float) -> int | float:
    if x <= 0: return 0
    if x >= 1: return 1

    return daubechies_core(x)

def daubechies_arr(x: np.ndarray) -> np.ndarray:
    cand = \
    [
        x <= 0,
        (0 < x) & (x < 1),
        1 <= x
    ]

    func = \
    [
        0,
        daubechies_core(x),
        1
    ]

    return np.select(cand, func)