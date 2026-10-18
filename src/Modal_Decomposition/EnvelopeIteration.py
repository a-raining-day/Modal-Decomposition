"""
该代码是测试性质, 不纳入项目的方法中
"""

import numpy as np

from .Utils import find_peaks

def EnvelopeMeanIteration(S: np.ndarray, iterations: int) -> np.ndarray:
    import scipy.interpolate as si

    X = np.arange(S.size)

    for i in range(iterations):
        x, y, max_pos = find_peaks(S, safe="safe", mod="both")

        min_x = x[~max_pos]
        max_x = x[max_pos]

        min_arr = y[~max_pos]
        max_arr = y[max_pos]

        f_max = si.CubicSpline(max_x, max_arr)
        f_min = si.CubicSpline(min_x, min_arr)

        MIN = f_min(X)
        MAX = f_max(X)

        S = (MAX + MIN) / 2

    return S