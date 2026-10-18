"""
该代码是测试性质, 不纳入项目的方法中
"""

import numpy as np

from Modal_Decomposition.Utils import find_peaks

def NoisyEnvelopeMeanIteration(S: np.ndarray, iterations: int, noise_amp: float = 0.2, M: int = 1) -> np.ndarray:
    """
    M: 每轮独立加噪求包络均值的次数（默认 1，保持原行为）。
    """
    import scipy.interpolate as si

    X = np.arange(S.size)

    for i in range(iterations):
        base = S.copy()
        sigma = noise_amp * base.std()
        acc = None

        # M 次独立加噪实现：各自求包络均值，再平均
        for m in range(M):
            Y = base + np.random.normal(loc=0, scale=sigma, size=S.size)

            x, y, max_pos = find_peaks(Y, safe="safe", mod="both")

            min_x = x[~max_pos]
            max_x = x[max_pos]

            min_arr = y[~max_pos]
            max_arr = y[max_pos]

            f_max = si.CubicSpline(max_x, max_arr)
            f_min = si.CubicSpline(min_x, min_arr)

            MIN = f_min(X)
            MAX = f_max(X)

            e = (MAX + MIN) / 2
            acc = e if acc is None else acc + e

        S = acc / M

    return S
