"""
Python version:  (must)
    3.10.11

Lib and Version:  (if None write None)
    numpy - 2.2.6
	scipy - 1.15.3

Only accessed by:  (must)
    Only __init__.py

Modify:  (must)
    2026.3.25

Description: (if None write None)
    Realize the SVMD. Optimize the use of scipy lib.
"""

import numpy as np
from typing import Tuple

class SVMD:
    def __init__(self, num_modes=3, alpha=2000, tol=1e-7):
        self.num_modes = num_modes
        self.alpha = alpha
        self.tol = tol
        self.modes = []

    def extract_mode(self, res):
        from scipy.optimize import minimize
        from scipy.signal import hilbert

        def cost(omega):
            hilbert_transform = hilbert(res * np.cos(omega * np.arange(len(res))))

            return np.sum(np.abs(hilbert_transform) ** 2) + self.alpha * omega ** 2

        result = minimize(cost, x0=np.random.rand())

        omega = result.x

        mode = res * np.cos(omega * np.arange(len(res)))

        return mode

    def decompose(self, S) -> Tuple[np.ndarray, np.ndarray]:
        """

        :param S: Signal (1-dim)
        :return: IMFs(2-dim), Res(1-dim)
        """
        Res = S
        for _ in range(self.num_modes):
            mode = self.extract_mode(Res)

            self.modes.append(mode)

            Res -= mode

        return np.array(self.modes), Res