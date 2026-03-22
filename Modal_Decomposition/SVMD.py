import numpy as np
from scipy.optimize import minimize
from scipy.signal import hilbert

class SVMD:
    def __init__(self, num_modes=3, alpha=2000, tol=1e-7):
        self.num_modes = num_modes
        self.alpha = alpha
        self.tol = tol
        self.modes = []

    def extract_mode(self, res):
        def cost(omega):
            hilbert_transform = hilbert(res * np.cos(omega * np.arange(len(res))))

            return np.sum(np.abs(hilbert_transform) ** 2) + self.alpha * omega ** 2

        result = minimize(cost, x0=np.random.rand())

        omega = result.x

        mode = res * np.cos(omega * np.arange(len(res)))

        return mode

    def decompose(self, S):
        Res = S
        for _ in range(self.num_modes):
            mode = self.extract_mode(Res)

            self.modes.append(mode)

            Res -= mode

        return np.array(self.modes)