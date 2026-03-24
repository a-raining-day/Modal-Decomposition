import numpy as np
from numba import njit
from typing import Tuple

class SSA:
    def __init__(self, window_size=None):
        """
        :param:
        window_size: size of windows, default 1/3 of sequence
        """
        self.window_size = window_size
        self.components_ = None
        self.sigma_ = None
        self.U_ = None
        self.V_ = None

    def decompose(self, S, groups=None) -> Tuple[np.ndarray, np.ndarray]:
        """
        :parameter:
        S: Signal
        groups: group information, such as: [[0], [1,2], [3,4]] means which components will be merged. If None, return all

        :return:
        RCs: IMFs Matrix | each row is a IMF
        """
        series = np.asarray(S).flatten()
        N = len(series)

        if self.window_size is None:
            L = N // 3
        else:
            L = min(self.window_size, N // 2)

        K = N - L + 1

        X = np.zeros((L, K))
        for i in range(K):
            X[:, i] = series[i:i + L]

        U, sigma, VT = np.linalg.svd(X, full_matrices=False)

        idx = np.argsort(-sigma)
        sigma = sigma[idx]
        U = U[:, idx]
        VT = VT[idx, :]

        if groups is None:
            groups = [[i] for i in range(len(sigma))]

        RCs = []

        for group in groups:
            x_group = np.zeros((L, K))

            for i in group:
                x_group += sigma[i] * np.outer(U[:, i], VT[i, :])

            rc = np.zeros(N)
            count = np.zeros(N)
            for i in range(L):
                for j in range(K):
                    d = i + j
                    rc[d] += x_group[i, j]
                    count[d] += 1

            rc = rc / count
            RCs.append(rc)

        self.components_ = np.array(RCs)
        self.sigma_ = sigma
        self.U_ = U
        self.V_ = VT

        return self.components_, np.zeros((1, self.components_.shape[1]))


@njit
def SSAfast(series, L):
    N = len(series)
    K = N - L + 1

    X = np.zeros((L, K))
    for i in range(K):
        X[:, i] = series[i:i + L]

    # SVD decomposition
    U, s, VT = np.linalg.svd(X, full_matrices=False)

    return U, s, VT


if __name__ == '__main__':
    ...