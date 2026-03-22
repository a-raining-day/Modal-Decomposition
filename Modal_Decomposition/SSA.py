import numpy as np
from numba import njit


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

    def decompose(self, S, groups=None):
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
            RC = np.zeros((L, K))

            for i in group:
                Xi = sigma[i] * U[:, i:i + 1] @ VT[i:i + 1, :]

                for j in range(L):
                    for k in range(K):
                        if j + k < L:
                            continue
                        elif j + k < K + L - 1:
                            RC[j, k] += Xi[j + k - L, k] / min(j + 1, L, N - j - K + 1)

            rc = np.zeros(N)
            for n in range(N):
                count = 0
                for j in range(L):
                    for k in range(K):
                        if j + k == n:
                            rc[n] += RC[j, k]
                            count += 1
                if count > 0:
                    rc[n] /= count

            RCs.append(rc)

        self.components_ = np.array(RCs)
        self.sigma_ = sigma
        self.U_ = U
        self.V_ = VT

        return self.components_


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