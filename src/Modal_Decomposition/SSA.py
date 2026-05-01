"""

Python version:  (must)
    3.10.11

Lib and Version:  (if None write None)
    numpy - 2.2.6
	numba - 0.64.0

Only accessed by:  (must)
    Only __init__.py

Description: (if None write None)
    Realize SSA, and optimize the use of numba and SSAfast(function).

Modify:  (must)
    2026.3.25 - Create.
    2026.3.29 - Optimize the SSA.decompose function, it's use_JIT now. Please use SSA() which means SSA.decompose_fast().
"""

import numpy as np
from typing import Union


class SSA:
    """SSA: Singular Spectrum Analysis"""

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

    def __call__(self, S, groups=None, faster=False):
        """SSA: Singular Spectrum Analysis"""
        return self.decompose_fast(S, groups, faster)

    def hankel(self, S, L):
        N = len(S)
        K = N - L + 1

        _S = np.asarray(S)
        strides = (_S.strides[0], _S.strides[0])
        return np.lib.stride_tricks.as_strided(_S, shape=(L, K), strides=strides)

    @staticmethod
    def diagonal_average(X_rec: Union[list, np.ndarray], L: int, K: int) -> np.ndarray:
        """
        :param X_rec: 2-dim
        :param L: int
        :param K: int
        :return: rc (2-dim)
        """
        if not isinstance(X_rec, np.ndarray):
            X_rec = np.array(X_rec)

        N = L + K - 1
        rc = np.zeros(N)
        for n in range(N):
            i_min = max(0, n - K + 1)
            i_max = min(L - 1, n)

            total = 0.0
            count = 0
            for i in range(i_min, i_max + 1):
                j = n - i
                total += X_rec[i, j]
                count += 1
            rc[n] = total / count

        return rc

    @staticmethod
    def diagonal_average_fast(X_rec: Union[list, np.ndarray], L: int, K:int) -> np.ndarray:
        """
        like 'diagonal_average'
        :param X_rec:
        :param L:
        :param K:
        :return: rc (2-dim)
        """
        N = L + K - 1
        n_idx = np.add.outer(np.arange(L), np.arange(K)).flatten()
        values = X_rec.flatten()

        Sum = np.bincount(n_idx, weights=values, minlength=N)

        counts = np.bincount(n_idx, minlength=N)

        counts[counts == 0] = 1

        rc = Sum / counts

        return rc

    def decompose(self, S: Union[list, np.ndarray],  groups=None) -> np.ndarray:
        """
        :param S: Signal (1-dim)
        :param groups: group information, such as: [[0], [1,2], [3,4]] means which components will be merged. If None, return all

        :return: IMFs (2-dim)
        """
        if not isinstance(S, np.ndarray):
            S = np.array(S)

        series = np.asarray(S).flatten()
        N = len(S)

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

        return self.components_

    def decompose_fast(self, S: Union[list, np.ndarray], groups=None, faster: bool=True) -> np.ndarray:
        """
        SSA: Singular Spectrum Analysis

        :param S: Signal (1-dim)
        :param L:
        :param groups: group information, such as: [[0], [1,2], [3,4]] means which components will be merged. If None, return all
        :param faster: if you choose True, function will be use_JIT with wasting memory.
        :return: IMFs (2-dim)
        """

        if not isinstance(S, np.ndarray):
            S = np.array(S)

        N = len(S)

        if self.window_size is None:
            L = N // 3
        else:
            L = min(self.window_size, N // 2)

        K = N - L + 1

        x = self.hankel(S, L)

        U, s, Vt = np.linalg.svd(x, full_matrices=False)

        if groups is None:
            groups = [[i] for i in range(len(s))]

        RCs = []
        for group in groups:
            X_rec = np.zeros((L, K))

            # X_rec = np.sum(s[group] * np.outer(U[:, group], Vt[group, :]), keepdims=True)
            def f(i) -> np.ndarray:
                temp = s[i] * np.outer(U[:, i], Vt[i, :])
                return temp

            if faster:
                X_rec = U[:, group] @ (s[group, None] * Vt[group, :])  # but, you should consider the memory with this code.

            else:
                for i in group:
                    X_rec += f(i)  # also, you can use code above:

            rc = self.diagonal_average_fast(X_rec, L, K)

            RCs.append(rc)

        return np.array(RCs)


def ssa(S: Union[list, np.ndarray], window_size=None, groups=None, faster: bool=False) -> np.ndarray:
    """
    :param S: Signal (1-dim)
    :param window_size:
    :param groups: group information, such as: [[0], [1,2], [3,4]] means which components will be merged. If None, return all
    :param faster:
    :return: IMFs (2-dim)
    """

    Cls = SSA(window_size)

    return Cls(S, groups, faster)