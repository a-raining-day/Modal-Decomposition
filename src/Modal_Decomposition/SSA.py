"""
Singular Spectrum Analysis

Decomposes a signal via the singular value decomposition of its trajectory
matrix followed by diagonal averaging of grouped components.

References
----------
10.1016/j.mex.2020.101015
"""

import numpy as np
from dataclasses import dataclass
from typing import ClassVar, Sequence

from .Base import Config, Decomposer, DecompositionResult
from ._Registry import register_class
from .Utils import Check_Time_and_Signal

__all__ = ["SSA", "SSAConfig"]


@dataclass(frozen=True, kw_only=True)
class SSAConfig(Config):
    """
    Effective parameters of an SSA run.
    """
    window_size: int
    groups: tuple[tuple[int, ...], ...] | None


@register_class("SSA")
class SSA(Decomposer):
    name: ClassVar[str] = "SSA"

    def __init__(
        self,
        window_size: int | None = None,
        groups: Sequence[Sequence[int]] | None = None,
    ):
        """
        Parameters
        ----------
        window_size : int | None
            Window length; None selects N // 3.
        groups : sequence of sequences of int, optional
            Groups of elementary component indices to merge. None returns
            each elementary component separately.
        """
        self.window_size = window_size
        self.groups = groups

        self.components_ = None
        self.sigma_ = None
        self.U_ = None
        self.V_ = None

    def decompose(self, S, T=None) -> DecompositionResult:
        """
        Decompose the signal into grouped components.

        SSA has no residual concept: ``Res`` is None.
        """
        S, T, N = Check_Time_and_Signal(S, T, ndim={1}, method=self.name)

        if self.window_size is None:
            L = N // 3
        else:
            L = min(self.window_size, N // 2)

        K = N - L + 1

        x = self.hankel(S, L)

        U, s, Vt = np.linalg.svd(x, full_matrices=False)

        groups = self.groups
        if groups is None:
            groups = [[i] for i in range(len(s))]

        RCs = []
        for group in groups:
            X_rec = U[:, group] @ (s[group, None] * Vt[group, :])

            rc = self.diagonal_average_fast(X_rec, L, K)

            RCs.append(rc)

        IMFs: np.ndarray = np.array(RCs)

        self.components_ = IMFs
        self.sigma_ = s
        self.U_ = U
        self.V_ = Vt

        groups_eff = tuple(tuple(int(i) for i in g) for g in groups)

        return DecompositionResult(
            IMFs,
            None,
            {},
            SSAConfig(window_size=L, groups=groups_eff),
        )

    def hankel(self, S, L):
        """
        Build the trajectory matrix as a strided view of the signal.
        """
        N = len(S)
        K = N - L + 1

        _S = np.asarray(S)
        strides = (_S.strides[0], _S.strides[0])
        return np.lib.stride_tricks.as_strided(_S, shape=(L, K), strides=strides)

    @staticmethod
    def diagonal_average_fast(X_rec, L: int, K: int) -> np.ndarray:
        """
        Diagonal averaging of a reconstructed trajectory matrix.
        """
        N = L + K - 1
        n_idx = np.add.outer(np.arange(L), np.arange(K)).flatten()
        values = X_rec.flatten()

        Sum = np.bincount(n_idx, weights=values, minlength=N)

        counts = np.bincount(n_idx, minlength=N)

        counts[counts == 0] = 1

        rc = Sum / counts

        return rc
