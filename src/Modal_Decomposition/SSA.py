"""
Singular Spectrum Analysis

Decomposes a signal via the singular value decomposition of its trajectory
matrix followed by (strided) diagonal averaging of grouped components.

The trajectory matrix is built with ``numpy.lib.stride_tricks.sliding_window_view``:
``X[i, j] = S[i + stride * j]``, i.e. consecutive columns start ``stride``
samples apart. ``stride = 1`` reproduces the classical Hankel matrix exactly
(the special case); ``stride > 1`` downsamples the embedding, shrinking the
matrix from ``L x (N - L + 1)`` to ``L x ceil((N - L) / stride) + 1``, which
speeds up the SVD at the cost of time resolution. When ``(N - L)`` is not a
multiple of ``stride`` the signal is reflect-padded on the right so that
every sample is still covered and the reconstruction stays exact.

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
    stride: int
    groups: tuple[tuple[int, ...], ...] | None


@register_class("SSA")
class SSA(Decomposer):
    name: ClassVar[str] = "SSA"

    def __init__(
        self,
        window_size: int | None = None,
        groups: Sequence[Sequence[int]] | None = None,
        stride: int = 1,
    ):
        """
        Parameters
        ----------
        window_size : int | None
            Patch length, i.e. the embedding dimension / trajectory-matrix
            row count (``window_size`` *is* the patch length; no separate
            ``patch_len``). None selects N // 3; any value with
            ``1 <= window_size < N`` is allowed (no N // 2 cap).
        groups : sequence of sequences of int, optional
            Groups of elementary component indices to merge. None returns
            each elementary component separately.
        stride : int
            Step between consecutive patches (trajectory-matrix columns):
            ``X[i, j] = S[i + stride * j]``. With the defaults
            (``window_size=None`` -> N // 3, ``stride=1``) consecutive
            patches differ by one sample, which is exactly the classical
            Hankel matrix. Must be a positive integer and
            ``<= window_size``.

        Performance parameter matrix (tests/ssa/test_ssa_stride.py; N=4096
        two-tone + noise, full decomposition; reconstruction is exact
        (~1e-14) for every cell; "-" = invalid (stride > window_size)):

            time (s), window_size x stride:
                ws \\ stride      1       4       16      64
                2048            59.974   3.367   0.240   0.013
                1500            40.591   4.175   0.264   0.016
                1125            27.367   5.912   0.314   0.016
                1024            22.970   3.949   0.265   0.016
                682             11.280   2.600   0.226   0.015
                256             1.718    0.483   0.073   0.012
                64              0.120    0.029   0.012   0.005
                37              0.042    0.013   0.004   -
                32              0.036    0.010   0.002   -
                16              0.008    0.003   0.001   -
                8               0.003    0.001   -       -
                4               0.001    0.000   -       -
                1               0.000    -       -       -

            1st-component correlation with the 37 Hz tone:
                ws \\ stride      1       4       16      64
                2048            0.997    0.998   0.990   0.979
                1500            1.000    0.999   0.994   0.977
                1125            0.999    0.999   0.991   0.982
                1024            0.998    0.997   0.991   0.973
                682             0.998    0.999   0.991   0.982
                256             0.999    0.998   0.992   0.973
                64              0.998    0.997   0.949   0.721
                37              0.998    0.997   0.798   -
                32              0.998    0.997   0.800   -
                16              0.996    0.989   0.732   -
                8               0.998    0.986   -       -
                4               0.946    0.880   -       -
                1               0.856    -       -       -

            SVD time grows steeply with ``window_size`` at ``stride=1``
            (2048: 60 s, 1: 0.1 ms); ``stride >= 16`` is cheap at any
            ``window_size``. Tone tracking is ~1.0 for ``stride <= 4`` with
            ``window_size >= 8``, and LARGE patches keep ``stride=64``
            usable (1125-2048: corr 0.98). Degradation only appears when the
            patch is short AND the stride is large (64 x 64: 0.721,
            16 x 16: 0.732).
        """
        if window_size is not None:
            if not isinstance(window_size, (int, np.integer)) or int(window_size) < 1:
                raise ValueError(
                    f"window_size must be None or a positive integer, "
                    f"got {window_size!r}"
                )
            window_size = int(window_size)
        self.window_size = window_size
        self.groups = groups

        if not isinstance(stride, (int, np.integer)) or int(stride) < 1:
            raise ValueError(f"stride must be a positive integer, got {stride!r}")
        self.stride = int(stride)

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
            L = max(N // 3, 1)
        else:
            L = int(self.window_size)

        # Boundary conditions: 1 <= L < N (the patch length may be any value
        # below the signal length; the legacy N // 2 cap is removed).
        # N == 1 never reaches here: the shared input layer rejects signals
        # that squeeze to 0-d.
        if L < 1 or L >= N:
            raise ValueError(
                f"window_size (L={L}) must satisfy 1 <= L < N (N={N})"
            )

        s = self.stride
        if L < s:
            raise ValueError(
                f"window_size (L={L}) must be >= stride ({s}); "
                f"otherwise some samples are never covered"
            )

        # Edge-pad the right tail (same rule as the patch-SVD reference:
        # pad = stride - (N - L) % stride) so that the strided windows cover
        # every sample; for stride == 1 the padding length is 0 (classic
        # Hankel matrix).
        pad = (-(N - L)) % s
        if pad:
            S_pad = np.pad(S, (0, int(pad)), mode="edge")
        else:
            S_pad = S
        N_pad = N + pad

        x = self.hankel(S_pad, L, s)

        U, sv, Vt = np.linalg.svd(x, full_matrices=False)

        groups = self.groups
        if groups is None:
            groups = [[i] for i in range(len(sv))]

        K = x.shape[1]
        RCs = []
        for group in groups:
            # keep the original multiplication order so stride == 1 stays
            # bitwise identical to the classical implementation
            X_rec = U[:, group] @ (sv[group, None] * Vt[group, :])

            rc = self.diagonal_average_fast(X_rec, L, K, s)[:N]

            RCs.append(rc)

        IMFs: np.ndarray = np.array(RCs)

        self.components_ = IMFs
        self.sigma_ = sv
        self.U_ = U
        self.V_ = Vt

        groups_eff = tuple(tuple(int(i) for i in g) for g in groups)

        return DecompositionResult(
            IMFs,
            None,
            {"n_padded": int(N_pad)},
            SSAConfig(
                window_size=L, stride=s, groups=groups_eff
            ),
        )

    def hankel(self, S, L: int, stride: int = 1):
        """
        Build the patches (trajectory) matrix with numpy's sliding window
        view: patches of ``L`` samples starting every ``stride`` samples,
        i.e. ``X[i, j] = S[i + stride * j]``.

        ``stride = 1`` gives the classical Hankel matrix (consecutive
        patches shifted by one sample).
        """
        S = np.asarray(S)
        W = np.lib.stride_tricks.sliding_window_view(S, L)[:: int(stride)]
        return W.T

    @staticmethod
    def diagonal_average_fast(X_rec, L: int, K: int, stride: int = 1) -> np.ndarray:
        """
        Strided diagonal averaging of a reconstructed trajectory matrix:
        average every ``X[i, j]`` over constant ``i + stride * j``.
        ``stride = 1`` is the classical anti-diagonal (Hankelization).
        """
        n_idx = (np.arange(L)[:, None] + int(stride) * np.arange(K)[None, :]).ravel()
        values = np.asarray(X_rec).ravel()

        Sum = np.bincount(n_idx, weights=values, minlength=L + int(stride) * (K - 1))
        counts = np.bincount(n_idx, minlength=L + int(stride) * (K - 1))
        counts[counts == 0] = 1

        return Sum / counts
