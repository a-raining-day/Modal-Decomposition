"""
Singular Spectrum Analysis with rank-rule / window / stride control.

Decomposes a signal via the SVD of its (possibly strided, possibly windowed)
trajectory matrix and reconstructs the retained elementary components.

The embedding generalizes the classical Hankel matrix of SSA: patches of
length ``window_size`` (L) start ``stride`` (tau) samples apart, so

    X[i, j] = w[i] * S[i + stride * j],      i = 0..L-1,  j = 0..N-1,

with ``stride = 1`` reproducing the classical Hankel matrix exactly and
``w`` an optional analysis window. Reconstruction is overlap-add (OLA) with
the window as synthesis weight; for a rectangular window OLA equals the
classical diagonal averaging of SSA (bitwise identical at ``stride = 1``).
``stride > L`` leaves samples uncovered and is rejected (cover condition).

Rank selection (denoising)
---------------------------
Besides the classical "return every elementary component" mode
(``rank_rule=None``, perfect reconstruction), a rank rule selects how many
of the leading singular triples are retained and returned as components:

* ``rank_rule="energy"``    - keep the smallest k whose cumulative squared
  singular energy reaches ``energy_frac`` of the total
  (k_r0 rule; default fraction 0.995).
* ``rank_rule="svht"``      - Gavish-Donoho median singular-value hard
  threshold: keep sigma_i > omega(beta) * median(sigma) with
  omega(beta) = 0.56*beta^3 - 0.95*beta^2 + 1.82*beta + 1.43 and
  beta = min(L, N) / max(L, N).
* ``rank_rule="svht_clip"`` - the protected rule: clip(SVHT, floor=2,
  cap=k at ``svht_cap_frac`` energy), i.e. never keep fewer than 2 nor more
  components than the energy cap admits.

With a rank rule the returned IMFs are the *retained* elementary (or merged)
components and ``sum(IMFs, axis=0)`` is the denoised signal; the discarded
components are dropped (``Res`` stays None). With ``rank_rule=None`` (default)
every component is returned and ``reconstruct()`` reproduces the input
signal to machine precision.

Window functions (rect / hann / hamming) apply to the embedding rows and are
removed by the weighted OLA synthesis.  Experiments of the strided patch-SVD
study these rules stem from (stationary / spectrally separable signals):
energy-99.5 % keeps near-full rank under broadband noise (gain ~0 dB) while
SVHT-clip gains ~10-15 dB at low SNR; the quality-speed law of stride holds
with rank rules (stride ~L/4 keeps 73-88 % of the quality); hann/hamming
mildly help the energy rule and do not reorder rule comparisons.

References
----------
10.1016/j.mex.2020.101015
10.1109/TIT.2014.2312327   (Gavish-Donoho optimal hard threshold)
"""

import numpy as np
from dataclasses import dataclass
from typing import ClassVar, Literal, Sequence

from .Base import Config, Decomposer, DecompositionResult
from ._Registry import register_class
from .Utils import Check_Time_and_Signal

__all__ = ["SSA", "SSAConfig"]

#: analysis windows (canonical names; "none" is accepted as an alias)
WINDOW_OPTIONS: tuple[str, ...] = ("rect", "hann", "hamming")
#: rank-selection rules (None keeps every component)
RANK_RULES: tuple[str | None, ...] = (None, "energy", "svht", "svht_clip")


def _canon_window(window: str) -> str:
    """Resolve a window name to its canonical form."""
    if window is None:
        return "rect"
    key = str(window).strip().lower()
    if key == "none":
        key = "rect"
    if key not in WINDOW_OPTIONS:
        raise ValueError(
            f"window must be one of {WINDOW_OPTIONS} (or 'none'), got {window!r}"
        )
    return key


def window_vec(L: int, window: str) -> np.ndarray:
    """
    Return the analysis window of length ``L`` (ones for "rect").

    Endpoints: hann/hamming are zero / near-zero at the borders, so the first
    and last samples of a windowed OLA reconstruction have (near-)zero
    synthesis weight; weighted OLA then reproduces the signal exactly on the
    interior samples only.
    """
    kind = _canon_window(window)
    if kind == "hann":
        return np.hanning(L)
    if kind == "hamming":
        return np.hamming(L)
    return np.ones(L, dtype=np.float64)


def svht_omega(beta: float) -> float:
    """
    Gavish-Donoho median multiplier for an m x n matrix, beta = min(m,n)/max(m,n).
    """
    return 0.56 * beta ** 3 - 0.95 * beta ** 2 + 1.82 * beta + 1.43


def energy_rank(sv: np.ndarray, fraction: float) -> int:
    """
    Smallest k whose cumulative squared-singular energy reaches ``fraction``.

    Returns ``len(sv)`` for a zero-energy (or degenerate) spectrum.
    """
    total = float(np.sum(sv ** 2))
    if total <= 0.0:
        return len(sv)
    k = int(np.argmax(np.cumsum(sv ** 2) >= fraction * total)) + 1
    return min(k, len(sv))


def svht_rank(sv: np.ndarray, beta: float) -> int:
    """Gavish-Donoho median hard-threshold rank (no floor/cap)."""
    if sv.size == 0:
        return 0
    tau = svht_omega(beta) * float(np.median(sv))
    return int(np.sum(sv > tau))


def svht_clip_rank(sv: np.ndarray, beta: float,
                   floor: int = 2, cap_frac: float = 0.95) -> int:
    """
    Protected SVHT rule: clip(SVHT, floor, energy-rank at cap_frac).
    """
    k_sv = svht_rank(sv, beta)
    k_cap = energy_rank(sv, cap_frac)
    return int(min(max(k_sv, floor), k_cap))


@dataclass(frozen=True, kw_only=True)
class SSAConfig(Config):
    """
    Effective parameters of an SSA run.
    """
    window_size: int
    stride: int
    groups: tuple[tuple[int, ...], ...] | None
    window: str
    rank_rule: str | None
    energy_frac: float
    svht_cap_frac: float


@register_class("SSA")
class SSA(Decomposer):
    name: ClassVar[str] = "SSA"

    def __init__(
        self,
        window_size: int | None = None,
        groups: Sequence[Sequence[int]] | None = None,
        stride: int = 1,
        window: Literal["rect", "hann", "hamming", "none"] = "rect",
        rank_rule: Literal["energy", "svht", "svht_clip"] | None = None,
        energy_frac: float = 0.995,
        svht_cap_frac: float = 0.95,
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
            each elementary component separately. With a rank rule, group
            members at or beyond the retained rank are dropped (groups left
            empty are omitted).
        stride : int
            Step between consecutive patches (trajectory-matrix columns):
            ``X[i, j] = S[i + stride * j]``. With the defaults
            (``window_size=None`` -> N // 3, ``stride=1``) consecutive
            patches differ by one sample, which is exactly the classical
            Hankel matrix. Must be a positive integer and
            ``<= window_size`` (cover condition: larger strides leave
            samples unreconstructable).
        window : {"rect", "hann", "hamming"}
            Analysis window applied to the embedding rows before the SVD and
            removed by the weighted OLA synthesis ("none" is an alias of
            "rect"). For a rectangular window the reconstruction is the
            classical diagonal averaging (exact at stride=1).
        rank_rule : {"energy", "svht", "svht_clip"} | None
            Rank-selection rule of the retained singular triples; None
            (default) returns every elementary component (classical SSA
            decomposition, exact reconstruction).
        energy_frac : float
            Cumulative squared-energy fraction of the ``"energy"`` rule
            (0.995 = the classical 99.5 % rule).
        svht_cap_frac : float
            Energy cap of the ``"svht_clip"`` rule (rank of the energy rule
            at this fraction; 0.95 by default).

        Notes
        -----
        The returned ``DecompositionResult`` has ``Res = None``. With
        ``rank_rule=None`` the sum of all components reproduces the input;
        with a rank rule ``sum(IMFs, axis=0)`` is the denoised signal and the
        dropped components are the removed (noise) part, obtainable as
        ``S - r.reconstruct()``.
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

        self.window = _canon_window(window)

        if rank_rule not in RANK_RULES:
            raise ValueError(
                f"rank_rule must be one of {RANK_RULES}, got {rank_rule!r}"
            )
        self.rank_rule = rank_rule

        if not isinstance(energy_frac, (int, float)) or not (0.0 < energy_frac <= 1.0):
            raise ValueError(f"energy_frac must be in (0, 1], got {energy_frac!r}")
        if not isinstance(svht_cap_frac, (int, float)) or not (
            0.0 < svht_cap_frac <= 1.0
        ):
            raise ValueError(
                f"svht_cap_frac must be in (0, 1], got {svht_cap_frac!r}"
            )
        self.energy_frac = float(energy_frac)
        self.svht_cap_frac = float(svht_cap_frac)

        self.components_ = None
        self.sigma_ = None
        self.U_ = None
        self.V_ = None

    # ------------------------------------------------------------------ #
    # decompose
    # ------------------------------------------------------------------ #
    def decompose(self, S, T=None) -> DecompositionResult:
        """
        Decompose the signal into (retained) reconstruction components.

        Returns
        -------
        DecompositionResult
            ``IMFs`` of shape (K, N) where K is the number of retained
            components (rank-rule selected, or all when ``rank_rule=None``);
            ``Res`` is None. ``config`` snapshots the effective parameters.
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

        win = window_vec(L, self.window)
        x = self.hankel(S_pad, L, s)
        X = np.asarray(x, dtype=np.float64)
        if self.window != "rect":
            X = X * win[:, None]

        U, sv, Vt = np.linalg.svd(X, full_matrices=False)

        # --- rank rule -------------------------------------------------- #
        K_cols = X.shape[1]
        if self.rank_rule is None:
            k = len(sv)
        elif self.rank_rule == "energy":
            k = energy_rank(sv, self.energy_frac)
        elif self.rank_rule == "svht":
            beta = min(X.shape) / max(X.shape)
            k = svht_rank(sv, beta)
        else:  # "svht_clip"
            beta = min(X.shape) / max(X.shape)
            k = svht_clip_rank(sv, beta, floor=2, cap_frac=self.svht_cap_frac)
        k = int(min(max(k, 0), len(sv)))

        # --- groups (intersect with the retained rank) ------------------- #
        if self.groups is None:
            groups_eff: list[tuple[int, ...]] = [(i,) for i in range(k)]
        else:
            groups_eff = []
            for g in self.groups:
                members = tuple(int(i) for i in g)
                if not members:
                    continue
                if any(i < 0 or i >= len(sv) for i in members):
                    raise ValueError(
                        f"group {members} contains indices outside "
                        f"[0, {len(sv)}), the number of singular values"
                    )
                kept = tuple(i for i in members if i < k)
                if kept:
                    groups_eff.append(kept)

        # --- per-group reconstruction (weighted OLA / diagonal averaging) - #
        RCs = []
        for group in groups_eff:
            # keep the original multiplication order so stride == 1 stays
            # bitwise identical to the classical implementation
            X_rec = U[:, group] @ (sv[group, None] * Vt[group, :])

            if self.window == "rect":
                rc = self.diagonal_average_fast(X_rec, L, K_cols, s)[:N]
            else:
                rc = self._ola_windowed(X_rec, win, L, s, K_cols)[:N]
            RCs.append(rc)

        IMFs: np.ndarray = np.array(RCs)
        if IMFs.shape[0] == 0:
            IMFs = np.empty((0, N), dtype=np.float64)

        self.components_ = IMFs
        self.sigma_ = sv
        self.U_ = U
        self.V_ = Vt

        groups_snapshot: tuple[tuple[int, ...], ...] | None
        if self.groups is None and self.rank_rule is None:
            groups_snapshot = None  # pure elementary decomposition
        else:
            groups_snapshot = (
                tuple(tuple(int(i) for i in g) for g in groups_eff)
                if groups_eff
                else None
            )

        return DecompositionResult(
            IMFs,
            None,
            {
                "k": k,
                "n_padded": int(N_pad),
                "window": self.window,
                "rank_rule": self.rank_rule,
            },
            SSAConfig(
                window_size=L,
                stride=s,
                groups=groups_snapshot,
                window=self.window,
                rank_rule=self.rank_rule,
                energy_frac=self.energy_frac,
                svht_cap_frac=self.svht_cap_frac,
            ),
        )

    # ------------------------------------------------------------------ #
    # embedding / reconstruction primitives
    # ------------------------------------------------------------------ #
    def hankel(self, S, L: int, stride: int = 1):
        """
        Build the patches (trajectory) matrix with numpy's sliding window
        view: patches of ``L`` samples starting every ``stride`` samples,
        i.e. ``X[i, j] = S[i + stride * j]``.

        ``stride = 1`` gives the classical Hankel matrix (consecutive
        patches shifted by one sample). The analysis window is *not*
        applied here (see ``decompose``).
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

    @staticmethod
    def _ola_windowed(
        X_rec: np.ndarray, win: np.ndarray, L: int, stride: int, K: int
    ) -> np.ndarray:
        """
        Weighted overlap-add of a reconstructed patch matrix: every column
        ``j`` contributes ``win``-weighted samples to ``[j*stride, j*stride+L)``
        and the accumulated sum is divided by the accumulated window weight.
        Samples with zero accumulated weight (possible only at the very
        borders of zero-endpoint windows) yield 0.

        For a rectangular window this coincides with
        :meth:`diagonal_average_fast`; ``stride <= L`` guarantees every
        sample is covered.
        """
        padded_len = L + int(stride) * (K - 1)
        n_idx = (np.arange(L)[:, None] + int(stride) * np.arange(K)[None, :]).ravel()
        # flat (C-order) index of pair (i, j) is i*K + j, so the window
        # weight of element (i, j) is win[i]: repeat, not tile, matches.
        win_repeat = np.repeat(win, K)

        Sum = np.bincount(n_idx, weights=np.asarray(X_rec).ravel(),
                          minlength=padded_len)
        Wsum = np.bincount(n_idx, weights=win_repeat, minlength=padded_len)
        out = np.zeros(padded_len, dtype=np.float64)
        np.divide(Sum, Wsum, out=out, where=Wsum > 0)
        return out
