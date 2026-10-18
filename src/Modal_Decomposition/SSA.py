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
--------------------------
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

Selection arguments
-------------------
The two string arguments ``window`` and ``rank_rule`` follow the project-wide
``Literal | Enum`` convention (same shape as ``Utils.Hilbert``): the module
declares ``SSA_WINDOW_MODS_ARGS`` / ``SSA_RANK_MODS_ARGS`` (``Literal``) and
``SSA_WINDOW_MODS`` / ``SSA_RANK_MODS`` (``Enum``) above the Config class, both
the literal spelling and the Enum member are accepted on input, and
``param_check`` normalizes the instance attribute to the Enum member through
``Check.validate_mod``. ``rank_rule=None`` (the classical mode) stays ``None``.
"""

import numpy as np
from dataclasses import dataclass
from enum import Enum
from typing import ClassVar, Literal, Sequence

from .Define import Config, Decomposer, DecompositionResult
from .Utils import Check, Check_Time_and_Signal

__all__ = [
    "SSA",
    "SSAConfig",
    "SSA_WINDOW_MODS",
    "SSA_WINDOW_MODS_ARGS",
    "SSA_RANK_MODS",
    "SSA_RANK_MODS_ARGS",
    "WINDOW_OPTIONS",
    "RANK_RULES",
    "window_vec",
    "svht_omega",
    "energy_rank",
    "svht_rank",
    "svht_clip_rank",
    "f_SSA",
]

SSA_WINDOW_MODS_ARGS = Literal["rect", "hann", "hamming"]
SSA_RANK_MODS_ARGS = Literal["energy", "svht", "svht_clip"]

class SSA_WINDOW_MODS(Enum):
    Rect = "rect"
    Hann = "hann"
    Hamming = "hamming"

class SSA_RANK_MODS(Enum):
    Energy = "energy"
    SVHT = "svht"
    SVHT_Clip = "svht_clip"

# 兼容旧的字符串常量: tests/ssa 与外部脚本仍按 (None, "energy", ...) 迭代
WINDOW_OPTIONS: tuple[str, ...] = tuple(mod.value for mod in SSA_WINDOW_MODS)
RANK_RULES: tuple[str | None, ...] = (None, *(mod.value for mod in SSA_RANK_MODS))


"""
Config
"""
@dataclass(frozen=True, kw_only=True)
class SSAConfig(Config):
    """
    Effective parameters of an SSA run.
    """
    window_size: int | None
    stride: int
    groups: tuple[tuple[int, ...], ...] | None
    window: SSA_WINDOW_MODS
    rank_rule: SSA_RANK_MODS | None
    energy_frac: float
    svht_cap_frac: float
    dtype: type | np.dtype | None


"""
Class and Function of decomposer
"""
class SSA(Decomposer):
    name: ClassVar[str] = "SSA"

    def __init__(
        self,
        window_size: int | None = None,
        groups: Sequence[Sequence[int]] | None = None,
        stride: int = 1,
        window: SSA_WINDOW_MODS_ARGS | SSA_WINDOW_MODS = "rect",
        rank_rule: SSA_RANK_MODS_ARGS | SSA_RANK_MODS | None = None,
        energy_frac: float = 0.995,
        svht_cap_frac: float = 0.95,
        dtype: type | np.dtype | None = None,
        **kwargs,
    ):
        """
        Parameters
        ----------
        window_size : int | None
            Patch (embedding) length ``L``; None auto-selects ``N // 3``.
        groups : Sequence[Sequence[int]] | None
            Singular-triple groups reconstructed jointly; None returns every
            elementary component separately.
        stride : int
            Distance between consecutive patches (``1`` = classical Hankel).
        window : SSA_WINDOW_MODS_ARGS | SSA_WINDOW_MODS
            Analysis window applied to the embedding rows, either the literal
            name (``"rect"`` / ``"hann"`` / ``"hamming"``; ``"none"`` is accepted
            as an alias of ``"rect"``) or an ``SSA_WINDOW_MODS`` member. Stored
            as the Enum member.
        rank_rule : SSA_RANK_MODS_ARGS | SSA_RANK_MODS | None
            Rank selection rule (``"energy"`` / ``"svht"`` / ``"svht_clip"`` or
            the matching Enum member); None keeps every component.
        energy_frac : float
            Cumulative squared-singular energy of ``rank_rule="energy"``.
        svht_cap_frac : float
            Energy cap of ``rank_rule="svht_clip"``.
        dtype : type | np.dtype | None
            Working dtype of the returned IMFs. ``None`` keeps the input
            precision; integer/bool inputs are promoted to float64.
        """
        self.window_size: int | None = window_size
        self.groups: Sequence[Sequence[int]] | None = groups
        self.stride: int = stride
        self.window: SSA_WINDOW_MODS | SSA_WINDOW_MODS_ARGS = window
        self.rank_rule: SSA_RANK_MODS | SSA_RANK_MODS_ARGS | None = rank_rule
        self.energy_frac: float = energy_frac
        self.svht_cap_frac: float = svht_cap_frac
        self.dtype: type | np.dtype | None = dtype

        unknown = set(kwargs) - {"safe", "copy"}
        if unknown:
            raise TypeError(f"unexpected keyword arguments: {unknown}")

        self.safe: str = kwargs.get("safe", "unknown")
        self.copy: bool = kwargs.get("copy", True)

        # 最近一次 decompose 的有效参数 (None = 还没跑过): config_generate 用它们
        # 快照"自动窗口长度"和"被秩规则裁剪后真正保留的分组"
        self._effective_window_size: int | None = None
        self._effective_groups: tuple[tuple[int, ...], ...] | None = None

        self.components_ = None
        self.sigma_ = None
        self.U_ = None
        self.V_ = None

        self.param_check()

    def param_check(self) -> None:
        if self.window_size is not None:
            if not isinstance(self.window_size, (int, np.integer)) \
                    or int(self.window_size) < 1:
                raise ValueError(
                    f"window_size must be None or a positive integer, "
                    f"got {self.window_size!r}"
                )
            self.window_size = int(self.window_size)

        if not isinstance(self.stride, (int, np.integer)) or int(self.stride) < 1:
            raise ValueError(f"stride must be a positive integer, got {self.stride!r}")
        self.stride = int(self.stride)

        # window: 先按历史写法把 "none"/None 规范化成 "rect", 再由 Check.validate_mod
        # 转成 Enum (`_canon_window` 保留大小写不敏感的旧行为); 其余非法值在
        # validate_mod 处抛 ValueError。
        self.window = Check.validate_mod(_canon_window(self.window), SSA_WINDOW_MODS)

        # rank_rule: None 是"经典模式", 保持 None; 非 None 一律转成 Enum
        if self.rank_rule is not None:
            self.rank_rule = Check.validate_mod(self.rank_rule, SSA_RANK_MODS)

        if not isinstance(self.energy_frac, (int, float)) \
                or not (0.0 < self.energy_frac <= 1.0):
            raise ValueError(f"energy_frac must be in (0, 1], got {self.energy_frac!r}")
        if not isinstance(self.svht_cap_frac, (int, float)) or not (
            0.0 < self.svht_cap_frac <= 1.0
        ):
            raise ValueError(
                f"svht_cap_frac must be in (0, 1], got {self.svht_cap_frac!r}"
            )
        self.energy_frac = float(self.energy_frac)
        self.svht_cap_frac = float(self.svht_cap_frac)

        if self.dtype is not None and np.dtype(self.dtype).kind != "f":
            raise ValueError("the `dtype` must be a float numpy dtype or None.")

        if not isinstance(self.copy, bool):
            raise TypeError("the `copy` must be bool")

        # 只做合法性校验, 取值仍按 str 保存, 由 Check_Time_and_Signal 统一解释
        Check.validate_safe(self.safe)

    def config_generate(self) -> SSAConfig:
        """
        Effective parameter snapshot of this run.

        Before the first ``decompose`` the configured values are reported
        (``window_size`` may still be None = auto); afterwards the effective
        ones of the last run, i.e. the resolved patch length and the groups
        actually retained by the rank rule. ``window`` / ``rank_rule`` are the
        Enum members (``rank_rule`` stays None in the classical mode).
        """
        if self._effective_window_size is None:
            window_size = self.window_size
            groups = (
                None
                if self.groups is None
                else tuple(tuple(int(i) for i in g) for g in self.groups)
            )
        else:
            window_size = self._effective_window_size
            groups = self._effective_groups

        return SSAConfig(
            window_size=window_size,
            stride=self.stride,
            groups=groups,
            window=self.window,
            rank_rule=self.rank_rule,
            energy_frac=self.energy_frac,
            svht_cap_frac=self.svht_cap_frac,
            dtype=self.dtype,
        )

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
        # SSA 是纯时域嵌入方法, 不用时间轴; 调用方显式给了 T 才校验它。need_T=False
        # 省掉 arange(N) 这一份与信号等长的临时数组。
        S, N = Check_Time_and_Signal(S, T=T, ndim={1}, method=self.name,
                                     dtype=self.dtype, need_T=False,
                                     safe=self.safe, copy=self.copy)

        # int/bool 的原地运算会被静默截断, f16 精度不足以承载 SVD: 统一升到 float64。
        # f32/f64 保留原精度 (结果 cast 回 S.dtype)。
        if not (S.dtype.kind == "f" and S.dtype.itemsize >= 4):
            S = Check.to_dtype(S, np.dtype(np.float64))

        # 每次调用都从干净状态开始
        self.components_ = None
        self.sigma_ = None
        self.U_ = None
        self.V_ = None
        self._effective_window_size = None
        self._effective_groups = None

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
        if self.window is not SSA_WINDOW_MODS.Rect:
            X = X * win[:, None]

        U, sv, Vt = np.linalg.svd(X, full_matrices=False)

        # --- rank rule -------------------------------------------------- #
        K_cols = X.shape[1]
        match self.rank_rule:
            case None:
                k = len(sv)
            case SSA_RANK_MODS.Energy:
                k = energy_rank(sv, self.energy_frac)
            case SSA_RANK_MODS.SVHT:
                beta = min(X.shape) / max(X.shape)
                k = svht_rank(sv, beta)
            case _:  # SSA_RANK_MODS.SVHT_Clip
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

            if self.window is SSA_WINDOW_MODS.Rect:
                rc = self.diagonal_average_fast(X_rec, L, K_cols, s)[:N]
            else:
                rc = self._ola_windowed(X_rec, win, L, s, K_cols)[:N]
            RCs.append(rc)

        # 输出精度跟随输入 (dtype=None) 或调用方指定的 dtype; 内部计算仍是 float64
        IMFs: np.ndarray = (
            np.asarray(RCs, dtype=S.dtype) if RCs else np.empty((0, N), dtype=S.dtype)
        )

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

        # config_generate() 快照的是**本次有效参数**, 不是构造参数
        self._effective_window_size = L
        self._effective_groups = groups_snapshot

        return DecompositionResult(
            IMFs,
            None,
            {
                "k": k,
                "n_padded": int(N_pad),
                # info 是给人看的运行日志: 这里放字符串而不是 Enum 成员,
                # 保持与旧版 info["window"] / info["rank_rule"] 的字面值一致
                "window": self.window.value,
                "rank_rule": (
                    None if self.rank_rule is None else self.rank_rule.value
                ),
            },
            self.config_generate(),
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

"""
other function for module
"""
def _canon_window(window: SSA_WINDOW_MODS_ARGS | SSA_WINDOW_MODS | None) -> str:
    """
    Resolve a window name to its canonical ``str`` form.

    ``None`` and ``"none"`` both mean ``"rect"`` (the historical alias); Enum
    members pass through as their value. Unknown names are returned unchanged
    and rejected by the caller's ``Check.validate_mod``.
    """
    if isinstance(window, SSA_WINDOW_MODS):
        return window.value
    if window is None:
        return "rect"
    key = str(window).strip().lower()
    if key == "none":
        return "rect"
    return key


def window_vec(L: int, window: SSA_WINDOW_MODS_ARGS | SSA_WINDOW_MODS) -> np.ndarray:
    """
    Return the analysis window of length ``L`` (ones for "rect").

    Accepts both the literal name (``"rect"`` / ``"hann"`` / ``"hamming"``, with
    ``"none"`` as an alias of ``"rect"``) and an ``SSA_WINDOW_MODS`` member, so
    the pre-Enum calling convention keeps working; the argument is validated
    here as well.

    Endpoints: hann/hamming are zero / near-zero at the borders, so the first
    and last samples of a windowed OLA reconstruction have (near-)zero
    synthesis weight; weighted OLA then reproduces the signal exactly on the
    interior samples only.
    """
    mod = Check.validate_mod(_canon_window(window), SSA_WINDOW_MODS)
    match mod:
        case SSA_WINDOW_MODS.Hann:
            return np.hanning(L)
        case SSA_WINDOW_MODS.Hamming:
            return np.hamming(L)
        case _:  # SSA_WINDOW_MODS.Rect
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


"""
other function for module
"""
def f_SSA(S: list | np.ndarray, T=None, *, window_size: int | None = None, groups: Sequence[Sequence[int]] | None = None, stride: int = 1, window: SSA_WINDOW_MODS_ARGS | SSA_WINDOW_MODS = "rect", rank_rule: SSA_RANK_MODS_ARGS | SSA_RANK_MODS | None = None, energy_frac: float = 0.995, svht_cap_frac: float = 0.95, dtype: type | np.dtype | None = None, **kwargs) -> DecompositionResult:
    ssa = SSA(
        window_size=window_size,
        groups=groups,
        stride=stride,
        window=window,
        rank_rule=rank_rule,
        energy_frac=energy_frac,
        svht_cap_frac=svht_cap_frac,
        dtype=dtype,
        **kwargs
    )

    return ssa.decompose(S, T)
