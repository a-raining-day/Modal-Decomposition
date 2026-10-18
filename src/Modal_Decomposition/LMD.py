"""
Local Mean Decomposition

Extracts product functions by iteratively smoothing the local mean and
envelope estimate from extrema interpolation.

包络概念区分 (Hilbert vs Spline)
--------------------------------
- 模态分解筛分迭代里的"包络 / 局部均值"是对**极值**做插值得到 —— 本方法默认
  ``envelope="midpoint"`` (Smith 经典筛分): 幅度包络用保形 Pchip 插值,
  局部均值用自然三次样条 (均经 ``scipy.interpolate`` 提供; 极值端点镜像由
  本模块顶层的 ``mirror_extrema`` 完成)。自然三次样条曾因极值间过冲导致幅度
  爆炸 (峰值 ~1e12, 重构损失 ~2.4e-4), 故幅度一律走 Pchip; 最后一次筛分的
  局部均值按经典 LMD 语义留在残差中。
- 解析 Hilbert 包络 ``a(t) = |H[h(t)]|`` **不是**筛分迭代中的包络: 只有确实
  需要解析幅度时才使用 (``envelope="hilbert"`` 显式选择, 单次解调 / 实验
  对比)。迭代该解调在多分量信号上不收敛 (包络拍频、极值数发散), 故该变体
  是搁置的历史选项。Hilbert 后端经 ``hilbert_mod`` 传入, 取 ``Utils.Hilbert``
  的 ``HILBERT_MODS`` 枚举: ``hilbert_mod="Scipy"`` (默认, scipy.signal.hilbert;
  历史写法, 大小写归一后落到 ``HILBERT_MODS.Scipy``, canonical 值 ``"scipy"``)
  或 ``hilbert_mod="FHT"`` (第三方 C 内核, 见 ``Utils.Hilbert``)。实例上存的
  是 ``HILBERT_MODS`` 成员 (Enum), 不再保存裸字符串。

References
----------
10.1098/rsif.2005.0058
"""

import numpy as np
from dataclasses import dataclass
from enum import Enum
from typing import ClassVar, Literal

from .Define import Config, Decomposer, DecompositionResult, DEFAULT_NUMPY_TYPE
from .Base import Cache
from .Utils import (
    Check,
    Check_Time_and_Signal,
    hilbert,
    HILBERT_MODS,
    HILBERT_MODS_ARGS,
    monotonic,
)

__all__ = ["LMD", "LMDConfig", "LMD_ENVELOPE_MODS", "LMD_ENVELOPE_MODS_ARGS", "f_LMD"]

LMD_ENVELOPE_MODS_ARGS = Literal["midpoint", "hilbert"]
class LMD_ENVELOPE_MODS(Enum):
    Midpoint = "midpoint"
    Hilbert = "hilbert"

"""
Config
"""
@dataclass(frozen=True, kw_only=True)
class LMDConfig(Config):
    """
    Effective parameters of an LMD run.
    """
    max_pf: int | None
    max_iter: int
    eps: float
    eps_stable: float
    min_amp: float
    max_amp: float
    converge_mean: float
    smooth_window: int
    envelope: LMD_ENVELOPE_MODS | LMD_ENVELOPE_MODS_ARGS
    hilbert_mod: HILBERT_MODS | HILBERT_MODS_ARGS
    dtype: type | np.dtype | None


"""
Class and Function of decomposer
"""
class LMD(Decomposer):
    name: ClassVar[str] = "LMD"

    def __init__(
        self,
        max_pf: int | None = None,
        max_iter: int = 37,
        eps: float = 0.05,
        eps_stable: float = 1e-12,
        min_amp: float = 1e-12,
        max_amp: float = 1e12,
        converge_mean: float = 1e-3,
        smooth_window: int = 5,
        envelope: LMD_ENVELOPE_MODS_ARGS | LMD_ENVELOPE_MODS = "midpoint",
        hilbert_mod: HILBERT_MODS_ARGS | HILBERT_MODS = "Scipy",
        hilbert_backend: Literal["Scipy", "scipy", "FHT"] | None = None,
        dtype: type | np.dtype | None = None,
        **kwargs,
    ):
        """
        Parameters
        ----------
        max_pf : int | None
            Maximum number of product functions. None selects log2(num_imfs), -1
            decomposes completely.
        max_iter : int
            Maximum sifting iterations per product function (midpoint
            envelope only).
        eps : float
            Envelope convergence threshold (midpoint envelope only).
        eps_stable : float
            Minimum energy of an accepted product function.
        min_amp : float
            Lower bound of the amplitude estimate.
        max_amp : float
            Upper bound of the amplitude estimate.
        converge_mean : float
            Mean convergence threshold (midpoint envelope only).
        smooth_window : int
            Savitzky-Golay window length for the amplitude estimate.
        envelope : LMD_ENVELOPE_MODS_ARGS | LMD_ENVELOPE_MODS
            Amplitude-estimate strategy: ``"midpoint"`` (default) = classical
            Smith sift with extrema (Pchip) interpolation, the modal-
            decomposition envelope; ``"hilbert"`` = single-shot analytic
            demodulation per PF (``|H[h(t)]|``), only for users who genuinely
            need the analytic envelope — it is NOT the sift envelope and does
            not converge when iterated.
        hilbert_mod : HILBERT_MODS_ARGS | HILBERT_MODS
            Hilbert backend when ``envelope="hilbert"``, taken from
            ``Utils.Hilbert``'s ``HILBERT_MODS``: ``"Scipy"`` (default,
            scipy.signal.hilbert; the historical spelling is case-normalised
            onto the canonical ``"scipy"`` / ``HILBERT_MODS.Scipy``) or
            ``"FHT"`` (compiled third-party C kernel); the remaining declared
            names are accepted by the enum but not implemented by the
            dispatcher. The alias ``hilbert_backend`` is also accepted.
        hilbert_backend : {"Scipy", "scipy", "FHT"} | None
            Deprecated alias of ``hilbert_mod`` (kept for the historical API).
        dtype : type | np.dtype | None
            Working dtype of the returned product functions. ``None`` keeps
            the input precision; integer/bool inputs are promoted to float64.
        **kwargs
            ``safe`` (validation level, see ``Utils.Check``) and ``copy``
            (hand the decomposer its own copy of the signal).

        Strategy comparison (measured, n=16384, tests/comparison):

        +------------------+------------------------------------------+------------------------------------------+
        | strategy         | pros                                     | cons                                     |
        +==================+==========================================+==========================================+
        | midpoint         | classical Smith semantics; smooth PFs;  | slowest of the three (~0.13-0.19 s;      |
        | (default)        | reconstruction fixed (~1e-15, was       | multi-round sifting per PF); weak        |
        |                  | ~2.4e-4 = 2 ulp(1e12) from the cubic-   | separation of multi-tone/wideband        |
        |                  | spline overshoot explosion)             | signals; requires the shape-preserving   |
        |                  |                                          | (Pchip) amplitude interpolation          |
        +------------------+------------------------------------------+------------------------------------------+
        | hilbert + Scipy  | fastest (~0.009 s, single-shot          | trend recovery is weaker (~0.7 corr);    |
        |                  | demodulation per PF, ~20x midpoint);    | iterated Hilbert demodulation does NOT   |
        |                  | exact reconstruction (~1e-15); best     | converge on multi-component signals      |
        |                  | AM-FM / two-tone recovery (0.99+)       | (single-shot only); envelope sensitive   |
        |                  |                                          | to noise                                 |
        +------------------+------------------------------------------+------------------------------------------+
        | hilbert + FHT    | bit-identical to the Scipy path         | the C kernel itself is ~1.9x slower per  |
        | (third-party C)  | (diff ~1e-16) in this pipeline;         | Hilbert call than scipy's FFT (hidden    |
        |                  | independent C backend (SAO Discrete     | inside LMD); needs the compiled          |
        |                  | Hilbert/Fourier/Hartley Transforms)     | ``_fht_native`` extension (falls back    |
        |                  |                                          | to the NumPy reference otherwise)        |
        +------------------+------------------------------------------+------------------------------------------+
        """
        self.max_pf: int | None = max_pf
        self.max_iter: int = max_iter
        self.eps: float = eps
        self.eps_stable: float = eps_stable
        self.min_amp: float = min_amp
        self.max_amp: float = max_amp
        self.converge_mean: float = converge_mean
        self.smooth_window: int = smooth_window
        self.envelope: LMD_ENVELOPE_MODS | LMD_ENVELOPE_MODS_ARGS = envelope

        # 历史别名: hilbert_mod 仍是默认写法 (未显式给出) 时用 hilbert_backend 覆盖
        if hilbert_backend is not None and hilbert_mod in ("Scipy", HILBERT_MODS.Scipy):
            hilbert_mod = hilbert_backend
        self.hilbert_mod: HILBERT_MODS | HILBERT_MODS_ARGS = hilbert_mod

        self.dtype: type | np.dtype | None = dtype

        unknown = set(kwargs) - {"safe", "copy"}
        if unknown:
            raise TypeError(f"unexpected keyword arguments: {unknown}")

        self.safe: str = kwargs.get("safe", "unknown")
        self.copy: bool = kwargs.get("copy", True)

        self.imfs: np.ndarray | None = None
        self.res: np.ndarray | None = None

        self.param_check()

    def param_check(self) -> None:
        self.envelope = Check.validate_mod(self.envelope, LMD_ENVELOPE_MODS)

        # ``HILBERT_MODS`` 的成员值是 canonical 名 (``"scipy"`` 小写), 历史写法
        # ``"Scipy"`` 直接喂给 ``HILBERT_MODS("Scipy")`` 会 ValueError; 失败后按
        # 大小写归一再匹配一次, 最终实例上一定是 Enum 成员。
        try:
            self.hilbert_mod = Check.validate_mod(self.hilbert_mod, HILBERT_MODS)
        except ValueError:
            self.hilbert_mod = _validate_hilbert_mod(self.hilbert_mod)

        if self.dtype is not None and np.dtype(self.dtype).kind != "f":
            raise ValueError("the `dtype` must be a float numpy dtype or None.")

        if not isinstance(self.copy, bool):
            raise TypeError("the `copy` must be bool")

    def config_generate(self) -> LMDConfig:
        return LMDConfig(
            max_pf=self.max_pf,
            max_iter=self.max_iter,
            eps=self.eps,
            eps_stable=self.eps_stable,
            min_amp=self.min_amp,
            max_amp=self.max_amp,
            converge_mean=self.converge_mean,
            smooth_window=self.smooth_window,
            envelope=self.envelope,
            hilbert_mod=self.hilbert_mod,
            dtype=self.dtype,
        )

    def decompose(self, S, T=None) -> DecompositionResult:
        """
        Decompose the signal into product functions and a residual.
        """
        scipy_signal = Cache.import_module("scipy.signal")
        argrelextrema = scipy_signal.argrelextrema
        savgol_filter = scipy_signal.savgol_filter

        # LMD 只在样本序号上插值, 不使用外部时间轴
        S, N = Check_Time_and_Signal(S, T=None, ndim={1}, method=self.name,
                                     dtype=self.dtype, need_T=False, safe=self.safe, copy=self.copy)

        if not (S.dtype.kind == "f" and S.dtype.itemsize >= 4):  # int/bool 会让包络除法直接报错, f16 会让样条出非有限值
            S = Check.to_dtype(S, np.dtype(np.float64))

        # 每次调用都从干净状态开始, 同一实例可反复复用
        self.imfs = None
        self.res = None

        n_samples = S.size

        if n_samples < 8:
            raise ValueError("LMD requires S length >= 8 (Smith 2005 standard)")

        t = np.arange(n_samples, dtype=np.float64)

        max_pf = self.max_pf
        if max_pf is None:
            max_pf = int(np.log2(n_samples))

        match self.envelope:
            case LMD_ENVELOPE_MODS.Hilbert:
                PFs, residue = self._sift_hilbert(S, n_samples, t, max_pf,
                                                  argrelextrema, savgol_filter)
            case _:
                PFs, residue = self._sift_midpoint(S, n_samples, t, max_pf,
                                                   argrelextrema, savgol_filter)

        self.imfs = np.array(PFs, dtype=S.dtype) if len(PFs) \
            else np.empty((0, n_samples), dtype=S.dtype)
        self.res = residue

        info = \
        {
            "envelope": self.envelope,
            "hilbert_mod": self.hilbert_mod,
            "max_pf": max_pf,
        }

        return DecompositionResult(
            IMFs=self.imfs,
            Res=self.res,
            info=info,
            config=self.config_generate(),
        )

    # ------------------------------------------------------------------
    def _sift_midpoint(self, S, n_samples, t, max_pf, argrelextrema, savgol_filter):
        """
        Extrema-midpoint sift with two safeguards against the historical
        amplitude explosion (see module docstring):

        * the amplitude envelope is interpolated with shape-preserving Pchip
          (no overshoot), and
        * the accumulated amplitude and the PF samples are clamped to
          1e4 x the ORIGINAL signal scale (ill-conditioned demodulation
          points with tiny envelopes can otherwise inject huge spikes).

        The local mean of the final sift iteration stays in the residue on
        purpose (classical LMD semantics); it is NOT part of the
        reconstruction error.
        """
        residue = S.copy()
        PFs: list[np.ndarray] = []

        # Physical scale fixed at the ORIGINAL signal (not the residue, whose
        # scale is inflated by any spike that already leaked into a PF):
        amp_ref = max(float(np.max(np.abs(S))), self.min_amp)
        amp_cap = min(self.max_amp, 1e4 * amp_ref)
        pf_cap = min(self.max_amp, 1e4 * amp_ref)

        while max_pf == -1 or len(PFs) < max_pf:
            h = residue.copy()
            a_total = np.ones(n_samples, dtype=np.float64)
            converged = False

            for _ in range(self.max_iter):
                max_loc = argrelextrema(h, np.greater)[0]
                min_loc = argrelextrema(h, np.less)[0]
                ext_idx = np.unique(np.concatenate([max_loc, min_loc]))

                if len(ext_idx) < 3:
                    break

                ext_idx, ext_vals = mirror_extrema(
                    ext_idx, h[ext_idx], nbsym=0,
                    edges=(float(h[0]), float(h[-1]), n_samples),
                )

                t_mid = (ext_idx[:-1] + ext_idx[1:]) / 2.0
                m_vals = (ext_vals[:-1] + ext_vals[1:]) / 2.0
                a_vals = np.abs(ext_vals[:-1] - ext_vals[1:]) / 2.0

                m_t = _safe_interpolate(t_mid, m_vals, t)
                a_t = _safe_interpolate_pchip(t_mid, a_vals, t)

                a_t = np.clip(a_t, self.min_amp, self.max_amp)
                a_t = savgol_filter(a_t, self.smooth_window, 2)

                if _check_convergence(m_t, a_t, self.eps, self.converge_mean):
                    converged = True
                    break

                s_new = (h - m_t) / a_t
                a_total = np.clip(a_total * a_t, self.min_amp, amp_cap)
                h = s_new

            if not converged:
                break

            current_pf = np.clip(a_total * s_new, -pf_cap, pf_cap)

            if np.any(np.isnan(current_pf)) or np.any(np.isinf(current_pf)):
                break

            if np.sum(current_pf ** 2) < self.eps_stable:
                break

            PFs.append(current_pf)
            residue -= current_pf

            if (
                monotonic(residue, mod="monotonic", safe="safe") or
                np.sum(residue ** 2) < self.eps_stable or
                len(argrelextrema(residue, np.greater)[0]) + len(argrelextrema(residue, np.less)[0]) <= 2
            ):
                break

        return PFs, residue

    # ------------------------------------------------------------------
    def _sift_hilbert(self, S, n_samples, t, max_pf, argrelextrema, savgol_filter):
        """
        Hilbert-envelope variant: single-shot demodulation per PF.

            a(t) = |H[h(t)]| (smoothed),  s_FM = (h - m_t) / a,
            PF = a * s_FM = h - m_t

        Iterating this demodulation is deliberately NOT done: on
        multi-component signals the analytic envelope beats and the sift
        diverges (extrema count grows without bound), which is why the
        historical Hilbert variant was shelved.
        """
        residue = S.copy()
        PFs: list[np.ndarray] = []

        while max_pf == -1 or len(PFs) < max_pf:
            h = residue.copy()

            max_loc = argrelextrema(h, np.greater)[0]
            min_loc = argrelextrema(h, np.less)[0]
            ext_idx = np.unique(np.concatenate([max_loc, min_loc]))

            if len(ext_idx) < 3:
                break

            ext_idx, ext_vals = mirror_extrema(
                ext_idx, h[ext_idx], nbsym=0,
                edges=(float(h[0]), float(h[-1]), n_samples),
            )

            t_mid = (ext_idx[:-1] + ext_idx[1:]) / 2.0
            m_vals = (ext_vals[:-1] + ext_vals[1:]) / 2.0

            m_t = _safe_interpolate(t_mid, m_vals, t)
            a_t = np.abs(hilbert(h, self.hilbert_mod))

            a_t = np.clip(a_t, self.min_amp, self.max_amp)
            a_t = savgol_filter(a_t, self.smooth_window, 2)

            s_new = (h - m_t) / a_t
            current_pf = a_t * s_new

            if np.any(np.isnan(current_pf)) or np.any(np.isinf(current_pf)):
                break

            if np.sum(current_pf ** 2) < self.eps_stable:
                break

            PFs.append(current_pf)
            residue -= current_pf

            if (
                monotonic(residue, mod="monotonic", safe="safe") or
                np.sum(residue ** 2) < self.eps_stable or
                len(argrelextrema(residue, np.greater)[0]) + len(argrelextrema(residue, np.less)[0]) <= 2
            ):
                break

        return PFs, residue


def _validate_hilbert_mod(mod) -> HILBERT_MODS:
    """
    Case-insensitive fallback for the ``hilbert_mod`` argument (backward
    compatibility with the historical spelling ``"Scipy"``).

    ``Check.validate_mod`` / ``HILBERT_MODS(value)`` match the **canonical**
    member values, and ``HILBERT_MODS.Scipy`` is ``"scipy"`` (lower case), so
    ``HILBERT_MODS("Scipy")`` raises ``ValueError``. This helper normalises the
    string with ``str.casefold`` and matches it against the member values once
    more; anything that still does not hit raises ``ValueError`` just like
    ``Check.validate_mod``.

    Returns
    -------
    HILBERT_MODS
        The matching enum member (never the raw string).
    """
    if isinstance(mod, str):
        key = mod.casefold()
        for member in HILBERT_MODS:
            if member.value.casefold() == key:
                return member

    raise ValueError(
        f"the parameter of `hilbert_mod` can only be {HILBERT_MODS._member_names_}"
    )


def _safe_interpolate(x: np.ndarray, y: np.ndarray, x_new: np.ndarray) -> np.ndarray:
    """自然三次样条插值 (scipy.interpolate.CubicSpline), 界外线性外延 + np.interp 兜底。"""
    try:
        interpolate = Cache.import_module("scipy.interpolate")
        sp = interpolate.CubicSpline(x, y, bc_type="natural", extrapolate=True)
        res = sp(x_new)
    except Exception:
        res = np.interp(x_new, x, y)

    left_mask = x_new < x[0]
    right_mask = x_new > x[-1]
    if np.any(left_mask):
        slope = (y[1] - y[0]) / (x[1] - x[0])
        res[left_mask] = y[0] + slope * (x_new[left_mask] - x[0])
    if np.any(right_mask):
        slope = (y[-1] - y[-2]) / (x[-1] - x[-2])
        res[right_mask] = y[-1] + slope * (x_new[right_mask] - x[-1])

    return res


def _safe_interpolate_pchip(x: np.ndarray, y: np.ndarray, x_new: np.ndarray) -> np.ndarray:
    """
    Shape-preserving (Pchip) interpolation for the amplitude envelope
    (scipy.interpolate.PchipInterpolator; extrapolate=False, 界外 NaN ->
    np.interp 兜底)。

    Unlike the natural cubic spline, Pchip cannot overshoot between the data
    points, which keeps ``a_t`` within the extrema amplitude range and stops
    the multiplicative ``a_total`` explosion of the old code.
    """
    try:
        interpolate = Cache.import_module("scipy.interpolate")
        sp = interpolate.PchipInterpolator(x, y, extrapolate=False)
        res = sp(x_new)
        res = np.where(np.isnan(res), np.interp(x_new, x, y), res)
    except Exception:
        res = np.interp(x_new, x, y)
    return res


def _check_convergence(m_t: np.ndarray, a_t: np.ndarray, eps: float, converge_mean) -> bool:
    mean_ok = np.max(np.abs(m_t)) < converge_mean
    envelope_ok = np.max(np.abs(a_t - 1.0)) < eps
    return mean_ok and envelope_ok


def mirror_extrema(
    idx: np.ndarray,
    vals: np.ndarray,
    nbsym: int = 3,
    edges = None,
    _dtype: np.dtype = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    端点镜像外延 (EMD nbsym / LMD 边界两用, 全库共享)。

    Parameters
    ----------
    idx : np.ndarray (1-D)
        极值位置 (须严格递增)。
    vals : np.ndarray (1-D)
        与 ``idx`` 对应的极值取值 (``signal[idx]``)。
    nbsym : int
        EMD 语义: 每端镜像的极值个数 (默认 0 = 不镜像; 极值少于 2 个时跳过)。
    edges : tuple[float, float, int] | None
        LMD 语义: ``(信号首值, 信号末值, 样本数)``; None 时不补边界点。

    Returns
    -------
    (positions, values)
        ``positions`` : np.ndarray (float64) — 镜像后位置, 严格递增;
        ``values``    : np.ndarray — 与输入 ``vals`` 同 dtype 的对应取值。

    Examples
    --------
    >>> import numpy as np
    >>> from Modal_Decomposition.LMD import mirror_extrema
    >>> idx = np.array([5, 9]); vals = np.array([2.0, 3.0])
    >>> pos, val = mirror_extrema(idx, vals, nbsym=1)
    >>> pos.tolist()   # 1 个左镜像 + 原极值 + 1 个右镜像
    [1.0, 5.0, 9.0, 13.0]
    """
    vals = Check.to_array(vals, dtype=_dtype, ndim=1)

    _dtype = vals.dtype if _dtype is None else _dtype

    idx = Check.to_array(idx, dtype=_dtype, ndim=1)

    if idx.size != vals.size:
        raise ValueError("mirror_extrema requires idx == vals")

    positions = idx
    values = vals
    if positions.size == 0:
        return positions, values

    # --- EMD nbsym 语义: 每端镜像 nbsym 个极值 ---------------------------- #
    if nbsym > 0 and len(positions) >= 2:
        n = min(nbsym, len(positions) - 1)
        p = len(positions)
        L = p + 2* n

        new_positions = np.empty(L, dtype=positions.dtype)
        new_values = np.empty(L, dtype=values.dtype)

        # 左侧扩展
        new_positions[:n] = 2 * positions[0] - positions[1:n + 1][::-1]
        new_values[:n] = values[1:n + 1][::-1]

        # 中间原始数据
        new_positions[n:n + p] = positions
        new_values[n:n + p] = values

        # 右侧扩展
        new_positions[n + p:] = 2 * positions[-1] - positions[-2:-(n + 2):-1]
        new_values[n + p:] = values[-1 - n:-1][::-1]

        positions = new_positions
        values = new_values

    # --- LMD 边界语义: 未覆盖边界处奇反射补点 ----------------------------- #
    if edges is not None:
        lo, hi, n_samples = float(edges[0]), float(edges[1]), int(edges[2])

        left_pad = positions[0] > 0
        right_pad = positions[-1] < n_samples - 1

        total_len = len(positions) + int(left_pad) + int(right_pad)
        new_positions = np.empty(total_len, dtype=positions.dtype)
        new_values = np.empty(total_len, dtype=values.dtype)

        idx = 0
        if left_pad:
            new_positions[0] = 0.0
            new_values[0] = 2.0 * lo - float(values[0])
            idx = 1

        new_positions[idx:idx + len(positions)] = positions
        new_values[idx:idx + len(values)] = values
        idx += len(positions)

        if right_pad:
            new_positions[idx] = float(n_samples - 1)
            new_values[idx] = 2.0 * hi - float(values[-1])

        positions = new_positions
        values = new_values

    return positions, values


"""
other function for module
"""
def f_LMD(S: list | np.ndarray, T=None, *, max_pf: int | None = None, max_iter: int = 37, eps: float = 0.05, eps_stable: float = 1e-12, min_amp: float = 1e-12, max_amp: float = 1e12, converge_mean: float = 0.001, smooth_window: int = 5, envelope: LMD_ENVELOPE_MODS_ARGS | LMD_ENVELOPE_MODS = "midpoint", hilbert_mod: HILBERT_MODS_ARGS | HILBERT_MODS = "Scipy", hilbert_backend: Literal["Scipy", "scipy", "FHT"] | None = None, dtype: type | np.dtype | None = None, **kwargs) -> DecompositionResult:
    lmd = LMD(
        max_pf=max_pf,
        max_iter=max_iter,
        eps=eps,
        eps_stable=eps_stable,
        min_amp=min_amp,
        max_amp=max_amp,
        converge_mean=converge_mean,
        smooth_window=smooth_window,
        envelope=envelope,
        hilbert_mod=hilbert_mod,
        hilbert_backend=hilbert_backend,
        dtype=dtype,
        **kwargs
    )

    return lmd.decompose(S, T)
