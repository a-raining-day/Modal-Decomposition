"""
Local Mean Decomposition

Extracts product functions by iteratively smoothing the local mean and
envelope estimate from extrema interpolation.

包络概念区分 (Hilbert vs Spline)
--------------------------------
- 模态分解筛分迭代里的"包络 / 局部均值"是对**极值**做插值得到 —— 本方法默认
  ``envelope="midpoint"`` (Smith 经典筛分): 幅度包络用保形 Pchip 插值,
  局部均值用自然三次样条 (均经 ``Utils.Spline`` 提供; 极值端点镜像经
  ``Utils.Mirror``)。自然三次样条曾因极值间过冲导致幅度爆炸 (峰值 ~1e12,
  重构损失 ~2.4e-4), 故幅度一律走 Pchip; 最后一次筛分的局部均值按经典
  LMD 语义留在残差中。
- 解析 Hilbert 包络 ``a(t) = |H[h(t)]|`` **不是**筛分迭代中的包络: 只有确实
  需要解析幅度时才使用 (``envelope="hilbert"`` 显式选择, 单次解调 / 实验
  对比)。迭代该解调在多分量信号上不收敛 (包络拍频、极值数发散), 故该变体
  是搁置的历史选项。Hilbert 后端经 ``**kwargs`` 传入:
  ``hilbert_mod="Scipy"`` (默认) 或 ``hilbert_mod="FHT"`` (第三方 C 内核,
  见 ``Utils.Hilbert``)。

References
----------
10.1098/rsif.2005.0058
"""

import numpy as np
from dataclasses import dataclass
from typing import ClassVar, Literal

from .Base import Config, Decomposer, DecompositionResult
from ._Registry import register_class
from .Utils import (
    Check_Time_and_Signal,
    get_mirror,
    get_spline,
    is_monotonic,
)
from .Utils.Hilbert import hilbert

__all__ = ["LMD", "LMDConfig"]


@dataclass(frozen=True, kw_only=True)
class LMDConfig(Config):
    """
    Effective parameters of an LMD run.
    """
    max_pf: int
    max_iter: int
    eps: float
    eps_stable: float
    min_amp: float
    max_amp: float
    converge_mean: float
    smooth_window: int
    envelope: str
    hilbert_mod: str


@register_class("LMD")
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
        envelope: Literal["midpoint", "hilbert"] = "midpoint",
        **kwargs,
    ):
        """
        Parameters
        ----------
        max_pf : int | None
            Maximum number of product functions. None selects log2(N), -1
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
        envelope : {"midpoint", "hilbert"}
            Amplitude-estimate strategy: ``"midpoint"`` (default) = classical
            Smith sift with extrema (Pchip) interpolation, the modal-
            decomposition envelope; ``"hilbert"`` = single-shot analytic
            demodulation per PF (``|H[h(t)]|``), only for users who genuinely
            need the analytic envelope — it is NOT the sift envelope and does
            not converge when iterated.
        **kwargs
            Hilbert backend when ``envelope="hilbert"``: ``hilbert_mod="Scipy"``
            (default, scipy.signal.hilbert) or ``hilbert_mod="FHT"`` (compiled
            third-party C kernel). The alias ``hilbert_backend`` is also accepted.

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
        self.max_pf = max_pf
        self.max_iter = max_iter
        self.eps = eps
        self.eps_stable = eps_stable
        self.min_amp = min_amp
        self.max_amp = max_amp
        self.converge_mean = converge_mean
        self.smooth_window = smooth_window

        if envelope not in ("midpoint", "hilbert"):
            raise ValueError(
                f"envelope must be 'midpoint' or 'hilbert', got {envelope!r}"
            )
        self.envelope = envelope

        hilbert_mod = kwargs.pop("hilbert_mod", kwargs.pop("hilbert_backend", "Scipy"))
        if hilbert_mod not in ("Scipy", "FHT"):
            raise ValueError(
                f"hilbert_mod must be 'Scipy' or 'FHT', got {hilbert_mod!r}"
            )
        self.hilbert_mod = hilbert_mod

        if kwargs:
            raise TypeError(
                f"unexpected keyword arguments: {sorted(kwargs)}"
            )

    def decompose(self, S, T=None) -> DecompositionResult:
        """
        Decompose the signal into product functions and a residual.
        """
        from scipy.signal import argrelextrema, savgol_filter

        S, T, _N = Check_Time_and_Signal(S, T, ndim={1}, method=self.name)

        S = np.asarray(S, dtype=np.float64).ravel()
        n_samples = S.size

        if n_samples < 8:
            raise ValueError("LMD requires S length >= 8 (Smith 2005 standard)")

        t = np.arange(n_samples, dtype=np.float64)

        max_pf = self.max_pf
        if max_pf is None:
            max_pf = int(np.log2(n_samples))

        if self.envelope == "hilbert":
            PFs, residue = self._sift_hilbert(S, n_samples, t, max_pf,
                                              argrelextrema, savgol_filter)
        else:
            PFs, residue = self._sift_midpoint(S, n_samples, t, max_pf,
                                               argrelextrema, savgol_filter)

        IMFs = np.array(PFs, dtype=np.float64)
        if IMFs.shape[0] == 0:
            IMFs = np.empty((0, n_samples), dtype=np.float64)

        return DecompositionResult(
            IMFs,
            residue,
            {"envelope": self.envelope, "hilbert_mod": self.hilbert_mod},
            LMDConfig(
                max_pf=max_pf,
                max_iter=self.max_iter,
                eps=self.eps,
                eps_stable=self.eps_stable,
                min_amp=self.min_amp,
                max_amp=self.max_amp,
                converge_mean=self.converge_mean,
                smooth_window=self.smooth_window,
                envelope=self.envelope,
                hilbert_mod=self.hilbert_mod,
            ),
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

                ext_idx, ext_vals = get_mirror().mirror_extrema(
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
                is_monotonic(residue) or
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

            ext_idx, ext_vals = get_mirror().mirror_extrema(
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
                is_monotonic(residue) or
                np.sum(residue ** 2) < self.eps_stable or
                len(argrelextrema(residue, np.greater)[0]) + len(argrelextrema(residue, np.less)[0]) <= 2
            ):
                break

        return PFs, residue


def _safe_interpolate(x: np.ndarray, y: np.ndarray, x_new: np.ndarray) -> np.ndarray:
    """自然三次样条插值 (经 Utils.Spline), 界外线性外延 + np.interp 兜底。"""
    try:
        sp = get_spline().spline(
            x, y, spline_kind="CubicSpline", bc_type="natural", extrapolate=True
        )
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
    (经 Utils.Spline; extrapolate=False, 界外 NaN -> np.interp 兜底)。

    Unlike the natural cubic spline, Pchip cannot overshoot between the data
    points, which keeps ``a_t`` within the extrema amplitude range and stops
    the multiplicative ``a_total`` explosion of the old code.
    """
    try:
        sp = get_spline().spline(x, y, spline_kind="PCHIP", extrapolate=False)
        res = sp(x_new)
        res = np.where(np.isnan(res), np.interp(x_new, x, y), res)
    except Exception:
        res = np.interp(x_new, x, y)
    return res


def _check_convergence(m_t: np.ndarray, a_t: np.ndarray, eps: float, converge_mean) -> bool:
    mean_ok = np.max(np.abs(m_t)) < converge_mean
    envelope_ok = np.max(np.abs(a_t - 1.0)) < eps
    return mean_ok and envelope_ok
