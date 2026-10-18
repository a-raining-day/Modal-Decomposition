"""
Metrics for the denoising benchmark.

The generic metrics compare a denoised signal ``y`` with the clean reference.
Feature metrics are case-specific and are only computed when the case metadata
provides the required ground truth (known bands, impact times, step times,
spike locations).
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
from scipy.signal import hilbert

FS_DEFAULT = 1000.0


def _as_float(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.float64)


def _trim(x: np.ndarray, trim: int) -> np.ndarray:
    if trim <= 0:
        return x
    if x.size <= 2 * trim:
        return x
    return x[trim:-trim]


def rms(x: np.ndarray) -> float:
    x = _as_float(x)
    return float(np.sqrt(np.mean(x * x))) if x.size else float("nan")


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    a, b = _as_float(a), _as_float(b)
    if a.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean((a - b) ** 2)))


def snr_db(clean: np.ndarray, y: np.ndarray) -> float:
    """10*log10(||clean||^2 / ||clean-y||^2); inf if the residual is zero."""
    clean, y = _as_float(clean), _as_float(y)
    num = float(np.dot(clean, clean))
    den = float(np.dot(clean - y, clean - y))
    if num <= 0.0:
        return float("nan")
    if den <= 0.0:
        return float("inf")
    return float(10.0 * np.log10(num / den))


def correlation(a: np.ndarray, b: np.ndarray) -> float:
    a, b = _as_float(a), _as_float(b)
    if a.size < 2 or a.std() <= 0.0 or b.std() <= 0.0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _weighted_power(x: np.ndarray, fs: float = FS_DEFAULT) -> tuple[np.ndarray, np.ndarray]:
    """One-sided PSD with proper DC/Nyquist weights (sum/n = time energy)."""
    x = _as_float(x)
    n = x.size
    spec = np.abs(np.fft.rfft(x - x.mean())) ** 2
    if n % 2 == 0:
        spec[1:-1] *= 2.0
    else:
        spec[1:] *= 2.0
    freqs = np.fft.rfftfreq(n, 1.0 / fs)
    return spec, freqs


def band_energy(x: np.ndarray, f1: float, f2: float, fs: float = FS_DEFAULT) -> float:
    x = _as_float(x)
    if x.size < 2:
        return 0.0
    spec, freqs = _weighted_power(x, fs)
    return float(spec[(freqs >= f1) & (freqs < f2)].sum() / x.size)


def band_fraction(
    x: np.ndarray, f1: float, f2: float, fs: float = FS_DEFAULT
) -> float:
    x = _as_float(x)
    if x.size < 2:
        return float("nan")
    spec, freqs = _weighted_power(x, fs)
    total = float(spec.sum())
    if total <= 0.0:
        return float("nan")
    return float(spec[(freqs >= f1) & (freqs < f2)].sum() / total)


def band_retention(
    y: np.ndarray,
    clean: np.ndarray,
    f1: float,
    f2: float,
    fs: float = FS_DEFAULT,
) -> float:
    """Energy in [f1, f2) retained relative to the clean signal (1.0 is ideal)."""
    den = band_energy(clean, f1, f2, fs)
    if den <= 0.0:
        return float("nan")
    return float(band_energy(y, f1, f2, fs) / den)


def spectral_flatness(x: np.ndarray, eps: float = 1e-30) -> float:
    x = _as_float(x)
    if x.size < 2:
        return float("nan")
    x = x - x.mean()
    spec = np.abs(np.fft.rfft(x)) ** 2 + eps
    return float(np.exp(np.mean(np.log(spec))) / np.mean(spec))


def best_scale_nrmse(clean: np.ndarray, y: np.ndarray) -> float:
    """NRMSE after the best affine rescaling ``a*y+b``.

    This measures *shape* rather than amplitude: an attenuated or offset
    output is not penalised if its waveform shape is right.
    """
    clean, y = _as_float(clean), _as_float(y)
    if clean.size < 2 or y.size != clean.size:
        return float("nan")
    design = np.column_stack([y, np.ones_like(y)])
    coef, *_ = np.linalg.lstsq(design, clean, rcond=None)
    fit = design @ coef
    denom = float(np.sum((clean - clean.mean()) ** 2))
    if denom <= 0.0:
        return float("nan")
    return float(np.sqrt(np.sum((fit - clean) ** 2) / denom))


def log_spectral_distance(
    clean: np.ndarray, y: np.ndarray, eps: float = 1e-12
) -> float:
    clean, y = _as_float(clean), _as_float(y)
    if clean.size < 2 or y.size < 2:
        return float("nan")
    pc = np.abs(np.fft.rfft(clean - clean.mean())) ** 2 + eps
    py = np.abs(np.fft.rfft(y - y.mean())) ** 2 + eps
    return float(np.sqrt(np.mean((np.log(py) - np.log(pc)) ** 2)))


def clean_bandwidth(
    clean: np.ndarray, fs: float = FS_DEFAULT, frac: float = 0.99
) -> float:
    """Smallest frequency below which ``frac`` of the clean energy lies."""
    clean = _as_float(clean)
    if clean.size < 2:
        return float("nan")
    spec, freqs = _weighted_power(clean, fs)
    total = float(spec.sum())
    if total <= 0.0:
        return float("nan")
    idx = int(np.searchsorted(np.cumsum(spec), frac * total))
    idx = min(idx, freqs.size - 1)
    return float(freqs[idx])


def _lowpass_fft(
    x: np.ndarray, fc: float, fs: float = FS_DEFAULT
) -> np.ndarray:
    """Zero-phase ideal low-pass via FFT (used only for the trend metric)."""
    x = _as_float(x)
    if x.size < 2:
        return x.copy()
    spec = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(x.size, 1.0 / fs)
    spec[freqs > fc] = 0.0
    return np.fft.irfft(spec, n=x.size)


def _bandpass_envelope(
    x: np.ndarray, f1: float, f2: float, fs: float = FS_DEFAULT
) -> np.ndarray:
    x = _as_float(x)
    if x.size < 2:
        return np.abs(x)
    spec = np.fft.rfft(x - x.mean())
    freqs = np.fft.rfftfreq(x.size, 1.0 / fs)
    spec[(freqs < f1) | (freqs > f2)] = 0.0
    return np.abs(hilbert(np.fft.irfft(spec, n=x.size)))


def envelope_spectrum_peak(
    x: np.ndarray,
    fs: float = FS_DEFAULT,
    band: tuple[float, float] = (120.0, 240.0),
    f_range: tuple[float, float] = (5.0, 20.0),
) -> tuple[float, float]:
    """Return ``(peak_frequency, peak_to_median_ratio)`` of the envelope spectrum."""
    env = _bandpass_envelope(x, band[0], band[1], fs)
    env = env - env.mean()
    spec = np.abs(np.fft.rfft(env))
    freqs = np.fft.rfftfreq(env.size, 1.0 / fs)
    mask = (freqs >= f_range[0]) & (freqs <= f_range[1])
    if not np.any(mask):
        return float("nan"), float("nan")
    spec = spec[mask]
    freqs = freqs[mask]
    i = int(np.argmax(spec))
    med = float(np.median(spec))
    ratio = float(spec[i] / med) if med > 0.0 else float("inf")
    return float(freqs[i]), ratio


def envelope_correlation(
    y: np.ndarray,
    clean: np.ndarray,
    fs: float = FS_DEFAULT,
    band: tuple[float, float] = (120.0, 240.0),
) -> float:
    e1 = _bandpass_envelope(y, band[0], band[1], fs)
    e2 = _bandpass_envelope(clean, band[0], band[1], fs)
    return correlation(e1, e2)


def kurtosis(x: np.ndarray) -> float:
    x = _as_float(x)
    x = x - x.mean()
    var = float(np.mean(x * x))
    if var <= 0.0:
        return float("nan")
    return float(np.mean(x**4) / (var * var))


def _step_metrics(
    y: np.ndarray,
    clean: np.ndarray,
    step_times: Iterable[float],
    fs: float = FS_DEFAULT,
) -> dict:
    """10-90% edge width, overshoot, and plateau RMSE around known steps."""
    y, clean = _as_float(y), _as_float(clean)
    widths: list[float] = []
    overshoots: list[float] = []
    for t0 in step_times:
        idx = int(round(t0 * fs))
        if idx <= 0 or idx >= y.size - 1:
            continue
        half = min(25, idx, y.size - idx)
        pre = float(np.median(clean[max(0, idx - 100) : idx]))
        post = float(np.median(clean[idx : min(y.size, idx + 100)]))
        height = post - pre
        if abs(height) < 1e-12:
            continue
        lo = pre + 0.1 * height
        hi = pre + 0.9 * height

        def _cross(level: float, sign: float) -> int | None:
            search = np.arange(max(0, idx - half), min(y.size, idx + half))
            vals = sign * (y[search] - level)
            hit = np.flatnonzero(vals >= 0.0)
            return int(search[hit[0]]) if hit.size else None

        sign = 1.0 if height > 0 else -1.0
        i10 = _cross(lo, sign)
        i90 = _cross(hi, sign)
        if i10 is not None and i90 is not None:
            widths.append(float(abs(i90 - i10)))
        after = y[idx : min(y.size, idx + half)]
        if sign > 0:
            overshoots.append(float(np.max(after) - post))
        else:
            overshoots.append(float(post - np.min(after)))

    # plateau RMSE: exclude a window around each step
    mask = np.ones(y.size, dtype=bool)
    for t0 in step_times:
        idx = int(round(t0 * fs))
        mask[max(0, idx - 100) : min(y.size, idx + 100)] = False
    plateau_rmse = rmse(y[mask], clean[mask]) if np.any(mask) else float("nan")

    return {
        "edge_width_samples": float(np.mean(widths)) if widths else float("nan"),
        "overshoot": float(np.max(overshoots)) if overshoots else float("nan"),
        "plateau_rmse": plateau_rmse,
    }


def _impact_metrics(
    y: np.ndarray,
    clean: np.ndarray,
    meta: dict,
    fs: float = FS_DEFAULT,
    trim: int = 50,
) -> dict:
    band = tuple(meta.get("env_band", (120.0, 240.0)))
    peak_freq, peak_snr = envelope_spectrum_peak(_trim(y, trim), fs, band)
    idx = [i for i in meta.get("impact_idx", []) if trim <= i < y.size - trim]
    ratios: list[float] = []
    for i in idx:
        lo = i - 10
        hi = i + 20
        if lo < 0 or hi > y.size:
            continue
        cy = float(np.max(np.abs(clean[lo:hi])))
        if cy > 1e-12:
            ratios.append(float(np.max(np.abs(y[lo:hi])) / cy))
    return {
        "env_peak_freq": peak_freq,
        "env_peak_snr": peak_snr,
        "env_corr": envelope_correlation(_trim(y, trim), _trim(clean, trim), fs, band),
        "kurtosis": kurtosis(_trim(y, trim)),
        "impact_amp_ratio": float(np.median(ratios)) if ratios else float("nan"),
    }


def _spike_metrics(
    y: np.ndarray,
    clean: np.ndarray,
    meta: dict,
) -> dict:
    y, clean = _as_float(y), _as_float(clean)
    window = int(meta.get("spike_window", 25))
    idx = list(meta.get("spike_idx", []))
    leakage = 0.0
    mask = np.ones(y.size, dtype=bool)
    for i in idx:
        lo = max(0, i - window)
        hi = min(y.size, i + window + 1)
        leakage = max(leakage, float(np.max(np.abs(y[lo:hi] - clean[lo:hi]))))
        mask[lo:hi] = False
    clean_rmse = rmse(y[mask], clean[mask]) if np.any(mask) else float("nan")
    return {
        "spike_leakage": leakage,
        "clean_region_rmse": clean_rmse,
        "maxabs": float(np.max(np.abs(y))),
    }


def evaluate(
    observation,
    y: np.ndarray,
    fs: float = FS_DEFAULT,
    trim: int = 50,
) -> dict:
    """
    Compute generic + feature metrics for one observation and one denoised signal.

    ``observation`` is an :class:`explore.denoise.signals.Observation`.
    """
    clean = _as_float(observation.clean)
    noisy = _as_float(observation.noisy)
    y = _as_float(y)
    meta = observation.meta
    kind = meta.get("kind", "generic")
    out: dict = {}

    if kind == "null":
        out.update(
            {
                "output_rms": rms(y),
                "output_rms_ratio": rms(y) / rms(noisy) if rms(noisy) > 0 else float("nan"),
                "maxabs": float(np.max(np.abs(y))),
                "spectral_flatness": spectral_flatness(y),
                "edge_max": float(
                    max(np.max(np.abs(y[:trim])), np.max(np.abs(y[-trim:])))
                )
                if y.size > 2 * trim
                else float(np.max(np.abs(y))),
            }
        )
        return out

    y_t, clean_t, noisy_t = _trim(y, trim), _trim(clean, trim), _trim(noisy, trim)
    snr_in = snr_db(clean_t, noisy_t)
    snr_out = snr_db(clean_t, y_t)
    bw = clean_bandwidth(clean_t, fs)
    total_energy = float(np.dot(y_t - y_t.mean(), y_t - y_t.mean()))
    y_lp = _lowpass_fft(y_t, fc=bw, fs=fs) if np.isfinite(bw) and bw > 0 else y_t
    clean_lp = _lowpass_fft(clean_t, fc=bw, fs=fs) if np.isfinite(bw) and bw > 0 else clean_t

    out.update(
        {
            "rmse": rmse(y, clean),
            "rmse_int": rmse(y_t, clean_t),
            "snr_in": snr_in,
            "snr_out": snr_out,
            "snr_imp": snr_out - snr_in if np.isfinite(snr_in) and np.isfinite(snr_out) else float("nan"),
            "corr": correlation(y_t, clean_t),
            "shape_nrmse": best_scale_nrmse(clean_t, y_t),
            "shape_lp_corr": correlation(y_lp, clean_lp),
            "shape_lp_nrmse": best_scale_nrmse(clean_lp, y_lp),
            "maxabs_err": float(np.max(np.abs(y - clean))),
            "edge_rmse": rmse(y[:trim], clean[:trim])
            if y.size > 2 * trim
            else float("nan"),
            "edge_max": float(max(np.max(np.abs(y[:trim])), np.max(np.abs(y[-trim:]))))
            if y.size > 2 * trim
            else float(np.max(np.abs(y))),
            "clean_bw": bw,
            "hf_ratio": float(
                band_energy(y_t, bw, fs / 2.0, fs) / total_energy
            )
            if np.isfinite(bw) and total_energy > 0
            else float("nan"),
            "lsd": log_spectral_distance(clean_t, y_t),
        }
    )

    for band in meta.get("bands", []):
        f1, f2 = float(band[0]), float(band[1])
        out[f"band_{f1:g}_{f2:g}_retention"] = band_retention(y_t, clean_t, f1, f2, fs)
        out[f"band_{f1:g}_{f2:g}_fraction"] = band_fraction(y_t, f1, f2, fs)

    if kind == "trend":
        trend = _trim(observation.components["trend"], trim)
        out["trend_rmse"] = rmse(trend, y_t)
        out["trend_corr"] = correlation(y_t, trend)
        trend_lp = _lowpass_fft(y_t, fc=2.0, fs=fs)
        out["trend_lp_rmse"] = rmse(trend, trend_lp)
        out["trend_lp_corr"] = correlation(trend_lp, trend)
        weak = meta.get("weak_band", (150.0, 170.0))
        out["weak_retention"] = band_retention(y_t, clean_t, weak[0], weak[1], fs)
    elif kind == "impact":
        out.update(_impact_metrics(y, clean, meta, fs, trim))
    elif kind == "step":
        out.update(_step_metrics(y, clean, meta.get("step_times", []), fs))
    elif kind == "spike":
        out.update(_spike_metrics(y, clean, meta))

    return out
