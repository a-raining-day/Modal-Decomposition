"""
Denoising methods compared in the benchmark.

Classical baselines use fixed, representative settings.  ``IdealLP`` is an
oracle (it uses the clean spectrum to choose the cutoff) and is reported only
as a reference ceiling.  ``EEMD-corr`` is a correlation-thresholded EEMD
reconstruction, included as a decomposition-based reference.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.ndimage import median_filter, uniform_filter1d
from scipy.signal import butter, filtfilt, savgol_filter, wiener

from Modal_Decomposition.EEMD import EEMD
from Modal_Decomposition.NoisyEnvelopeIteration import NoisyEnvelopeMeanIteration

from .metrics import clean_bandwidth

FS_DEFAULT = 1000.0


def classical_method_fns(fs: float = FS_DEFAULT) -> dict[str, Callable]:
    """Fixed classical filters, keyed by display name."""
    b, a = butter(4, 150.0 / (fs / 2.0), btype="low")
    return {
        "raw": lambda x: np.array(x, dtype=np.float64, copy=True),
        "MA5": lambda x: uniform_filter1d(x, size=5, mode="reflect"),
        "SG11": lambda x: savgol_filter(x, window_length=11, polyorder=3, mode="interp"),
        "Median5": lambda x: median_filter(x, size=5, mode="reflect"),
        "Butter150": lambda x: filtfilt(b, a, x),
        "Wiener11": lambda x: wiener(x, mysize=11),
    }


def classical_methods(x: np.ndarray, fs: float = FS_DEFAULT) -> dict[str, np.ndarray]:
    """Apply all fixed classical filters to ``x``."""
    x = np.asarray(x, dtype=np.float64)
    return {name: fn(x) for name, fn in classical_method_fns(fs).items()}


def ideal_lowpass(
    x: np.ndarray, clean: np.ndarray, fs: float = FS_DEFAULT, frac: float = 0.99
) -> np.ndarray:
    """Oracle zero-phase low-pass at the 99% clean-energy bandwidth."""
    x = np.asarray(x, dtype=np.float64)
    bw = clean_bandwidth(clean, fs=fs, frac=frac)
    if not np.isfinite(bw) or bw <= 0.0 or x.size < 2:
        return x.copy()
    spec = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(x.size, 1.0 / fs)
    spec[freqs > bw] = 0.0
    return np.fft.irfft(spec, n=x.size)


def eemd_corr(
    x: np.ndarray,
    seed: int = 0,
    M: int = 10,
    noise_amp: float = 0.2,
    corr_thr: float = 0.2,
) -> np.ndarray:
    """EEMD reconstruction keeping IMFs correlated with the noisy input."""
    x = np.asarray(x, dtype=np.float64)
    np.random.seed(seed)
    result = EEMD(num_imf=-1, noise_amp=noise_amp, M=M, parallel=False).decompose(x)
    components = [row for row in np.atleast_2d(result.IMFs)]
    if result.Res is not None:
        components.append(np.asarray(result.Res, dtype=np.float64))

    keep = []
    for component in components:
        if component.size != x.size or np.std(component) <= 0.0:
            continue
        if abs(float(np.corrcoef(component, x)[0, 1])) >= corr_thr:
            keep.append(component)
    if not keep:
        return np.zeros_like(x)
    return np.sum(np.vstack(keep), axis=0)


def proposed_configs(noise_amp: float = 0.2) -> list[dict]:
    """Main proposed-method grid: K in {1,2,3} x M in {1,32}."""
    return [
        {"name": f"P-K{k}-M{m}", "K": k, "M": m, "noise_amp": noise_amp}
        for k in (1, 2, 3)
        for m in (1, 32)
    ]


def proposed_sensitivity_configs(noise_amp_values=(0.05, 0.1, 0.3, 0.5)) -> list[dict]:
    """Auxiliary-noise sensitivity at fixed K=2, M=32 (default 0.2 is in the main grid)."""
    return [
        {"name": f"P-K2-M32-a{a:g}", "K": 2, "M": 32, "noise_amp": a}
        for a in noise_amp_values
    ]


def run_proposed(
    x: np.ndarray, config: dict, seed: int
) -> np.ndarray:
    np.random.seed(seed)
    return NoisyEnvelopeMeanIteration(
        np.array(x, dtype=np.float64, copy=True),
        config["K"],
        noise_amp=config["noise_amp"],
        M=config["M"],
    )
