"""
Synthetic denoising benchmark signals.

All signals are 1-D float64 at ``FS = 1000 Hz``.  A case is a *clean* signal
plus the metadata needed by the feature metrics (known bands, impact times,
step times, spike locations).  Observations add white Gaussian noise at a
requested fraction of the clean signal's standard deviation.

The cases deliberately span different notions of "denoising":

* ``two_tone``      - stationary narrow-band tones; classical filters are strong.
* ``trend_tone``    - slow trend + strong tone + weak modulated tone; tests
                      trend extraction and weak-feature retention.
* ``am_fm``         - non-stationary AM-FM component + tone.
* ``impact``        - periodic decaying resonance bursts; tests impact /
                      envelope preservation (bearing-fault-like).
* ``step``          - two smoothed steps; tests edge preservation.
* ``spike_tone``    - two tones plus three large deterministic spikes; tests
                      impulse-noise robustness.
* ``pure_noise``    - no signal; tests whether the method invents structure.

The module is deterministic for a given ``(case, noise_level, seed)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

FS = 1000.0
N_DEFAULT = 2048


@dataclass(frozen=True)
class Observation:
    """One noisy observation plus its clean reference and metadata."""

    name: str
    clean: np.ndarray
    noisy: np.ndarray
    components: dict[str, np.ndarray]
    meta: dict


def _t(n: int) -> np.ndarray:
    return np.arange(n, dtype=np.float64) / FS


def _two_tone(n: int) -> tuple[np.ndarray, dict, dict]:
    t = _t(n)
    tone_37 = np.sin(2 * np.pi * 37.0 * t)
    tone_113 = 0.5 * np.sin(2 * np.pi * 113.0 * t)
    clean = tone_37 + tone_113
    meta = {
        "kind": "stationary",
        "bands": [(30.0, 45.0), (105.0, 120.0)],
        "freqs": [37.0, 113.0],
    }
    return clean, {"tone_37": tone_37, "tone_113": tone_113}, meta


def _trend_tone(n: int) -> tuple[np.ndarray, dict, dict]:
    t = _t(n)
    duration = n / FS
    trend = 0.5 * (t / duration) ** 2
    strong = np.sin(2 * np.pi * 40.0 * t)
    weak = 0.12 * np.sin(2 * np.pi * 160.0 * t) * (
        0.5 + 0.5 * np.cos(2 * np.pi * 8.0 * t)
    )
    clean = trend + strong + weak
    meta = {
        "kind": "trend",
        "bands": [(0.0, 2.0), (35.0, 45.0), (150.0, 170.0)],
        "weak_band": (150.0, 170.0),
        "weak_freq": 160.0,
        "mod_freq": 8.0,
    }
    return clean, {"trend": trend, "strong_40": strong, "weak_160": weak}, meta


def _am_fm(n: int) -> tuple[np.ndarray, dict, dict]:
    t = _t(n)
    amfm = (1.0 + 0.3 * np.sin(2 * np.pi * 2.0 * t)) * np.sin(
        2 * np.pi * 24.0 * t + 5.0 * np.sin(2 * np.pi * 0.8 * t)
    )
    tone_89 = 0.6 * np.sin(2 * np.pi * 89.0 * t)
    clean = amfm + tone_89
    meta = {
        "kind": "nonstationary",
        "bands": [(15.0, 35.0), (80.0, 100.0)],
        "freqs": [24.0, 89.0],
    }
    return clean, {"amfm": amfm, "tone_89": tone_89}, meta


def _impact(n: int) -> tuple[np.ndarray, dict, dict]:
    t = _t(n)
    clean = np.zeros(n, dtype=np.float64)
    period = int(round(FS / 12.0))
    impact_idx = list(range(period, n, period))
    for k in impact_idx:
        tt = t - t[k]
        mask = (tt >= 0.0) & (tt < 0.04)
        clean[mask] += np.exp(-tt[mask] / 0.008) * np.sin(
            2 * np.pi * 180.0 * tt[mask]
        )
    shaft = 0.2 * np.sin(2 * np.pi * 5.0 * t)
    clean += shaft
    meta = {
        "kind": "impact",
        "bands": [(3.0, 7.0), (120.0, 240.0)],
        "rep_freq": 12.0,
        "resonance_freq": 180.0,
        "impact_idx": impact_idx,
        "env_band": (120.0, 240.0),
    }
    return clean, {"impact": clean - shaft, "shaft_5": shaft}, meta


def _step(n: int) -> tuple[np.ndarray, dict, dict]:
    t = _t(n)
    step_a = 0.5 * np.tanh((t - 0.70) / 0.005)
    step_b = 0.5 * np.tanh((t - 1.35) / 0.005)
    clean = step_a + step_b - 0.5
    meta = {
        "kind": "step",
        "bands": [(0.0, 2.0)],
        "step_times": [0.70, 1.35],
        "step_heights": [0.5, 0.5],
    }
    return clean, {"step_a": step_a, "step_b": step_b}, meta


def _spike_tone(n: int) -> tuple[np.ndarray, dict, dict]:
    t = _t(n)
    tone_20 = 0.8 * np.sin(2 * np.pi * 20.0 * t)
    tone_80 = 0.3 * np.sin(2 * np.pi * 80.0 * t)
    clean = tone_20 + tone_80
    all_idx = [410, 1024, 1630]
    all_amp = [4.0, -4.0, 3.5]
    spike_idx = [i for i in all_idx if i < n]
    spike_amp = [a for i, a in zip(all_idx, all_amp) if i < n]
    meta = {
        "kind": "spike",
        "bands": [(15.0, 25.0), (75.0, 85.0)],
        "spike_idx": spike_idx,
        "spike_amp": spike_amp,
        "spike_window": 25,
    }
    return clean, {"tone_20": tone_20, "tone_80": tone_80}, meta


def _pure_noise(n: int) -> tuple[np.ndarray, dict, dict]:
    return np.zeros(n, dtype=np.float64), {}, {"kind": "null"}


_BUILDERS: dict[str, tuple[Callable[[int], tuple[np.ndarray, dict, dict]], str]] = {
    "two_tone": (_two_tone, "two stationary tones (37, 113 Hz)"),
    "trend_tone": (_trend_tone, "quadratic trend + 40 Hz strong + weak 160 Hz AM"),
    "am_fm": (_am_fm, "AM-FM chirp at 24 Hz + 89 Hz tone"),
    "impact": (_impact, "12 Hz impacts, 180 Hz resonance, 5 Hz shaft"),
    "step": (_step, "two smoothed steps (0.70 s, 1.35 s)"),
    "spike_tone": (_spike_tone, "20/80 Hz tones + three large spikes"),
    "pure_noise": (_pure_noise, "white noise only (null case)"),
}

CASE_ORDER = list(_BUILDERS)
CASE_DESCRIPTIONS = {name: desc for name, (_, desc) in _BUILDERS.items()}


def build_case(name: str, n: int = N_DEFAULT) -> tuple[np.ndarray, dict, dict]:
    """Return ``(clean, components, meta)`` for a case name."""
    if name not in _BUILDERS:
        raise ValueError(f"unknown case {name!r}; use one of {CASE_ORDER}")
    return _BUILDERS[name][0](n)


def make_observation(
    name: str,
    noise_level: float,
    seed: int,
    n: int = N_DEFAULT,
) -> Observation:
    """
    Build one observation.

    For ordinary cases ``noise_level`` is the noise std divided by the clean
    signal's std; for ``pure_noise`` it is the absolute noise std.
    """
    clean, components, meta = build_case(name, n=n)
    rng = np.random.default_rng(seed)

    if name == "pure_noise":
        noisy = noise_level * rng.standard_normal(n)
    else:
        noisy = clean + noise_level * float(clean.std()) * rng.standard_normal(n)
        if name == "spike_tone":
            for i, amp in zip(meta["spike_idx"], meta["spike_amp"]):
                noisy[i] += amp

    return Observation(
        name=name,
        clean=clean,
        noisy=noisy,
        components=components,
        meta=meta,
    )
