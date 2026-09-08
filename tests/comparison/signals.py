"""
Deterministic benchmark signals for the EMD comparison suite.

All signals are 1-D ``float64`` sampled at ``FS = 1000 Hz``; a length ``n``
signal covers ``n/FS`` seconds. Noise is drawn from a pre-generated stream so
that different lengths share one identical prefix (like real captures), and
every decomposition implementation receives the exact same array object.

Each case builder returns ``(S, modes)``:

* ``S``     - the signal that is passed to the implementations.
* ``modes`` - ground-truth components worth recovering (used only by the
  quality metrics, never by the EMD runs).
"""

from __future__ import annotations

import numpy as np

FS = 1000.0
_SEED = 20260906          # fixed seed (repo-era stamp: 2026-09-06)
_MAX_STREAM_N = 1 << 18   # 262144 -- noise stream prefix length


def _noise_stream(seed: int) -> np.ndarray:
    """Deterministic standard-normal stream shared by all lengths."""
    return np.random.default_rng(seed).standard_normal(_MAX_STREAM_N)


def _noise(seed: int, n: int) -> np.ndarray:
    if n <= _MAX_STREAM_N:
        return _noise_stream(seed)[:n]
    # longer than the stream: fall back to a length-seeded draw (still
    # deterministic per (seed, n)).
    return np.random.default_rng(seed + n).standard_normal(n)


def case_A(n: int):
    """
    Two pure tones (37 Hz, 113 Hz) plus weak noise (std 0.1).

    Classic EMD target: the two tones should be separated into distinct IMFs.
    """
    t = np.arange(n) / FS
    tone1 = np.sin(2 * np.pi * 37.0 * t)
    tone2 = 0.5 * np.sin(2 * np.pi * 113.0 * t)
    noise = 0.1 * _noise(20260906, n)
    S = tone1 + tone2 + noise
    return S, {"tone_37hz": tone1, "tone_113hz": tone2}


def case_B(n: int):
    """
    AM-FM chirp + pure tone + quadratic trend (noise-free).

    Checks quality on non-stationary structure: amplitude/frequency modulated
    mode, one stationary tone and a slow trend.
    """
    t = np.arange(n) / FS
    amfm = (1.0 + 0.3 * np.sin(2 * np.pi * 2.0 * t)) * np.sin(
        2 * np.pi * 24.0 * t + 5.0 * np.sin(2 * np.pi * 0.8 * t)
    )
    tone = 0.6 * np.sin(2 * np.pi * 89.0 * t)
    dur = t[-1] if n > 1 else 1.0
    trend = 0.5 * (t / dur) ** 2
    S = amfm + tone + trend
    return S, {"amfm": amfm, "tone_89hz": tone, "trend": trend}


def case_C(n: int):
    """
    Pure white noise (sifting stress case): extrema density ~ n/2, so it
    exercises the per-sift extrema scan and spline envelope path hardest.
    """
    S = _noise(771, n)
    return S, {}


_CASES = {
    "A": (case_A, "two tones + light noise"),
    "B": (case_B, "AM-FM chirp + tone + trend"),
    "C": (case_C, "white noise (sifting stress)"),
}


def build_case(case: str, n: int):
    """Return ``(S, modes)`` for a case name and sample count."""
    if case not in _CASES:
        raise ValueError(f"unknown case {case!r}; use one of {sorted(_CASES)}")
    return _CASES[case][0](n)


def case_desc(case: str) -> str:
    return _CASES[case][1]


def default_ns() -> list[int]:
    return [256, 1024, 4096, 16384, 65536]
