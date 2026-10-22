"""
Generate the figures used by ``docs/explore/denoise/Denoise_Performance_Report.md``.

Usage:
    python -m explore.denoise.make_figures

The waveform and envelope-spectrum figures recompute a *fixed* observation
(the same seed convention as ``run_benchmark.py``) so they can plot signals,
not just the metrics stored in JSON.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for path in (str(ROOT), str(ROOT / "src")):
    if path not in sys.path:
        sys.path.insert(0, path)

from explore.denoise.methods import (  # noqa: E402
    classical_methods,
    eemd_corr,
    run_proposed,
)
from explore.denoise.metrics import _bandpass_envelope  # noqa: E402
from explore.denoise.signals import FS, make_observation  # noqa: E402

RESULTS = ROOT / "explore/denoise/results"
FIG_DIR = ROOT / "fig/explore/denoise"
BASE_SEED = 20261017
LEVELS = (0.1, 0.3, 0.6)


def _load() -> list[dict]:
    with (RESULTS / "benchmark.json").open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _median(values):
    vals = [v for v in values if v is not None]
    return float(np.median(vals)) if vals else float("nan")


def _metric(records, case, level, method, metric):
    vals = [
        r["metrics"].get(metric)
        for r in records
        if r.get("case") == case
        and abs(float(r.get("noise_level", -1)) - level) < 1e-12
        and r.get("method") == method
        and "metrics" in r
    ]
    return _median(vals)


def _observation(case: str, level: float):
    case_i = list(
        [
            "two_tone",
            "trend_tone",
            "am_fm",
            "impact",
            "step",
            "spike_tone",
            "pure_noise",
        ]
    ).index(case)
    level_i = list(LEVELS).index(level) if level in LEVELS else 1
    seed = BASE_SEED + case_i * 1_000 + level_i * 10
    return make_observation(case, level, seed=seed)


def figure_waveforms() -> None:
    cases = ["two_tone", "trend_tone", "impact", "step"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=False)
    for ax, case in zip(axes.ravel(), cases):
        obs = _observation(case, 0.3)
        t = np.arange(obs.clean.size) / FS
        p = run_proposed(obs.noisy, {"K": 2, "M": 32, "noise_amp": 0.2}, seed=BASE_SEED + 7_000)
        butter = classical_methods(obs.noisy, fs=FS)["Butter150"]
        ax.plot(t, obs.clean, color="black", lw=1.2, label="clean")
        ax.plot(t, obs.noisy, color="0.65", lw=0.7, alpha=0.8, label="noisy")
        ax.plot(t, p, color="C3", lw=1.0, label="P-K2-M32")
        ax.plot(t, butter, color="C0", lw=1.0, label="Butter150")
        residual = obs.noisy - obs.clean
        snr_in = 10.0 * np.log10(np.dot(obs.clean, obs.clean) / np.dot(residual, residual))
        ax.set_title(f"{case}  (noise=0.3, input SNR={snr_in:.1f} dB)")
        ax.set_xlabel("time (s)")
        ax.set_ylabel("amplitude")
        ax.grid(alpha=0.2)
        if case == cases[0]:
            ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "example_waveforms.png", dpi=150)
    plt.close(fig)


def figure_snr_vs_noise(records: list[dict]) -> None:
    cases = ["two_tone", "trend_tone", "impact", "step"]
    methods = [
        ("SG11", "C0", "-"),
        ("Butter150", "C2", "-"),
        ("Wiener11", "C1", "--"),
        ("P-K2-M32", "C3", "-"),
        ("IdealLP", "0.3", ":"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True, sharey=False)
    for ax, case in zip(axes.ravel(), cases):
        for method, color, style in methods:
            y = [_metric(records, case, level, method, "snr_imp") for level in LEVELS]
            ax.plot(LEVELS, y, color=color, linestyle=style, marker="o", ms=3, label=method)
        ax.axhline(0.0, color="black", lw=0.6)
        ax.set_title(case)
        ax.set_xlabel("relative noise level")
        ax.set_ylabel("SNR improvement (dB)")
        ax.grid(alpha=0.2)
        if case == cases[0]:
            ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "snr_vs_noise.png", dpi=150)
    plt.close(fig)


def figure_km(records: list[dict]) -> None:
    cases = ["trend_tone", "impact"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=False)
    for ax, case in zip(axes, cases):
        for M, color, style in [(1, "C1", "--"), (32, "C3", "-")]:
            y = [
                _metric(records, case, 0.3, f"P-K{k}-M{M}", "snr_imp")
                for k in (1, 2, 3)
            ]
            ax.plot([1, 2, 3], y, color=color, linestyle=style, marker="o", label=f"M={M}")
        ax.axhline(0.0, color="black", lw=0.6)
        ax.set_title(f"{case} (noise=0.3)")
        ax.set_xlabel("iterations K")
        ax.set_ylabel("SNR improvement (dB)")
        ax.set_xticks([1, 2, 3])
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "km_sensitivity.png", dpi=150)
    plt.close(fig)


def figure_impact_envelope(records: list[dict]) -> None:
    obs = _observation("impact", 0.3)
    outputs = {
        "clean": obs.clean,
        "noisy": obs.noisy,
        "P-K2-M32": run_proposed(obs.noisy, {"K": 2, "M": 32, "noise_amp": 0.2}, seed=BASE_SEED + 7_000),
        "Butter150": classical_methods(obs.noisy, fs=FS)["Butter150"],
        "Wiener11": classical_methods(obs.noisy, fs=FS)["Wiener11"],
        "EEMD-corr": eemd_corr(obs.noisy, seed=BASE_SEED + 11_000, M=10),
    }
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name, y in outputs.items():
        env = _bandpass_envelope(y, 120.0, 240.0, FS)
        env = env - env.mean()
        spec = np.abs(np.fft.rfft(env))
        freqs = np.fft.rfftfreq(env.size, 1.0 / FS)
        mask = freqs <= 30.0
        spec = spec[mask]
        spec = spec / max(float(spec.max()), 1e-30)
        ax.plot(freqs[mask], spec, lw=1.2 if name == "clean" else 1.0, label=name)
    ax.axvline(12.0, color="black", lw=0.6, linestyle=":")
    ax.set_xlim(0, 30)
    ax.set_xlabel("envelope frequency (Hz)")
    ax.set_ylabel("normalized envelope amplitude")
    ax.set_title("impact case: envelope spectrum (noise=0.3, 12 Hz repetition)")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "impact_envelope_spectrum.png", dpi=150)
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    records = _load()
    figure_waveforms()
    figure_snr_vs_noise(records)
    figure_km(records)
    figure_impact_envelope(records)
    print(f"wrote figures -> {FIG_DIR}")


if __name__ == "__main__":
    main()
