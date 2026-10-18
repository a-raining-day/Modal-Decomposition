"""
Figures for the high-noise main-morphology comparison.

Usage:
    python -m explore.denoise.make_high_noise_figures
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

from explore.denoise.methods import run_proposed  # noqa: E402
from explore.denoise.morphology import morph_avg  # noqa: E402
from explore.denoise.signals import FS, make_observation  # noqa: E402

RESULTS = ROOT / "explore/denoise/results"
FIG_DIR = ROOT / "docs/explore/denoise/figures"
BASE_SEED = 20261017
LEVEL = 0.7


def _load_summary() -> tuple[list[dict], list[dict]]:
    with (RESULTS / "high_noise_summary.json").open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    return data["summary"], data["ranking"]


def _row(summary, case, level, method):
    for row in summary:
        if (
            row["case"] == case
            and abs(float(row["noise_level"]) - level) < 1e-12
            and row["method"] == method
        ):
            return row
    return None


def _family(method: str) -> str:
    if method.startswith("P-"):
        return "proposed"
    if method.startswith("Morph"):
        return "morph"
    if method.startswith("SG"):
        return "SG"
    if method.startswith("Butter"):
        return "Butter"
    return "other"


def figure_rank(ranking: list[dict]) -> None:
    rows = [r for r in ranking if r["mean_rank"] is not None][:12]
    labels = [r["method"] for r in rows]
    means = [r["mean_rank"] for r in rows]
    stds = [r["rank_std"] for r in rows]
    colors = {
        "proposed": "C3",
        "morph": "C2",
        "SG": "C0",
        "Butter": "C1",
        "other": "0.5",
    }
    fig, ax = plt.subplots(figsize=(8, 5))
    y = np.arange(len(rows))
    ax.barh(y, means, xerr=stds, color=[colors[_family(r["method"])] for r in rows], alpha=0.85)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("mean rank by shape NRMSE (lower is better); error bar = rank std")
    ax.set_title("Cross-shape consistency at high noise (0.5/0.7/0.9 x std)")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "high_noise_rank.png", dpi=150)
    plt.close(fig)


def figure_heatmap(summary: list[dict]) -> None:
    methods = [
        "raw",
        "MorphAvg5",
        "MorphAvg11",
        "MorphAvg21",
        "SG11",
        "SG21",
        "Butter100",
        "Butter150",
        "Butter250",
        "P-K1-M32",
        "P-K2-M32",
        "P-K2-M64",
        "IdealLP",
    ]
    cases = ["two_tone", "trend_tone", "am_fm", "impact", "step", "spike_tone"]
    data = np.full((len(methods), len(cases)), np.nan)
    for i, method in enumerate(methods):
        for j, case in enumerate(cases):
            row = _row(summary, case, LEVEL, method)
            if row and row["shape_nrmse"] is not None:
                data[i, j] = row["shape_nrmse"]
    fig, ax = plt.subplots(figsize=(9, 6))
    im = ax.imshow(data, aspect="auto", cmap="viridis_r")
    ax.set_xticks(np.arange(len(cases)))
    ax.set_xticklabels(cases, rotation=25, ha="right")
    ax.set_yticks(np.arange(len(methods)))
    ax.set_yticklabels(methods)
    ax.set_title("Shape NRMSE at noise=0.7 (lower is better)")
    for i in range(len(methods)):
        for j in range(len(cases)):
            if np.isfinite(data[i, j]):
                ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center",
                        color="white" if data[i, j] > 0.55 else "black", fontsize=7)
    fig.colorbar(im, ax=ax, label="shape NRMSE")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "high_noise_shape_heatmap.png", dpi=150)
    plt.close(fig)


def _obs(case: str, level: float):
    cases = ["two_tone", "trend_tone", "am_fm", "impact", "step", "spike_tone", "pure_noise"]
    case_i = cases.index(case)
    level_i = [0.5, 0.7, 0.9].index(level)
    seed = BASE_SEED + case_i * 1_000 + level_i * 10
    return make_observation(case, level, seed=seed)


def figure_examples() -> None:
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=False)
    specs = [
        ("trend_tone", "MorphAvg5"),
        ("step", "MorphAvg11"),
        ("impact", "Butter250"),
    ]
    for ax, (case, classical_name) in zip(axes, specs):
        obs = _obs(case, LEVEL)
        t = np.arange(obs.clean.size) / FS
        p = run_proposed(
            obs.noisy,
            {"K": 2, "M": 64, "noise_amp": 0.2},
            seed=BASE_SEED + 7_000,
        )
        if classical_name.startswith("MorphAvg"):
            size = int(classical_name.replace("MorphAvg", ""))
            c = morph_avg(obs.noisy, size)
        elif classical_name.startswith("Butter"):
            from scipy.signal import butter, filtfilt

            cutoff = int(classical_name.replace("Butter", ""))
            b, a = butter(4, cutoff / (FS / 2.0), btype="low")
            c = filtfilt(b, a, obs.noisy)
        else:
            raise ValueError(f"unsupported figure method {classical_name!r}")
        ax.plot(t, obs.clean, color="black", lw=1.3, label="clean")
        ax.plot(t, obs.noisy, color="0.65", lw=0.7, alpha=0.8, label="noisy")
        ax.plot(t, p, color="C3", lw=1.1, label="P-K2-M64")
        ax.plot(t, c, color="C2", lw=1.1, label=classical_name)
        ax.set_title(f"{case} (noise=0.7)")
        ax.set_ylabel("amplitude")
        ax.grid(alpha=0.2)
        if ax is axes[0]:
            ax.legend(loc="upper right", fontsize=8, ncol=2)
    axes[-1].set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "high_noise_examples.png", dpi=150)
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    summary, ranking = _load_summary()
    figure_rank(ranking)
    figure_heatmap(summary)
    figure_examples()
    print(f"wrote high-noise figures -> {FIG_DIR}")


if __name__ == "__main__":
    main()
