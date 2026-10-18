"""Figure for the 2-D image demo.

Usage:
    python -m explore.denoise.image_2d.make_figure
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter, median_filter

ROOT = Path(__file__).resolve().parents[3]
for path in (str(ROOT), str(ROOT / "src")):
    if path not in sys.path:
        sys.path.insert(0, path)

from explore.denoise.image_2d.run_image_demo import (  # noqa: E402
    each_subtract,
    final_subtract,
    load_image,
    midrange,
    proposed,
)

FIG_DIR = ROOT / "docs/explore/denoise/figures"


def main() -> None:
    clean = load_image()
    sigma = 40.0
    rng = np.random.default_rng(20261017 + int(sigma) * 100)
    noisy = clean + sigma * rng.standard_normal(clean.shape)
    gauss = gaussian_filter(noisy, 1.0)
    med = median_filter(noisy, 5)
    p2 = proposed(noisy, K=2, M=16, amp=0.6, seed=20261017 + int(sigma) * 100 + 7000)
    panels = [
        ("clean", clean),
        (f"noisy σ={sigma:g}", noisy),
        ("Gauss1", gauss),
        ("Median5", med),
        ("P-K2-M16-a0.6", p2),
        ("P residual", p2 - clean),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(11, 7))
    for ax, (title, image) in zip(axes.ravel(), panels):
        if title.endswith("residual"):
            im = ax.imshow(image, cmap="coolwarm", vmin=-60, vmax=60)
        else:
            im = ax.imshow(image, cmap="gray", vmin=0, vmax=255)
        ax.set_title(title)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    fig.suptitle("2-D morphological analogue: real image + Gaussian noise σ=40")
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / "image_2d_example.png", dpi=150)
    plt.close(fig)

    mean = proposed(noisy, K=2, M=16, amp=0.6, seed=20261017 + int(sigma) * 100 + 7000)
    final = final_subtract(noisy, K=2, M=16, amp=0.6, seed=20261017 + int(sigma) * 100 + 7000)
    each = each_subtract(noisy, K=2, M=16, amp=0.6, seed=20261017 + int(sigma) * 100 + 7000)

    fig, axes = plt.subplots(2, 2, figsize=(9, 8))
    items = [
        ("mean surface S_K", mean, "gray", 0, 255),
        ("final subtract: noisy - S_K", final, "coolwarm", -60, 60),
        ("each-iteration subtract", each, "coolwarm", -60, 60),
        ("clean detail: clean - E(clean)", clean - midrange(clean), "coolwarm", -60, 60),
    ]
    for ax, (title, image, cmap, vmin, vmax) in zip(axes.ravel(), items):
        im = ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    fig.suptitle("Mean surface vs subtract-each vs subtract-final (noise σ=40)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "image_2d_subtract.png", dpi=150)
    plt.close(fig)
    print(f"wrote image_2d_example.png, image_2d_subtract.png -> {FIG_DIR}")


if __name__ == "__main__":
    main()
