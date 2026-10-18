"""
2-D image denoising demo: morphological analogue of the proposed method.

Important distinction
----------------------
The 1-D method computes cubic-spline upper/lower envelopes.  A true 2-D
spline/BEMD envelope is a much harder problem (2-D extrema + surface
interpolation) and is NOT implemented here.  This demo uses the natural
flat-structuring-element analogue:

    E(x) = (grey_dilation(x, s) + grey_erosion(x, s)) / 2

i.e. a midrange / local morphological mean, followed by the same
add-noise + M-average + K-iteration loop.

Data: real ``grace_hopper.jpg`` shipped with matplotlib (grayscale, 256x256
center crop).  Noise levels are Gaussian sigma=20/40/60 on [0,255].

Usage:
    python -m explore.denoise.image_2d.run_image_demo
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image
from scipy.ndimage import (
    gaussian_filter,
    grey_dilation,
    grey_erosion,
    median_filter,
    uniform_filter,
)

ROOT = Path(__file__).resolve().parents[3]
FIG_DIR = ROOT / "docs/explore/denoise/figures"
RESULTS = ROOT / "explore/denoise/results"

SIGMAS = (20.0, 40.0, 60.0)
REPS = 5
SE_SIZE = 5


def load_image() -> np.ndarray:
    path = Path(matplotlib.get_data_path()) / "sample_data" / "grace_hopper.jpg"
    image = np.asarray(Image.open(path).convert("L"), dtype=np.float64)
    return image[128:384, 128:384]


def psnr(clean: np.ndarray, y: np.ndarray) -> float:
    mse = float(np.mean((clean - y) ** 2))
    return 10.0 * np.log10(255.0**2 / mse) if mse > 0 else float("inf")


def ssim(clean: np.ndarray, y: np.ndarray) -> float:
    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2
    mu_x = gaussian_filter(clean, 1.5)
    mu_y = gaussian_filter(y, 1.5)
    sx2 = gaussian_filter(clean * clean, 1.5) - mu_x * mu_x
    sy2 = gaussian_filter(y * y, 1.5) - mu_y * mu_y
    sxy = gaussian_filter(clean * y, 1.5) - mu_x * mu_y
    num = (2 * mu_x * mu_y + c1) * (2 * sxy + c2)
    den = (mu_x * mu_x + mu_y * mu_y + c1) * (sx2 + sy2 + c2)
    return float(np.mean(num / den))


def edge_corr(clean: np.ndarray, y: np.ndarray) -> float:
    def grad_mag(a):
        gx, gy = np.gradient(a)
        return np.hypot(gx, gy)
    a, b = grad_mag(clean).ravel(), grad_mag(y).ravel()
    if a.std() <= 0 or b.std() <= 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def midrange(x: np.ndarray, size: int = SE_SIZE) -> np.ndarray:
    upper = grey_dilation(x, size=size, mode="reflect")
    lower = grey_erosion(x, size=size, mode="reflect")
    return 0.5 * (upper + lower)


def proposed(
    x: np.ndarray,
    K: int,
    M: int,
    amp: float,
    size: int = SE_SIZE,
    seed: int = 0,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    current = np.array(x, dtype=np.float64, copy=True)
    for _ in range(K):
        acc = np.zeros_like(current)
        for _ in range(M):
            noisy = current + amp * current.std() * rng.standard_normal(current.shape)
            acc += midrange(noisy, size)
        current = acc / M
    return current


def mean_iter(
    x: np.ndarray,
    K: int,
    M: int,
    amp: float,
    size: int = SE_SIZE,
    seed: int = 0,
) -> np.ndarray:
    """Current algorithm: iterate on the mean surface; return the mean surface."""
    rng = np.random.default_rng(seed)
    current = np.array(x, dtype=np.float64, copy=True)
    for _ in range(K):
        acc = np.zeros_like(current)
        for _ in range(M):
            noisy = current + amp * current.std() * rng.standard_normal(current.shape)
            acc += midrange(noisy, size)
        current = acc / M
    return current


def each_subtract(
    x: np.ndarray,
    K: int,
    M: int,
    amp: float,
    size: int = SE_SIZE,
    seed: int = 0,
) -> np.ndarray:
    """Sifting variant: subtract the mean surface at every iteration."""
    rng = np.random.default_rng(seed)
    current = np.array(x, dtype=np.float64, copy=True)
    for _ in range(K):
        acc = np.zeros_like(current)
        for _ in range(M):
            noisy = current + amp * current.std() * rng.standard_normal(current.shape)
            acc += midrange(noisy, size)
        current = current - acc / M
    return current


def final_subtract(
    x: np.ndarray,
    K: int,
    M: int,
    amp: float,
    size: int = SE_SIZE,
    seed: int = 0,
) -> np.ndarray:
    """Subtract only the final mean surface: x - mean_iter(x)."""
    mean_surface = mean_iter(x, K, M, amp, size=size, seed=seed)
    return np.asarray(x, dtype=np.float64) - mean_surface


def soft_threshold(detail: np.ndarray, tau: float) -> np.ndarray:
    return np.sign(detail) * np.maximum(np.abs(detail) - tau, 0.0)


def methods(noisy: np.ndarray, seed: int, sigma: float) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {"raw": noisy}
    for sigma_g in (1.0, 2.0):
        out[f"Gauss{sigma_g:g}"] = gaussian_filter(noisy, sigma_g)
    for size in (3, 5):
        out[f"Median{size}"] = median_filter(noisy, size)
    out["Uniform5"] = uniform_filter(noisy, 5)

    # "subtract the mean surface": base + detail, then soft-shrink detail
    base = midrange(noisy)
    detail = noisy - base
    out["E-only"] = base
    out["detail-only"] = detail
    for factor in (0.5, 1.0, 1.5, 2.0):
        out[f"E+soft{factor:g}s"] = base + soft_threshold(detail, factor * sigma)

    for K, M, amp in ((1, 8, 0.5), (2, 8, 0.5), (2, 16, 0.6), (1, 16, 0.6)):
        out[f"P-K{K}-M{M}-a{amp}"] = proposed(noisy, K, M, amp, seed=seed + 7000)

    # K=2, M=16, amp=0.6: subtract each iteration vs only subtract at the end
    out["P-eachsub-K2-M16-a0.6"] = each_subtract(
        noisy, 2, 16, 0.6, seed=seed + 7000
    )
    out["P-finalsub-K2-M16-a0.6"] = final_subtract(
        noisy, 2, 16, 0.6, seed=seed + 7000
    )
    return out


def run() -> list[dict]:
    clean = load_image()
    clean_detail = clean - midrange(clean)
    records = []
    for sigma in SIGMAS:
        for rep in range(REPS):
            rng = np.random.default_rng(20261017 + int(sigma) * 100 + rep)
            noisy = clean + sigma * rng.standard_normal(clean.shape)
            for name, y in methods(
                noisy, seed=20261017 + int(sigma) * 100 + rep, sigma=sigma
            ).items():
                detail_psnr = None
                if "sub" in name:
                    detail_psnr = psnr(clean_detail, y)
                records.append(
                    {
                        "sigma": sigma,
                        "rep": rep,
                        "method": name,
                        "psnr": psnr(clean, y),
                        "psnr_detail": detail_psnr,
                        "ssim": ssim(clean, y),
                        "edge_corr": edge_corr(clean, y),
                        "std": float(y.std()),
                    }
                )
    return records


def summarize(records: list[dict]) -> list[dict]:
    groups = {}
    for record in records:
        groups.setdefault(record["method"], []).append(record)
    rows = []
    for method, group in groups.items():
        rows.append(
            {
                "method": method,
                "psnr": float(np.median([r["psnr"] for r in group])),
                "psnr_detail": float(
                    np.median([r["psnr_detail"] for r in group if r["psnr_detail"] is not None])
                )
                if any(r["psnr_detail"] is not None for r in group)
                else None,
                "ssim": float(np.median([r["ssim"] for r in group])),
                "edge_corr": float(np.median([r["edge_corr"] for r in group])),
            }
        )
    rows.sort(key=lambda r: -r["psnr"])
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, default=RESULTS)
    args = parser.parse_args()
    records = run()
    rows = summarize(records)
    args.outdir.mkdir(parents=True, exist_ok=True)
    with (args.outdir / "image_2d.json").open("w", encoding="utf-8") as fh:
        json.dump(records, fh, indent=2, ensure_ascii=False)
    with (args.outdir / "image_2d_summary.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["method", "psnr", "psnr_detail", "ssim", "edge_corr"]
        )
        writer.writeheader()
        writer.writerows(rows)
    print("| method | PSNR | detail PSNR | SSIM | edge corr |")
    print("|---|---:|---:|---:|---:|")
    for row in rows:
        detail = "-" if row["psnr_detail"] is None else f"{row['psnr_detail']:.3f}"
        print(
            f"| {row['method']} | {row['psnr']:.3f} | {detail} | "
            f"{row['ssim']:.4f} | {row['edge_corr']:.3f} |"
        )
    print(f"wrote image_2d.json/csv -> {args.outdir}")


if __name__ == "__main__":
    main()
