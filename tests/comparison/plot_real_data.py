"""真实数据分解图对比：CWRU 轴承故障 + ECG。

每个信号一张"方法 × (输入+各阶 IMF)"的叠置图，外加一张"最冲击模态的包络谱"对比
（轴承诊断的常用视图）。输出到 ``fig/methods/real_data/``；若该目录不可写则退回
``table/methods/real_data/``。
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.signal import hilbert  # noqa: E402
from scipy.stats import kurtosis  # noqa: E402

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]
for p in (str(_ROOT / "src"), str(_ROOT), str(_HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from bench_real_data import METHODS, load_signals  # noqa: E402

FIG = _ROOT / "fig" / "methods" / "real_data"
FALLBACK = _ROOT / "table" / "methods" / "real_data"

PANELS = {
    "cwru4k": ["MD-EMD", "MD-EFD", "MD-EWT", "MD-VMD", "MD-FMD-ck", "PySDKit-FMD"],
    "ecg4k": ["MD-EMD", "MD-EFD", "MD-FMD-ck", "PySDKit-FMD"],
}


def save(fig, name: str) -> Path:
    for base in (FIG, FALLBACK):
        try:
            base.mkdir(parents=True, exist_ok=True)
            path = base / name
            fig.savefig(path, dpi=140)
            return path
        except OSError:
            continue
    raise SystemExit(f"cannot write {name}")


def decomposition_figure(sig_name: str, S: np.ndarray, names: list[str]) -> Path:
    results = {}
    for name in names:
        t0 = time.perf_counter()
        try:
            rows = np.atleast_2d(np.asarray(METHODS[name](S), dtype=float))
        except Exception as exc:                       # noqa: BLE001
            print(f"  skip {name}: {type(exc).__name__}")
            continue
        results[name] = (rows, (time.perf_counter() - t0) * 1e3)

    n = len(results)
    fig, axes = plt.subplots(n, 1, figsize=(11, 1.55 * n + 0.6), sharex=True)
    axes = np.atleast_1d(axes)
    t = np.arange(S.size)
    for ax, (name, (rows, wall)) in zip(axes, results.items()):
        stack = [S] + list(rows)
        labels = ["S"] + [f"m{i}" for i in range(rows.shape[0])]
        span = max(np.max(np.abs(s)) for s in stack) or 1.0
        for k, s in enumerate(stack):
            off = -k * 2.1 * span
            ax.plot(t, s / span + off, lw=0.6 if k else 0.9,
                    color="black" if k == 0 else f"C{(k - 1) % 10}")
            ax.text(0.004, off / 1.0 + 0.12, labels[k], fontsize=6, va="bottom",
                    transform=ax.get_yaxis_transform())
        km = max(kurtosis(r) for r in rows) if rows.size else float("nan")
        ax.set_ylabel(f"{name}\n{wall:.1f} ms\nkurt*={km:.0f}", fontsize=6.5)
        ax.set_yticks([])
        ax.tick_params(labelsize=6)
    axes[-1].set_xlabel("sample", fontsize=7)
    fig.suptitle(f"{sig_name}: decomposition comparison (input + modes, offset-normalised)",
                 fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    path = save(fig, f"{sig_name}_decompositions.png")
    plt.close(fig)
    return path


def envelope_figure(sig_name: str, S: np.ndarray, names: list[str], fs: float) -> Path:
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 5.0))
    axes = axes.ravel()
    for ax, name in zip(axes, names):
        try:
            rows = np.atleast_2d(np.asarray(METHODS[name](S), dtype=float))
        except Exception:                              # noqa: BLE001
            ax.set_visible(False)
            continue
        kurt = np.array([kurtosis(r) for r in rows])
        best = rows[int(np.argmax(kurt))]
        env = np.abs(hilbert(best))
        env = env - env.mean()
        spec = np.abs(np.fft.rfft(env))
        f = np.fft.rfftfreq(env.size, 1 / fs)
        ax.plot(f, spec, lw=0.7)
        pk = f[int(np.argmax(spec[1:]) + 1)]
        ax.set_title(f"{name}: kurt={kurt.max():.0f}, env peak={pk:.0f} Hz", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.set_xlim(0, fs / 4)
        ax.set_xlabel("Hz", fontsize=6)
    for ax in axes[len(names):]:
        ax.set_visible(False)
    fig.suptitle(f"{sig_name}: envelope spectrum of the most impulsive mode", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path = save(fig, f"{sig_name}_envelope.png")
    plt.close(fig)
    return path


def main() -> int:
    sigs = load_signals()
    made = []
    for sig_name, names in PANELS.items():
        S = sigs[sig_name]
        fs = 12000.0 if sig_name.startswith("cwru") else 360.0
        print(f"== {sig_name} n={S.size}")
        made.append(decomposition_figure(sig_name, S, names))
        made.append(envelope_figure(sig_name, S, names, fs))
    for p in made:
        print("wrote", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
