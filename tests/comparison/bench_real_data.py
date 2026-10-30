"""真实数据上的多方法对比：EMD 家族 / VMD / FMD / EFD / EWT 等。

数据
----
* ``explore/denoise/data/cwru/IR007_105_0.mat`` → ``X105_DE_time``（CWRU 内圈故障，12 kHz）
* ``ref/PySDKit-main/pysdkit/data/real_world/{fmd_demo,ecg_055m}.npy``（PySDKit 自带实测信号）

口径
----
* 同进程、先 warm-up 一次，再取 2 次的最快值（真实数据只做量级对照，不追求 0.1 ms 精度）；
* 指标：``k`` 阶数、``recon`` 重构最大误差、``IO`` Huang 正交指数、
  ``kurt_max`` 各阶峰度的最大值（故障冲击性，越大越"像"轴承故障）、
  ``kurt_top`` 能量最大那阶的峰度；
* 每个方法独立 try，失败只记录不中断（例如 MEMD 需要多通道、CEEFD 可能很慢）。

产物：``table/methods/real_data/methods_matrix.csv``
"""

from __future__ import annotations

import csv
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import scipy.io as sio
from scipy.stats import kurtosis

_ROOT = Path(__file__).resolve().parents[2]
for p in (str(_ROOT / "src"), str(_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from Modal_Decomposition.Base import Seed  # noqa: E402
import Modal_Decomposition as MD  # noqa: E402

OUT = _ROOT / "table" / "methods" / "real_data" / "methods_matrix.csv"
PYSD = _ROOT / "ref" / "PySDKit-main" / "pysdkit" / "data" / "real_world"


# --------------------------------------------------------------------------- #
# 数据
# --------------------------------------------------------------------------- #
def load_signals() -> dict[str, np.ndarray]:
    mat = sio.loadmat(_ROOT / "explore/denoise/data/cwru/IR007_105_0.mat")
    key = next(k for k in mat if not k.startswith("__"))
    x = np.asarray(mat[key]).ravel().astype(float)
    out = {
        "cwru4k": x[2000:2000 + 4096],
        "cwru8k": x[2000:2000 + 8192],
        "ecg4k": np.load(PYSD / "ecg_055m.npy").ravel().astype(float)[:4096],
        "demo8k": np.load(PYSD / "fmd_demo.npy").ravel().astype(float)[:8192],
    }
    return out


# --------------------------------------------------------------------------- #
# 方法表（工厂函数 -> IMFs）
# --------------------------------------------------------------------------- #
def md(name, **kw):
    def run(S):
        cls = getattr(MD.Class, name)
        return np.asarray(cls(**kw).decompose(S).IMFs)
    return run


def pysd(name, **kw):
    def run(S):
        import pysdkit
        cls = getattr(pysdkit, name)
        out = cls(**kw).fit_transform(S)
        return np.asarray(out)
    return run


METHODS = {
    "MD-EMD":       md("EMD", num_imf=5),
    "MD-EEMD":      md("EEMD", num_imf=5, M=10),
    "MD-CEEMD":     md("CEEMD", num_imf=5, M=10),
    "MD-CEEMDAN":   md("CEEMDAN", num_imf=5, M=10),
    "MD-ICEEMDAN":  md("ICEEMDAN", num_imf=5, M=10),
    "MD-RPSEMD":    md("RPSEMD", num_imf=5, sin_smaple_num=4),
    "MD-EFD":       md("EFD", max_IMFs=5),
    "MD-EWT":       md("EWT", num_imfs=4),
    "MD-LMD":       md("LMD", max_pf=3),
    "MD-VMD":       md("VMD", num_imf=3, n=100),
    "MD-SVMD":      md("SVMD", num_modes=3, max_iter=50),
    "MD-SSA":       md("SSA", window_size=128),
    "MD-FMD-ck":    md("FMD", K=3, max_iter=10, num_hand=3, seed=0, solver="ck"),
    "MD-FMD-mckd":  md("FMD", K=3, max_iter=10, num_hand=3, seed=0, solver="mckd"),
    "PySDKit-EMD":  pysd("EMD"),
    "PySDKit-EFD":  pysd("EFD", max_imfs=5),
    "PySDKit-FMD":  pysd("FMD", fs=12000, mode_num=3),
    "PySDKit-VMD":  pysd("VMD", alpha=2000, K=3, tau=0),
}

PER_SIGNAL = {
    "cwru4k": list(METHODS),
    "cwru8k": ["MD-EMD", "MD-EFD", "MD-EWT", "MD-VMD", "MD-FMD-ck", "PySDKit-EFD",
               "PySDKit-FMD", "PySDKit-VMD"],
    "ecg4k":  ["MD-EMD", "MD-EFD", "MD-EWT", "MD-LMD", "MD-VMD", "MD-FMD-ck",
               "PySDKit-EFD", "PySDKit-FMD"],
    "demo8k": ["MD-EFD", "MD-FMD-ck", "MD-FMD-mckd", "PySDKit-EFD", "PySDKit-FMD"],
}


# --------------------------------------------------------------------------- #
def metrics(S: np.ndarray, rows: np.ndarray) -> dict:
    rows = np.atleast_2d(np.asarray(rows, dtype=float))
    if rows.size == 0 or rows.shape[0] == 0:
        return {"k": 0, "recon": float("nan"), "io": float("nan"),
                "kurt_max": float("nan"), "kurt_top": float("nan")}
    stack = np.vstack([rows, S - rows.sum(axis=0)])
    den = float(np.sum(stack * stack))
    num = 0.0
    for i in range(rows.shape[0]):
        for j in range(i + 1, rows.shape[0]):
            num += 2.0 * float(np.dot(rows[i], rows[j]))
    kurt = np.array([float(kurtosis(r)) for r in rows])
    energy = np.array([float(np.sum(r * r)) for r in rows])
    return {
        "k": int(rows.shape[0]),
        "recon": float(np.max(np.abs(S - rows.sum(axis=0)))),
        "io": float(num / den) if den else float("nan"),
        "kurt_max": float(np.max(kurt)),
        "kurt_top": float(kurt[int(np.argmax(energy))]),
    }


def run_one(name: str, fn, S: np.ndarray) -> dict:
    Seed.set(0, verbose=False)
    t0 = time.perf_counter()
    try:
        fn(S)
        t0 = time.perf_counter()
        rows = fn(S)
        wall = (time.perf_counter() - t0) * 1e3
        m = metrics(S, rows)
        rec = {"impl": name, "wall_ms": round(wall, 2), "status": "ok", **m}
    except Exception as exc:                       # noqa: BLE001
        rec = {"impl": name, "wall_ms": None, "status": f"{type(exc).__name__}: {exc}"[:60],
               "k": None, "recon": None, "io": None, "kurt_max": None, "kurt_top": None}
        traceback.print_exc(limit=1)
    print(f"  {name:15s} {str(rec['wall_ms']):>9} ms  k={rec['k']}  "
          f"kurt_max={rec['kurt_max'] if rec['kurt_max'] is None else round(rec['kurt_max'], 1)}  "
          f"IO={rec['io'] if rec['io'] is None else round(rec['io'], 3)}  {'' if rec['status']=='ok' else rec['status']}",
          flush=True)
    return rec


def main() -> int:
    signals = load_signals()
    rows: list[dict] = []
    for sig_name, S in signals.items():
        print(f"== {sig_name}  n={S.size}  std={S.std():.4f}  kurt={kurtosis(S):.2f}")
        for name in PER_SIGNAL[sig_name]:
            rec = run_one(name, METHODS[name], S)
            rec["signal"] = sig_name
            rec["n"] = int(S.size)
            rows.append(rec)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fields = ["signal", "n", "impl", "wall_ms", "status", "k", "recon", "io",
              "kurt_max", "kurt_top"]
    with open(OUT, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fields})
    print(f"\nwrote {OUT}  ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
