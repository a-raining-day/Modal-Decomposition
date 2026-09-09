"""刷新验证网格 JSON: MD 两档改用新默认配置重测, 外部基线缓存保留。"""
import json
import os
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for p in (ROOT, os.path.join(ROOT, "ref"), os.path.join(ROOT, "src"),
          os.path.join(ROOT, "tests", "comparison")):
    if p not in os.sys.path:
        os.sys.path.insert(0, p)

from signals import build_case  # noqa: E402
from src.Modal_Decomposition.EMD import EMD  # noqa: E402

PATH = os.path.join(ROOT, "tests", "comparison", "results", "emd_validation_raw.json")
data = json.load(open(PATH, encoding="utf-8"))

CONFIGS = [
    ("MD-def(CubicSpline/sd0.01)", {}),
    ("MD-linear/sd0.01", dict(spline_kind="linear", sd_thr=0.01)),
]


def zc_ext(row):
    zc = int(np.sum(np.diff(np.signbit(row)) != 0))
    m1 = (row[1:-1] > row[:-2]) & (row[1:-1] >= row[2:])
    m2 = (row[1:-1] < row[:-2]) & (row[1:-1] <= row[2:])
    return zc, int(np.sum(m1)) + int(np.sum(m2))


def max_abs_corr(row, ref):
    return max((abs(float(np.corrcoef(row[i], ref)[0, 1]))
                for i in range(row.shape[0])), default=0.0)


def measure(S, modes, kw):
    t0 = time.perf_counter()
    r = EMD(**kw).decompose(S)
    dt = (time.perf_counter() - t0) * 1e3
    imfs, res = r.IMFs, r.Res
    k = imfs.shape[0]
    valid = sum(1 for i in range(k) if abs(zc_ext(imfs[i])[0] - zc_ext(imfs[i])[1]) <= 1)
    q = dict(k=k, valid=f"{valid}/{k}",
             finite=bool(np.all(np.isfinite(imfs)) and np.all(np.isfinite(res))),
             recon=f"{float(np.max(np.abs(r.reconstruct() - S))):.2e}",
             t_ms=round(dt, 2))
    for name, mode in modes.items():
        q["corr_" + name] = round(max_abs_corr(imfs, mode), 4)
    if "trend" in modes:
        q["corr_Res_trend"] = round(
            abs(float(np.corrcoef(res, modes["trend"])[0, 1])), 4)
    if not modes:
        e1 = float(np.sum(imfs[0] ** 2)) if k else float("nan")
        q["share1"] = round(e1 / float(np.dot(S, S)), 4) if k else float("nan")
    return q


for cell in data["cells"]:
    case, n = cell["case"], cell["n"]
    S, modes = build_case(case, n)
    for name, kw in CONFIGS:
        q = measure(S, modes, kw)
        cell[name] = dict(impl=name, **q)
        print(f"  {case} n={n:>6} {name:<24} t={q['t_ms']:8.2f}ms k={q['k']:>2} "
              f"valid={q.get('valid','-')} "
              + " ".join(f"{kk}={vv}" for kk, vv in q.items()
                         if kk not in ("k", "valid", "finite", "t_ms")), flush=True)
    # 移除旧默认行
    for old in ("MD-default(linear/sd0.3)", "MD-cubic/sd0.2"):
        cell.pop(old, None)

with open(PATH, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)
print("updated", PATH, flush=True)
