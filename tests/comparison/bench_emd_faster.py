"""
EMD ``faster`` 分支四向对比基准:
MD-quality (EMD(), faster=False 新默认) vs MD-fast (EMD(faster=True) 旧高速档)
vs PyEMD vs PySDKit(ref) —— 全网格 case A/B/C × n。

指标:
  t_ms        中位耗时; k 行数; recon = max|sum(IMFs)+Res−S|; io 行间正交指数;
  corr_*      任一 IMF 与真值分量最大 |corr|; corr_Res_trend 残差趋势相关;
  share1      纯噪声下首 IMF 能量占比;
  valid       |zc−ext|≤1 的行数 (PyEMD indzer 过零口径 + MD Peaks 极值规则,
              对四实现的结果行统一重算; 无平台信号下与 PyEMD 内部口径一致);
  iters       MD 特有: 每行筛分迭代数。

运行:  python tests/comparison/bench_emd_faster.py
输出:  tests/comparison/results/emd_faster_raw.json (+ 控制台打印表)
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in (_ROOT, os.path.join(_ROOT, "ref"), os.path.join(_ROOT, "src"),
           os.path.join(_ROOT, "tests", "comparison")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from signals import build_case, case_desc  # noqa: E402
from src.Modal_Decomposition.EMD import EMD  # noqa: E402

RESULTS_DIR = os.path.join(_ROOT, "tests", "comparison", "results")


# ---------------------------------------------------------------- helpers
class _Res:
    """外部实现的轻量适配 (.IMFs / .Res / .reconstruct)。"""

    def __init__(self, IMFs, Res):
        self.IMFs = np.asarray(IMFs, dtype=np.float64)
        self.Res = np.asarray(Res, dtype=np.float64)
        self.info = {}

    def reconstruct(self):
        return self.IMFs.sum(axis=0) + self.Res


def pyemd_run(S):
    from PyEMD import EMD as PE
    arr = np.asarray(PE(spline_kind="cubic", nbsym=2).emd(S, max_imf=-1),
                     dtype=np.float64)
    if arr.ndim < 2:
        return _Res(arr.reshape(1, -1), np.zeros(arr.shape[0], dtype=np.float64))
    return _Res(arr[:-1], arr[-1])


def pysdkit_run(S):
    import pysdkit
    arr = np.asarray(pysdkit.EMD(max_imfs=-1).fit_transform(S), dtype=np.float64)
    if arr.ndim < 2:
        return _Res(arr.reshape(1, -1), np.zeros(arr.shape[0], dtype=np.float64))
    return _Res(arr[:-1], arr[-1])


def md_run(faster):
    def _run(S):
        return EMD(faster=faster).decompose(S)
    return _run


RUNNERS = [
    ("MD-quality(faster=False)", md_run(False), True),
    ("MD-fast(faster=True)", md_run(True), True),
    ("PyEMD", pyemd_run, False),
    ("PySDKit", pysdkit_run, False),
]

# (case, ns): 网格 (与主验证一致 + C/8192 中间点)
GRID = [
    ("A", [1024, 4096, 16384, 65536]),
    ("B", [1024, 4096, 16384]),
    ("C", [4096, 8192, 16384]),
]


def zero_cross_count(x):
    s1, s2 = x[:-1], x[1:]
    n = int(np.sum(s1 * s2 < 0))
    if np.any(x == 0):
        indz = np.nonzero(x == 0)[0]
        if np.any(np.diff(indz) == 1):
            z = x == 0
            dz = np.diff(np.concatenate(([0], z, [0])))
            debz = np.nonzero(dz == 1)[0]
            n += int(debz.size)
    return n


def row_balance(x):
    """|zc − ext| (PyEMD 过零口径 + MD Peaks 极值规则)。"""
    zc = zero_cross_count(x)
    m1 = int(np.sum((x[1:-1] > x[:-2]) & (x[1:-1] >= x[2:])))
    m2 = int(np.sum((x[1:-1] < x[:-2]) & (x[1:-1] <= x[2:])))
    return abs(zc - (m1 + m2))


def median_time(fn, repeats: int) -> float:
    fn()  # 预热
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def max_abs_corr(imfs, ref):
    return max((abs(float(np.corrcoef(imfs[i], ref)[0, 1]))
                for i in range(imfs.shape[0])), default=0.0)


def quality_of(r, S, modes):
    imfs = r.IMFs
    k = int(imfs.shape[0])
    finite = bool(np.all(np.isfinite(imfs)) and np.all(np.isfinite(r.Res)))
    recon = float(np.max(np.abs(r.reconstruct() - S)))
    io = float("nan")
    if k >= 2:
        denom = float(np.dot(S, S))
        if denom > 0:
            acc = 0.0
            for i in range(k):
                for j in range(i + 1, k):
                    acc += abs(float(np.dot(imfs[i], imfs[j])))
            io = acc / denom
    out = dict(k=k, finite=finite, recon=f"{recon:.2e}", io=round(io, 4),
               valid=sum(1 for row in imfs if row_balance(row) <= 1))
    for name, mode in modes.items():
        out["corr_" + name] = round(max_abs_corr(imfs, mode), 4)
    if "trend" in modes:
        out["corr_Res_trend"] = round(
            abs(float(np.corrcoef(r.Res, modes["trend"])[0, 1])), 4)
    if not modes and k:
        e1 = float(np.sum(imfs[0] ** 2))
        out["share1"] = round(e1 / float(np.dot(S, S)), 4)
    if r.info:  # MD: 每行迭代
        out["iters_sum"] = int(sum(r.info["iterations"]))
        out["iters"] = [int(v) for v in r.info["iterations"]]
    return out


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    cells = []

    for case, ns in GRID:
        print(f"\n===== case {case}: {case_desc(case)} =====", flush=True)
        for n in ns:
            S, modes = build_case(case, n)
            row = dict(case=case, n=n)
            print(f"--- case {case} n={n} ---", flush=True)
            for name, run, is_md in RUNNERS:
                rep = 2 if (not is_md and n >= 65536) else (3 if not is_md else 3)
                t_ms = round(median_time(lambda: run(S), repeats=rep) * 1e3, 2)
                r = run(S)
                q = quality_of(r, S, modes)
                cell = dict(impl=name, t_ms=t_ms, **q)
                row[name] = cell
                extra = " ".join(f"{kk}={vv}" for kk, vv in q.items()
                                 if kk not in ("k", "recon", "finite", "io",
                                               "iters_sum", "iters", "valid"))
                print(f"  {name:<24} t={t_ms:9.2f}ms k={q['k']:>2} "
                      f"valid={q['valid']}/{q['k']} "
                      + (f"iters={q.get('iters_sum')} " if is_md else "")
                      + extra, flush=True)
            cells.append(row)

    out = dict(numpy=np.__version__, cells=cells)
    path = os.path.join(RESULTS_DIR, "emd_faster_raw.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\nwrote {path}", flush=True)


if __name__ == "__main__":
    main()
