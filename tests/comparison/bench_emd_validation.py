"""
EMD 三方验证基准 (validation grid): 库 EMD (原生) vs PyEMD vs PySDKit(ref).

覆盖用户要求的扩展面:
  * 长度: 256 -> 65536 (双音调 case A 到 65536, 验证历史文档中
    PyEMD/PySDKit 在该长度出现最大 IMF 数分歧的尺度);
  * 信号: case A (双音调+轻噪声), case B (AM-FM 调频+纯音+趋势),
    case C (纯白噪声 —— 历史压力测试, 最能暴露拆分与停止判据差异);
  * 库 EMD 两档: 默认 linear/sd0.3 (最佳配置) 与 CubicSpline/sd0.2 (同包络档)。

运行:  python tests/comparison/bench_emd_validation.py
输出:  tests/comparison/results/emd_validation_raw.json (全量原始数据)
       控制台打印各表 (供 docs/EMD_Validation_and_Comparison_Report.md 引用)
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


class _Res:
    """外部实现的轻量适配 (.IMFs / .Res / .reconstruct)。"""

    def __init__(self, IMFs, Res):
        self.IMFs = np.asarray(IMFs, dtype=np.float64)
        self.Res = np.asarray(Res, dtype=np.float64)

    def reconstruct(self):
        return self.IMFs.sum(axis=0) + self.Res


def pyemd_run(S):
    from PyEMD import EMD as PE
    arr = np.asarray(PE(spline_kind="cubic", nbsym=2).emd(S, max_imf=-1),
                     dtype=np.float64)
    return _Res(arr[:-1], arr[-1]) if arr.ndim >= 2 else \
        _Res(arr.reshape(1, -1), np.zeros(arr.shape[0], dtype=np.float64))


def pysdkit_run(S):
    import pysdkit
    arr = np.asarray(pysdkit.EMD(max_imfs=-1).fit_transform(S), dtype=np.float64)
    return _Res(arr[:-1], arr[-1]) if arr.ndim >= 2 else \
        _Res(arr.reshape(1, -1), np.zeros(arr.shape[0], dtype=np.float64))


RUNNERS = [
    ("PyEMD", pyemd_run),
    ("PySDKit", pysdkit_run),
    ("MD-default(linear/sd0.3)", lambda S: EMD().decompose(S)),
    ("MD-cubic/sd0.2", lambda S: EMD(spline_kind="CubicSpline", sd_thr=0.2).decompose(S)),
]

# (case, ns): 网格长度
GRID = [
    ("A", [1024, 4096, 16384, 65536]),
    ("B", [1024, 4096, 16384]),
    ("C", [4096, 16384]),
]


def median_time(fn, repeats: int) -> float:
    fn()  # 预热 (import 已在上方发生过; 排除实例化抖动)
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def max_abs_corr(row, ref):
    """任一 IMF 与参考分量的最大 |相关系数| (行可能与参考反相)。"""
    return max((abs(float(np.corrcoef(row[i], ref)[0, 1]))
                for i in range(row.shape[0])), default=0.0)


def quality_of(r, S, modes):
    imfs = r.IMFs
    finite = bool(np.all(np.isfinite(imfs)) and np.all(np.isfinite(r.Res)))
    recon = float(np.max(np.abs(r.reconstruct() - S)))
    k = int(imfs.shape[0])
    io = float("nan")
    if k >= 2:
        denom = float(np.dot(S, S))
        if denom > 0:
            acc = 0.0
            for i in range(k):
                for j in range(i + 1, k):
                    acc += abs(float(np.dot(imfs[i], imfs[j])))
            io = acc / denom
    out = dict(k=k, finite=finite, recon=f"{recon:.2e}", io=round(io, 4))
    for name, mode in modes.items():
        out["corr_" + name] = round(max_abs_corr(imfs, mode), 4)
    if "trend" in modes:
        out["corr_Res_trend"] = round(
            abs(float(np.corrcoef(r.Res, modes["trend"])[0, 1])), 4)
    if not modes:  # 白噪声: 首 IMF 能量占比 (拆分度指标)
        e1 = float(np.sum(imfs[0] ** 2)) if k else float("nan")
        out["share1"] = round(e1 / float(np.dot(S, S)), 4) if k else float("nan")
    return out


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    env = dict(numpy=np.__version__)
    cells = []

    for case, ns in GRID:
        print(f"\n===== case {case}: {case_desc(case)} =====", flush=True)
        for n in ns:
            S, modes = build_case(case, n)
            row = dict(case=case, n=n)
            for name, run in RUNNERS:
                ext = name.startswith(("PyEMD", "PySDKit"))
                # 外部实现: 65536 每跑 ~20-24 s -> 3 次中位; 其余 5 次
                rep = 3 if ext and n >= 65536 else (5 if ext else 5)
                t_ms = round(median_time(lambda: run(S), repeats=rep) * 1e3, 2)
                r = run(S)
                q = quality_of(r, S, modes)
                cell = dict(impl=name, t_ms=t_ms, **q)
                row[name] = cell
                print(f"  {case} n={n:>6} {name:<22} t={t_ms:9.2f}ms "
                      f"k={q['k']:>2} recon={q['recon']} "
                      + " ".join(f"{kk}={vv}" for kk, vv in q.items()
                                 if kk not in ("k", "recon", "finite", "io"))
                      , flush=True)
            cells.append(row)

    out = dict(env=env, cells=cells)
    path = os.path.join(RESULTS_DIR, "emd_validation_raw.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\nwrote {path}", flush=True)


if __name__ == "__main__":
    main()
