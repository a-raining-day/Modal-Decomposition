"""
MD EMD 停止判据变体实验 —— 回答「为什么 MD 快但质量低 / 如何优化 / 速度影响」。

不修改库代码: 用子类覆写 MD.EMD._sift 的停止判据, 复刻 PyEMD 式语义:
- sd   : 能量型 Cauchy  sd = sum(m^2)/sum(h_prev^2) < sd_thr      (MD 现状)
- svar : 幅值归一型        svar = sum(m^2)/range(h_prev) < svar_thr (PyEMD 主判据,
         默认 0.001, 本实验对 MD 路径的近似——忽略其 std/energy 次级判据)
- *_nb : 在上述收敛判据之外再要求「窄带门」: |zc - ext| <= 1 在更新后的 h 上成立
         (PyEMD 默认停止 = check_imf(f1) and |ext-zc|<2 (f2), 单次)
- *_nbK : 窄带门连续 K 次成立才停 (PyEMD 的记账思想, FIXE_H)

对照: PyEMD 1.9 默认 (含每迭代 find_extrema 调用计数, ~3 scans/iter)。
信号: case A (双音调+噪) n=4096/16384/65536; case C (纯白噪) n=8192。
运行: python tests/comparison/bench_emd_stopping_variants.py
输出: tests/comparison/results/emd_stopping_variants_raw.json
"""
from __future__ import annotations

import json
import os
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for p in (ROOT, os.path.join(ROOT, "src"), os.path.join(ROOT, "tests", "comparison")):
    if p not in os.sys.path:
        os.sys.path.insert(0, p)

from signals import build_case  # noqa: E402
from src.Modal_Decomposition.EMD import EMD as MD_EMD  # noqa: E402
from PyEMD import EMD as PyEMD_EMD  # noqa: E402

RES = os.path.join(ROOT, "tests", "comparison", "results")


# ---------------------------------------------------------------- helpers
def zero_cross_count(x: np.ndarray) -> int:
    """PyEMD indzer 语义的过零计数: 符号积 <0 加零点段取段数。"""
    s1, s2 = x[:-1], x[1:]
    n = int(np.sum(s1 * s2 < 0))
    if np.any(x == 0):
        indz = np.nonzero(x == 0)[0]
        if np.any(np.diff(indz) == 1):  # 存在零点段才做段合并 (PyEMD 同款条件)
            z = x == 0
            dz = np.diff(np.concatenate(([0], z, [0])))
            debz = np.nonzero(dz == 1)[0]
            n += int(debz.size)
    return n


def zc_ext_balance(x: np.ndarray) -> int:
    """|zc - ext|: zc 用过零段计数, ext 用 MD Peaks 规则 (与 E-F 同法同口径)。"""
    zc = zero_cross_count(x)
    m1 = int(np.sum((x[1:-1] > x[:-2]) & (x[1:-1] >= x[2:])))
    m2 = int(np.sum((x[1:-1] < x[:-2]) & (x[1:-1] <= x[2:])))
    return abs(zc - (m1 + m2))


def best_corr(imfs: np.ndarray, tone: np.ndarray) -> float:
    return max(abs(np.corrcoef(r, tone)[0, 1]) for r in imfs)


# ---------------------------------------------------------------- MD 变体
class EMDVariant(MD_EMD):
    """
    mode in {"sd","svar","sd_nb","svar_nb","svar_nb3"}:
      sd/svar   — 纯收敛判据 (无窄带门);
      *_nb      — 收敛判据且当前 h 满足 |zc-ext|<=1 才停;
      svar_nb3  — 收敛判据且窄带门连续 3 次成立才停。
    """

    def __init__(self, mode: str, svar_thr: float = 0.001, **kw) -> None:
        super().__init__(**kw)
        assert mode in ("sd", "svar", "sd_nb", "svar_nb", "svar_nb3")
        self.mode = mode
        self.svar_thr = svar_thr
        self._nb_wait = 3 if mode.endswith("3") else 1

    def _sift(self, h, up_idx, dn_idx, grid):
        last = 0.0
        good = 0
        iters = 0
        for it in range(self.max_iter):
            if it > 0:
                up_idx, _ = self.find_peaks(h, mod=self.find_peaks_mod)
                dn_idx, _ = self.find_peaks(-h, mod=self.find_peaks_mod)
                if up_idx.size < 2 or dn_idx.size < 2:
                    break

            up_pos, up_vals = self.mirror_extrema(up_idx, h[up_idx], self.nbsym)
            dn_pos, dn_vals = self.mirror_extrema(dn_idx, h[dn_idx], self.nbsym)
            up = self._envelope(up_pos, up_vals, grid)
            dn = self._envelope(dn_pos, dn_vals, grid)
            mean = (up + dn) * 0.5

            prev_energy = float(np.dot(h, h))
            if prev_energy == 0.0:
                break
            h_new = h - mean
            iters = it + 1

            if self.mode in ("sd", "sd_nb"):
                conv = float(np.dot(mean, mean)) / prev_energy < self.sd_thr
                last = conv
            else:
                rng = float(np.max(h) - np.min(h))
                svar = (float(np.dot(mean, mean)) / rng) if rng > 0 else 0.0
                conv = svar < self.svar_thr
                last = svar

            h = h_new
            if not conv:
                good = 0
                continue
            if self.mode in ("sd", "svar"):   # 无窄带门: 收敛即停
                break
            # 窄带门: 收敛基础上还要求更新后 h 满足 |zc-ext|<=1
            if zc_ext_balance(h) <= 1:
                good += 1
                if good >= self._nb_wait:
                    break
            else:
                good = 0
        return h, iters, last


# ---------------------------------------------------------------- PyEMD 计数
class CountingPyEMD(PyEMD_EMD):
    def __init__(self, **kw) -> None:
        super().__init__(**kw)
        self.calls = 0

    def find_extrema(self, T, S):
        self.calls += 1
        return super().find_extrema(T, S)


# ---------------------------------------------------------------- 实验矩阵
def run_md(cfg, S, tones, repeats: int):
    best_t = np.inf
    last = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        rm = cfg.decompose(S)
        dt = (time.perf_counter() - t0) * 1e3
        if dt < best_t:
            best_t = dt
            last = rm
    rm = last
    row = {
        "t_ms": round(best_t, 2),
        "k": int(rm.IMFs.shape[0]),
        "iters": [int(v) for v in rm.info["iterations"]],
        "valid": None,
        "corr37": None,
        "corr113": None,
        "finite": bool(np.all(np.isfinite(rm.IMFs))),
    }
    if rm.IMFs.shape[0]:
        row["valid"] = sum(1 for r in rm.IMFs if zc_ext_balance(r) <= 1)
        if tones:
            row["corr37"] = round(best_corr(rm.IMFs, tones["tone_37hz"]), 4)
            row["corr113"] = round(best_corr(rm.IMFs, tones["tone_113hz"]), 4)
    return row


def run_py(S, tones):
    emd = CountingPyEMD()
    t0 = time.perf_counter()
    arr = emd.emd(S)
    dt = (time.perf_counter() - t0) * 1e3
    imfs = arr[:-1] if arr.shape[0] > 1 else arr
    row = {
        "t_ms": round(dt, 2),
        "k": int(imfs.shape[0]),
        "iters": [],                      # 未直接暴露; 见 ext_calls
        "ext_calls": emd.calls,
        "est_iters_per_row": round(emd.calls / max(imfs.shape[0], 1) / 3, 1),
        "valid": sum(1 for r in imfs if zc_ext_balance(r) <= 1),
        "corr37": None,
        "corr113": None,
        "finite": bool(np.all(np.isfinite(arr))),
    }
    if tones:
        row["corr37"] = round(best_corr(imfs, tones["tone_37hz"]), 4)
        row["corr113"] = round(best_corr(imfs, tones["tone_113hz"]), 4)
    return row


def main() -> None:
    os.makedirs(RES, exist_ok=True)
    cells = []

    variants = [
        ("V0_sd.01",        dict(mode="sd",     sd_thr=0.01, max_iter=100)),
        ("V1_sd.002",       dict(mode="sd",     sd_thr=0.002, max_iter=200)),
        ("V2_sd.0005",      dict(mode="sd",     sd_thr=0.0005, max_iter=400)),
        ("V3_svar.001",     dict(mode="svar",   max_iter=800)),
        ("V4_sd.01_nb",     dict(mode="sd_nb",  sd_thr=0.01, max_iter=300)),
        ("V5_svar.001_nb",  dict(mode="svar_nb", max_iter=800)),
        ("V6_svar.001_nb3", dict(mode="svar_nb3", max_iter=800)),
    ]

    for case, ns, repeats in (("A", (4096, 16384, 65536), 3), ("C", (8192,), 2)):
        for n in ns:
            S, tones = build_case(case, n)
            print(f"--- case {case} n={n} ---", flush=True)
            cell = {"case": case, "n": n, "rows": {}}
            if case == "A" or n <= 8192:
                pr = run_py(S, tones)
                cell["rows"]["PyEMD"] = pr
                print(f"  PyEMD      : t={pr['t_ms']:9.2f}ms k={pr['k']:2d} "
                      f"c37={pr['corr37']} c113={pr['corr113']} valid={pr['valid']}/{pr['k']} "
                      f"calls={pr['ext_calls']}", flush=True)
            for name, kw in variants:
                r = run_md(EMDVariant(**kw), S, tones, repeats)
                cell["rows"][name] = r
                extra = ""
                if r["corr37"] is not None:
                    extra = f"c37={r['corr37']} c113={r['corr113']}"
                print(f"  {name:16s}: t={r['t_ms']:9.2f}ms k={r['k']:2d} "
                      f"iters={sum(r['iters']):4d} valid={r['valid']}/{r['k']} {extra}", flush=True)
            cells.append(cell)

    out = {"cells": cells}
    path = os.path.join(RES, "emd_stopping_variants_raw.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=str)
    print("wrote", path, flush=True)


if __name__ == "__main__":
    main()
