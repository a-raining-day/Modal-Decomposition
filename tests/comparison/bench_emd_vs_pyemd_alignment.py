"""
MD (库原生 EMD) vs PyEMD 1.9 —— 可执行对照实验集 (供
docs/EMD_vs_PyEMD_Detailed_Comparison.md 使用)。

覆盖: 极值检测(平台/普通)计数、3 点样条曲线差、linear 分支行为、
幅值标定下的整体停止、病态输入(单调/常值/全零/平台/纯正弦端点)、
白噪声 Wu-Huang 显著性检验、模式能量泄漏、|zc-ext| 分布。
运行: python tests/comparison/bench_emd_vs_pyemd_alignment.py
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
from src.Modal_Decomposition.Utils.Peaks import find_peaks as md_find_peaks  # noqa: E402
from PyEMD import EMD as PyEMD_EMD  # noqa: E402
from PyEMD.splines import cubic_spline_3pts  # noqa: E402
from PyEMD.checks import whitenoise_check  # noqa: E402
from scipy.interpolate import CubicSpline  # noqa: E402

OUT = {}
RES = os.path.join(ROOT, "tests", "comparison", "results")


def md_run(S):
    return MD_EMD().decompose(S)


def py_run(S, **kw):
    emd = PyEMD_EMD(**kw)
    arr = emd.emd(S)
    return emd, arr


def zc_ext_stat(imfs):
    diffs = []
    for row in imfs:
        zc = int(np.sum(np.diff(np.signbit(row)) != 0))
        m1 = (row[1:-1] > row[:-2]) & (row[1:-1] >= row[2:])
        m2 = (row[1:-1] < row[:-2]) & (row[1:-1] <= row[2:])
        diffs.append(abs(zc - (int(np.sum(m1)) + int(np.sum(m2)))))
    return diffs


def regress_r2(y, X):
    """y 在列空间 X (rows=IMFs) 上的 OLS R^2 (跨行泄漏代理)。"""
    X = np.asarray(X, float).T
    y = np.asarray(y, float)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ coef
    ss = float(np.sum((y - y.mean()) ** 2))
    if ss == 0:
        return 1.0
    return float(1 - np.sum((y - pred) ** 2) / ss)


def timed(fn):
    t0 = time.perf_counter()
    out = fn()
    return out, (time.perf_counter() - t0) * 1e3


# ---------------------------------------------------------------- E-A scale
ea = {}
for scale in (1.0, 1000.0):
    S, modes = build_case("A", 4096)
    S = S * scale
    (emd, arr), tp = timed(lambda: py_run(S))
    t2 = time.perf_counter()
    rm = md_run(S)
    tm = (time.perf_counter() - t2) * 1e3
    res_py = S - arr[:-1].sum(0) if arr.ndim >= 2 else S - arr.sum(0)
    res_md = rm.Res
    ea[scale] = dict(
        py_t_ms=round(tp, 2), md_t_ms=round(tm, 2),
        py_k=arr.shape[0] - (1 if arr.ndim >= 2 and not np.allclose(res_py, 0) else 0),
        md_k=int(rm.IMFs.shape[0]),
        py_res_range=float(np.max(res_py) - np.min(res_py)),
        md_res_range=float(np.max(res_md) - np.min(res_md)),
        py_sumabs=float(np.sum(np.abs(res_py))), md_sumabs=float(np.sum(np.abs(res_md))),
        py_relres=float(np.linalg.norm(res_py) / np.linalg.norm(S)),
        md_relres=float(np.linalg.norm(res_md) / np.linalg.norm(S)),
    )
    print(f"E-A scale={scale}: PyEMD k={ea[scale]['py_k']} t={tp:.1f}ms "
          f"res_range={ea[scale]['py_res_range']:.3e} sumabs={ea[scale]['py_sumabs']:.3e} | "
          f"MD k={ea[scale]['md_k']} t={tm:.1f}ms "
          f"res_range={ea[scale]['md_res_range']:.3e} sumabs={ea[scale]['md_sumabs']:.3e}", flush=True)
OUT["E_A_scale"] = ea

# ---------------------------------------------------------------- E-B 端点
S = np.sin(2 * np.pi * 7.0 * np.arange(512) / 100.0)
(emd, arr), _ = timed(lambda: py_run(S))
rm = md_run(S)
sin = S
imf_py = arr[0]
imf_md = rm.IMFs[0]
def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))
OUT["E_B_endpoint"] = dict(
    py_head50=rmse(imf_py[:50], sin[:50]), py_tail50=rmse(imf_py[-50:], sin[-50:]),
    md_head50=rmse(imf_md[:50], sin[:50]), md_tail50=rmse(imf_md[-50:], sin[-50:]),
    py_k=arr.shape[0] - 1, md_k=int(rm.IMFs.shape[0]),
)
print("E-B endpoint RMSE head/tail 50:", OUT["E_B_endpoint"], flush=True)

# ---------------------------------------------------------------- E-C 病态
ec = {}
cases = {
    "monotonic": np.linspace(0, 1, 512),
    "constant": np.full(512, 3.0),
    "zeros": np.zeros(512),
    "plateau5": (5 * np.round(np.sin(np.linspace(0, 6 * np.pi, 512)))).astype(float),
}
for name, S in cases.items():
    (emd, arr), tp = timed(lambda: py_run(S))
    t_md = time.perf_counter()
    rm = md_run(S)
    tm = (time.perf_counter() - t_md) * 1e3
    row = dict(
        py_k=(arr.shape[0] - 1 if arr.ndim >= 2 and arr.shape[0] > 0 else 0),
        md_k=int(rm.IMFs.shape[0]), py_t_ms=round(tp, 2),
        md_t_ms=round(tm, 2),
    )
    ec[name] = row
    print(f"E-C {name}: PyEMD k={row['py_k']} MD k={row['md_k']}", flush=True)
# 同一量化平台信号在 PyEMD parabol 检测下的对照 (验证默认 simple 行为是否检测器特例)
(emdp, arrp), tpp = timed(lambda: py_run(cases["plateau5"], extrema_detection="parabol"))
ec["plateau5_parabol"] = dict(
    py_k=(arrp.shape[0] - 1 if arrp.ndim >= 2 and arrp.shape[0] > 0 else 0),
    py_t_ms=round(tpp, 2),
)
print(f"E-C plateau5 (parabol): PyEMD k={ec['plateau5_parabol']['py_k']}", flush=True)
OUT["E_C_pathological"] = ec

# ---------------------------------------------------------------- E-D 3 点样条
x = np.array([10.0, 30.0, 60.0])
y = np.array([0.0, 2.0, 0.0])
grid = np.linspace(0, 100, 1001)
t3, q_py = cubic_spline_3pts(x, y, grid)
q_sc = CubicSpline(x, y)(grid)
mask = (grid >= t3[0]) & (grid <= t3[-1])  # PyEMD 3pt 分支只返回节点区间
OUT["E_D_spline3"] = dict(
    py_len=len(q_py), grid_len=len(grid),            # 长度截断行为
    max_abs_diff_overlap=float(np.max(np.abs(q_py - q_sc[mask]))),
    max_abs_diff_full=float(np.max(np.abs(
        np.interp(grid, t3, q_py) - q_sc))),          # 界外线性延拓后全长差
)
print("E-D 3pt spline:", OUT["E_D_spline3"], flush=True)

# ---------------------------------------------------------------- E-E linear
S, _ = build_case("A", 1024)
try:
    (emd, arr), tp = timed(lambda: py_run(S, spline_kind="linear"))
    el = dict(ok=True, rows=arr.shape[0], t_ms=round(tp, 2))
except Exception as e:
    el = dict(ok=False, err=repr(e))
rm = md_run(S)                                    # MD 默认档 (CubicSpline)
t_lin = time.perf_counter()
rm_lin = MD_EMD(spline_kind="linear").decompose(S)  # MD linear (= np.interp)
tm_lin = (time.perf_counter() - t_lin) * 1e3
OUT["E_E_linear"] = dict(
    pyemd=el, md_default_rows=int(rm.IMFs.shape[0]),
    md_linear_ok=True, md_linear_rows=int(rm_lin.IMFs.shape[0]),
    md_linear_t_ms=round(tm_lin, 2),
)
print("E-E linear:", OUT["E_E_linear"], flush=True)

# ---------------------------------------------------------------- E-F 白噪声检验
S = build_case("C", 8192)[0]
(emd, arr), tp = timed(lambda: py_run(S))
imfs_py, res_py = emd.get_imfs_and_residue()
rm = md_run(S)
try:
    sig_py = whitenoise_check(np.asarray(imfs_py, float))
except Exception as e:
    sig_py = {"err": repr(e)}
try:
    sig_md = whitenoise_check(np.asarray(rm.IMFs, float))
except Exception as e:
    sig_md = {"err": repr(e)}
OUT["E_F_noise_check"] = dict(
    py_k=int(imfs_py.shape[0]), md_k=int(rm.IMFs.shape[0]),
    sig_py=sig_py, sig_md=sig_md,
    py_zcext=zc_ext_stat(np.asarray(imfs_py, float)),
    md_zcext=zc_ext_stat(np.asarray(rm.IMFs, float)),
)
print("E-F whitenoise: PyEMD", sig_py, "| MD", sig_md, flush=True)

# ---------------------------------------------------------------- E-G 泄漏
S, modes = build_case("A", 4096)
(emd, arr), _ = timed(lambda: py_run(S))
imfs_py, _ = emd.get_imfs_and_residue()
rm = md_run(S)
tone = modes["tone_37hz"]
OUT["E_G_leak"] = dict(
    py_r2_all=round(regress_r2(tone, np.asarray(imfs_py, float)), 4),
    md_r2_all=round(regress_r2(tone, np.asarray(rm.IMFs, float)), 4),
)
print("E-G OLS R^2(tone37|all rows):", OUT["E_G_leak"], flush=True)

# ---------------------------------------------------------------- E-H 极值计数
S_p = (5 * np.round(np.sin(np.linspace(0, 6 * np.pi, 512)))).astype(float)
emu = PyEMD_EMD()
ext = emu.find_extrema(np.arange(512, dtype=float), S_p)
idx_md, _ = md_find_peaks(S_p, mod="numpy")
idx_md_min, _ = md_find_peaks(-S_p, mod="numpy")
S_c, _ = build_case("A", 1024)
ext2 = emu.find_extrema(np.arange(1024, dtype=float), S_c)
idx_c, _ = md_find_peaks(S_c, mod="numpy")
idx_cmin, _ = md_find_peaks(-S_c, mod="numpy")
OUT["E_H_extrema"] = dict(
    plateau_py_max=len(ext[0]), plateau_py_min=len(ext[2]), plateau_py_zer=len(ext[4]),
    plateau_md_max=len(idx_md), plateau_md_min=len(idx_md_min),
    plain_py_max=len(ext2[0]), plain_py_min=len(ext2[2]), plain_py_zer=len(ext2[4]),
    plain_md_max=len(idx_c), plain_md_min=len(idx_cmin),
)
print("E-H extrema counts:", OUT["E_H_extrema"], flush=True)

os.makedirs(RES, exist_ok=True)
path = os.path.join(RES, "emd_vs_pyemd_alignment_raw.json")
with open(path, "w", encoding="utf-8") as f:
    json.dump(OUT, f, ensure_ascii=False, indent=2, default=str)
print("wrote", path, flush=True)
