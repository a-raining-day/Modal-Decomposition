"""EFD / FMD 原生实现 vs PySDKit —— 同口径基准（时间 + 质量）。

用法::

    # 1) 现状对照（默认参数）
    .\\.venv\\Scripts\\python.exe tests/comparison/bench_efd_fmd_native.py --mode compare

    # 2) FMD 参数网格（找 Pareto：时间 vs 质量）
    .\\.venv\\Scripts\\python.exe tests/comparison/bench_efd_fmd_native.py --mode grid

    # 3) 只测某一组 FMD 参数
    .\\.venv\\Scripts\\python.exe tests/comparison/bench_efd_fmd_native.py --mode compare \
        --fmd '{"L_factor": 3, "K": 3, "max_iter": 10, "num_hand": 3}'

产物：

* ``tests/comparison/results/efd_fmd_native/{efd,fmd,grid}.json``  原始逐格
* ``table/methods/efd_fmd_native/*.csv``                          汇总表

口径
----
* 用时 = 同进程 warm-up 1 次后 3 次调用的**中位**；外部依赖的 import 已预热。
* 质量 = ``tests/comparison/quality.py``：重构最大误差、Huang 正交指数 IO、
  以及每条真值分量的最优 |corr| / NRMSE（跨全部输出模态取最优）。
* PySDKit 侧：EFD 用 ``max_imfs``、FMD 用 ``mode_num``，其余用其自带默认
  （FMD: ``filter_size=30, cut_num=7, max_iter_num=20``）。
"""

from __future__ import annotations

import argparse
import importlib
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]
for p in (str(_ROOT / "src"), str(_ROOT), str(_HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import quality  # noqa: E402  (tests/comparison/quality.py)
from signals import FS, build_case  # noqa: E402  (tests/comparison/signals.py)

RESULTS = _HERE / "results" / "efd_fmd_native"
TABLES = _ROOT / "table" / "methods" / "efd_fmd_native"
JSON_OUT = RESULTS / "compare.json"

CASES = ("A", "B", "C")
DEFAULT_NS = (256, 1024, 4096, 16384)
FMD_DEFAULT = {"K": 3, "max_iter": 10, "num_hand": 3}   # L_factor 用库默认（现为 2）
EFD_DEFAULT = {"max_IMFs": 3}


# --------------------------------------------------------------------------- #
# 被测实现
# --------------------------------------------------------------------------- #
def _load_native():
    pkg = importlib.import_module("Modal_Decomposition")
    return pkg


def stack_of(rows: np.ndarray, S: np.ndarray) -> np.ndarray:
    rows = np.asarray(rows, dtype=np.float64)
    return np.vstack([rows, S - np.sum(rows, axis=0)])


def metrics(S: np.ndarray, rows: np.ndarray, gt: dict) -> dict:
    st = stack_of(rows, S)
    rec = {
        "k": int(np.asarray(rows).shape[0]),
        "recon_max_abs": quality.recon_max_abs(S, st),
        "orthogonality_index": float(quality.orthogonality_index(st)),
        "energy_frac": [float(np.sum(r ** 2) / max(float(np.sum(S ** 2)), 1e-300))
                        for r in np.asarray(rows, dtype=np.float64)],
    }
    for name, ref in gt.items():
        corr, nrmse = quality._best_corr_nrmse(ref, np.asarray(rows, dtype=np.float64))
        rec[f"corr_{name}"] = float(corr)
        rec[f"nrmse_{name}"] = float(nrmse)
    return rec


def timeit(fn, repeat: int = 3) -> dict:
    fn()                                             # warm-up（含 scipy/cache 首次导入）
    xs = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        out = fn()
        xs.append(time.perf_counter() - t0)
    return {"wall_s": statistics.median(xs), "wall_min": min(xs), "reps": repeat,
            "out": out}


# --------------------------------------------------------------------------- #
# 单格
# --------------------------------------------------------------------------- #
def run_cell(impl: str, case: str, n: int, fmd_params: dict, efd_params: dict) -> dict:
    S, gt = build_case(case, n)
    pkg = _load_native()

    if impl == "MD-EFD":
        cls = pkg.EFD(**efd_params)
        fn = lambda: cls.decompose(S).IMFs                     # noqa: E731
    elif impl == "PySDKit-EFD":
        from pysdkit import EFD as PyEFD
        obj = PyEFD(max_imfs=efd_params["max_IMFs"])
        fn = lambda: np.asarray(obj.fit_transform(S))           # noqa: E731
    elif impl == "MD-FMD":
        cls = pkg.FMD(**fmd_params)
        fn = lambda: cls.decompose(S).IMFs                      # noqa: E731
    elif impl == "PySDKit-FMD":
        from pysdkit import FMD as PyFMD
        obj = PyFMD(fs=int(FS), mode_num=int(fmd_params["K"]))
        fn = lambda: np.asarray(obj.fit_transform(S))           # noqa: E731
    else:
        raise ValueError(impl)

    timing = timeit(fn)
    rows = timing.pop("out")
    return {"impl": impl, "case": case, "n": n, **timing, **metrics(S, rows, gt)}


# --------------------------------------------------------------------------- #
# 输出
# --------------------------------------------------------------------------- #
def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    import csv

    flat = []
    for r in rows:
        d = {k: v for k, v in r.items() if not isinstance(v, list)}
        d["energy_frac"] = ";".join(f"{v:.4f}" for v in r.get("energy_frac", []))
        flat.append(d)
    fields: list[str] = []                       # 各 case 的真值键不同 -> 取并集
    for d in flat:
        for k in d:
            if k not in fields:
                fields.append(k)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(flat)


def write_json(path: Path, payload) -> None:
    """写原始 JSON；当前沙箱只允许 ``table/`` 写入，所以失败时把内容转到 stdout。"""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"wrote {path}")
    except OSError as exc:                       # pragma: no cover - 沙箱相关
        print(f"[warn] cannot write {path} ({type(exc).__name__}); payload below")
        print("@@JSON@@" + json.dumps(payload, ensure_ascii=False))


def env() -> dict:
    import scipy

    return {"python": platform.python_version(), "numpy": np.__version__,
            "scipy": scipy.__version__, "machine": platform.machine(),
            "platform": platform.platform()}


# --------------------------------------------------------------------------- #
# 模式
# --------------------------------------------------------------------------- #
def mode_compare(ns, fmd_params, efd_params, cases=CASES) -> int:
    rows = []
    for case in cases:
        for n in ns:
            for impl in ("MD-EFD", "PySDKit-EFD", "MD-FMD", "PySDKit-FMD"):
                r = run_cell(impl, case, n, fmd_params, efd_params)
                rows.append(r)
                ratio = ""
                print(f"  {impl:12s} case {case} n={n:6d}  {r['wall_s']*1e3:9.2f} ms  "
                      f"k={r['k']} recon={r['recon_max_abs']:.2e} "
                      f"IO={r['orthogonality_index']:+.4f} {ratio}", flush=True)

    write_json(RESULTS / "compare.json", {"env": env(), "fmd": fmd_params,
                                          "efd": efd_params, "rows": rows})
    write_csv(TABLES / "compare.csv", rows)
    print(f"wrote {RESULTS / 'compare.json'} and {TABLES / 'compare.csv'}")
    return 0


def mode_grid(ns, efd_params) -> int:
    grid = [{"L_factor": lf, "K": 3, "max_iter": mi, "num_hand": nh}
            for lf in (1, 2, 3, 5, 10)
            for mi in (5, 10, 20)
            for nh in (1, 3)]
    rows = []
    for params in grid:
        for case in ("A", "B"):
            for n in (1024, 4096):
                r = run_cell("MD-FMD", case, n, params, efd_params)
                r["params"] = json.dumps(params, sort_keys=True)
                rows.append(r)
                print(f"  L_factor={params['L_factor']:<3} max_iter={params['max_iter']:<3} "
                      f"hand={params['num_hand']:<2} case {case} n={n:5d}  "
                      f"{r['wall_s']*1e3:8.2f} ms  "
                      + "  ".join(f"{k}={r[k]:.3f}" for k in r if k.startswith("corr_")),
                      flush=True)
    write_json(RESULTS / "grid.json", {"env": env(), "rows": rows})
    write_csv(TABLES / "fmd_grid.csv", rows)
    print(f"wrote {RESULTS / 'grid.json'} and {TABLES / 'fmd_grid.csv'}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("compare", "grid"), default="compare")
    ap.add_argument("--ns", type=int, nargs="*", default=list(DEFAULT_NS))
    ap.add_argument("--fmd", type=str, default=None,
                    help='JSON, e.g. \'{"L_factor":3,"K":3,"max_iter":10,"num_hand":3}\'')
    ap.add_argument("--efd", type=str, default=None,
                    help='JSON, e.g. \'{"max_IMFs":3}\'')
    args = ap.parse_args()

    fmd_params = {**FMD_DEFAULT, **json.loads(args.fmd)} if args.fmd else dict(FMD_DEFAULT)
    efd_params = {**EFD_DEFAULT, **json.loads(args.efd)} if args.efd else dict(EFD_DEFAULT)

    print(f"env: {env()}")
    print(f"FMD params: {fmd_params}")
    print(f"EFD params: {efd_params}")
    if args.mode == "grid":
        return mode_grid(tuple(args.ns), efd_params)
    return mode_compare(tuple(args.ns), fmd_params, efd_params)


if __name__ == "__main__":
    raise SystemExit(main())
