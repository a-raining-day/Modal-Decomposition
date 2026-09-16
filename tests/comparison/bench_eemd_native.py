"""
EEMD 换版对比实验 (0.3.0 起两条路径同在一个类): PyEMD 路径 (``Class.EEMD(pyemd=True)``) vs 原生路径
(原生实现, ``Class.EEMD``, 见 ``src/Modal_Decomposition/EEMD.py``)。

两个实现的**语义差异**决定了对拍口径:

* 新版 ``Res = S − ΣIMFs`` (硬性精确重构), 行数 = 各轮最小阶数 (逐轮截断);
* 旧版把 PyEMD ``eemd()`` 返回堆栈的**最后一行 IMF 当残差**, 该行不进 ``IMFs``
  ⇒ 少一阶 IMF 且 ``reconstruct()`` 不精确 (实测 0.2-0.35)。

因此每个格子记录:

* ``recon_max_abs_err`` —— **主重构指标** (含 ``Res``): 新版 ~1e-17 … 1e-16,
  旧版 ~0.1 (量级随信号/尺寸变化);
* ``imf_sum_max_abs_err`` —— ``max|ΣIMFs − S|``: 新版 = 真实余量的幅度 (本仓
  测试信号上 ~0.1 量级, 属正常 —— EEMD 只取第 1 阶 IMF, 剩余能量进 ``Res``),
  旧版 = 被丢掉那一行的能量;
* ``n_rows`` —— 各实现**自报**行数 (新版 = 各轮最小阶数, 旧版少一阶);
* ``res_is_exact_remainder`` / ``res_minus_true_residue_max`` —— ``Res`` 是否等于
  ``S − ΣIMFs`` 的判别位与偏差。新版 True (真实余量, 只有它能精确重构); 旧版 False,
  因为它的 ``Res`` 是 PyEMD 原始堆栈最后一行 **IMF** (逐行比对证据见报告), 而
  PyEMD 内部算出的真余量 ``self.residue = S − ΣE_IMF`` 从未被返回;
* ``missing_residue_*`` —— ``S − ΣIMFs`` 的幅度/RMS/相对值: 新版 = 真实余量,
  旧版 = 被丢弃的真实余量;
* ``*_firstk`` (驱动侧 ``_fair_fill`` 补算) —— 两边都只取前
  ``cmp_rows = min(新版行数, 旧版行数)`` 行时的**行级**指标 (正交指数 / 模式捕获):
  同阶数比较可把"旧版少一阶"从行级指标里摘掉; 但该口径把新版 ``Res`` 也排除在
  外, 所以 ``recon_max_abs_err_firstk`` 两边都不接近 0, **不是**重构质量指标;
* ``xcheck_*`` —— 同一格子内三种算法 (自建 ``ΣIMFs+Res`` / ``reconstruct()`` /
  ``quality.analyze``) 的一致性校验, ``xcheck_max_diff`` 应 ≈0
  (``quality.analyze`` 的 ``recon`` 只对行求和, 故单列为
  ``xcheck_recon_err_over_imf_rows``, 它**不是**重构误差指标)。

指标: 中位墙钟 (预热 1 次后 ``--repeats`` 次; ``n>=65536`` 自动降到 1 次) /
RSS 峰值增量 / 行数 / 重构误差 / 行和误差 / Huang 正交指数 / 对真值分量的模式捕获
(best |corr| + 对应 NRMSE) / 新版 ``info`` 诊断 (逐阶逐轮筛分迭代 + ``ensemble_std``)。

用法::

    python tests/comparison/bench_eemd_native.py                       # 默认 headtohead
    python tests/comparison/bench_eemd_native.py --sweeps all --budget-s 1750
    python tests/comparison/bench_eemd_native.py --sweeps trials_sweep noise_sweep
    python tests/comparison/bench_eemd_native.py --impls new --cases A B
    python tests/comparison/bench_eemd_native.py --sweeps all --estimate   # 只出计划
    python tests/comparison/bench_eemd_native.py --worker '{"impl":"new",...}'

结果处理:
  * 每个格子跑在独立子进程 (``--worker`` + ``@@RESULT@@`` 单行 JSON 协议),
    墙钟上限 ``--time-cap-s``; 每格完成即合并落盘, 中途中断也保留已完成部分;
  * **渐进重跑**: 已 ``status=="ok"`` 的格子直接复用 (打印 ``[cached]``),
    只有失败的格子会被重试; 要强制重测用 ``--force``; 本轮计划见 ``--estimate``;
  * ``--budget-s`` 是闸门: 格子预估耗时会让本轮累计超出预算时被跳过并打印
    ``[budget]`` (不会半途浪费), 因此超大尺寸格在紧预算下会被整格放弃。

原始结果 (累积, 形如 ``{"meta":..., "cell_count":..., "cells": {key: row}}``)
写入 ``tests/comparison/results/eemd_native_raw.json``; 直接产出的报告是
``docs/EEMD_Native_vs_PyEMD_Report.md``。

Python version: 3.10
Only accessed by: ``docs/EEMD_Native_vs_PyEMD_Report.md`` (实验复现)
Modify: 2026.3.7
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]
_SRC = _ROOT / "src"
_RESULTS_DIR = _HERE / "results"
_RAW_JSON = _RESULTS_DIR / "eemd_native_raw.json"

#: 默认随机种子 —— 必给: 全局种子未设置 (``Utils.Seed._GLOBAL_SEED is None``) 时
#: ``seed=None`` 会退化为 OS 熵, 两个实现都不可复现。
_SEED = 20260906

#: 原生 ``EEMD.spline_kind`` -> PyEMD ``EMD(spline_kind=...)`` 拼写。
#: 注意 CubicSpline ↔ "cubic" 只是**同一包络家族**, 不是同一插值器:
#: scipy ``CubicSpline`` 默认 not-a-knot, PyEMD "cubic" 是自然三次样条。
_SPLINE_TO_PYEMD = {"CubicSpline": "cubic", "PCHIP": "pchip", "linear": "linear"}

#: 旧版必须串行: PyEMD 自带默认 ``parallel=True`` → ``multiprocessing.Pool`` →
#: 本沙箱禁止命名管道 → ``PermissionError: [WinError 5]``。
_OLD_PARALLEL = False

#: 每个格子 = (impl, case, n, trials, noise_width, spline_kind)。
#: 估算成本 (s) 仅用于预算闸门 (实测校准的保守值, 不写进结果)。
_SWEEPS: dict[str, dict] = {
    # 主对照: 案例 A/B/C × 长度; n=65536 用 trials=5 / repeats=1 (预算所限)。
    "headtohead": {
        "desc": "主对照 A/B/C × 长度 (trials=30; n=16384 与 65536 降 trials 省预算)",
        "cells": [
            ("both", "A", 1024, 30, 0.05, "CubicSpline", 0.8),
            ("both", "A", 4096, 30, 0.05, "CubicSpline", 2.6),
            ("both", "A", 16384, 30, 0.05, "CubicSpline", 40.0),
            ("both", "A", 65536, 5, 0.05, "CubicSpline", 106.0),
            ("both", "B", 1024, 30, 0.05, "CubicSpline", 0.8),
            ("both", "B", 4096, 30, 0.05, "CubicSpline", 2.6),
            ("both", "C", 1024, 30, 0.05, "CubicSpline", 0.9),
            ("both", "C", 4096, 30, 0.05, "CubicSpline", 3.0),
        ],
    },
    # 集成规模扫描: 固定 n=4096 (与主对照同格, 直接复用测量)。
    "trials_sweep": {
        "desc": "集成轮数扫描 (case A/B/C, n=4096, noise_width=0.05)",
        "cells": [
            ("both", "A", 4096, 10, 0.05, "CubicSpline", 1.0),
            ("both", "A", 4096, 30, 0.05, "CubicSpline", 2.6),
            ("both", "A", 4096, 100, 0.05, "CubicSpline", 14.0),
            ("both", "B", 4096, 10, 0.05, "CubicSpline", 1.0),
            ("both", "B", 4096, 30, 0.05, "CubicSpline", 2.6),
            ("both", "B", 4096, 100, 0.05, "CubicSpline", 14.0),
            ("both", "C", 4096, 10, 0.05, "CubicSpline", 1.1),
            ("both", "C", 4096, 30, 0.05, "CubicSpline", 3.0),
            ("both", "C", 4096, 100, 0.05, "CubicSpline", 15.0),
        ],
    },
    # 噪声幅度扫描 (含 rich_info=True 的集成标准差诊断格)。
    "noise_sweep": {
        "desc": "噪声幅度扫描 (case A/B/C, n=4096, trials=30; 附 rich_info 诊断)",
        "cells": [
            ("both", "A", 4096, 30, 0.01, "CubicSpline", 2.6),
            ("both", "A", 4096, 30, 0.05, "CubicSpline", 2.6),
            ("both", "A", 4096, 30, 0.20, "CubicSpline", 3.0),
            ("both", "B", 4096, 30, 0.01, "CubicSpline", 2.6),
            ("both", "B", 4096, 30, 0.05, "CubicSpline", 2.6),
            ("both", "B", 4096, 30, 0.20, "CubicSpline", 3.0),
            ("both", "C", 4096, 30, 0.01, "CubicSpline", 3.0),
            ("both", "C", 4096, 30, 0.05, "CubicSpline", 3.0),
            ("both", "C", 4096, 30, 0.20, "CubicSpline", 3.0),
            ("new", "A", 4096, 30, 0.01, "CubicSpline", 1.2),
            ("new", "A", 4096, 30, 0.05, "CubicSpline", 1.2),
            ("new", "A", 4096, 30, 0.20, "CubicSpline", 1.4),
            ("new", "C", 4096, 30, 0.01, "CubicSpline", 1.4),
            ("new", "C", 4096, 30, 0.05, "CubicSpline", 1.4),
            ("new", "C", 4096, 30, 0.20, "CubicSpline", 1.4),
        ],
    },
    # 包络后端扫描: 新版三档之间**自比**; 旧版前端作为 "PyEMD cubic 基线" 漂移
    # 参考点 (不成对, 不算比值 —— 它与新版之间隔着噪声流差异)。
    "spline_sweep": {
        "desc": "包络后端 (新版 CubicSpline/PCHIP/linear 自比 + PyEMD cubic 基线)",
        "cells": [
            ("new", "A", 4096, 30, 0.05, "CubicSpline", 1.2),
            ("new", "A", 4096, 30, 0.05, "PCHIP", 1.4),
            ("new", "A", 4096, 30, 0.05, "linear", 1.0),
            ("old", "A", 4096, 30, 0.05, "CubicSpline", 1.6),
            ("new", "A", 16384, 30, 0.05, "CubicSpline", 30.0),
            ("new", "A", 16384, 30, 0.05, "PCHIP", 34.0),
            ("new", "A", 16384, 30, 0.05, "linear", 20.0),
            ("old", "A", 16384, 30, 0.05, "CubicSpline", 60.0),
            ("new", "B", 4096, 30, 0.05, "CubicSpline", 1.2),
            ("new", "B", 4096, 30, 0.05, "PCHIP", 1.4),
            ("new", "B", 4096, 30, 0.05, "linear", 1.0),
            ("old", "B", 4096, 30, 0.05, "CubicSpline", 1.6),
        ],
        #: 同一 (case, n, trials, noise_width) 下的新版三档互为对照 (自比组)。
        "self_groups": [
            ("A", 4096, 30, 0.05),
            ("A", 16384, 30, 0.05),
            ("B", 4096, 30, 0.05),
        ],
    },
}

#: rich_info 诊断格 (新版, ensemble_std —— 每次试验同阶结果的逐点标准差)。
_RICH_CELLS = [
    {"case": c, "n": 4096, "trials": 30, "noise_width": nw, "spline_kind": "CubicSpline"}
    for c in ("A", "C")
    for nw in (0.01, 0.05, 0.20)
]

_IMPL_NAMES = {"new": "EEMD (native, pyemd=False)", "old": "EEMD (pyemd=True -> PyEMD)"}


# --------------------------------------------------------------------------- #
# 工具
# --------------------------------------------------------------------------- #
class _RssSampler:
    """按 1 ms 采样 RSS 峰值 (与 ``bench_timing.py`` 同口径)。"""

    def __init__(self, interval: float = 0.001):
        import psutil

        self._proc = psutil.Process()
        self._interval = interval
        self.baseline = int(self._proc.memory_info().rss)
        self.peak = self.baseline
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _loop(self):
        while not self._stop.is_set():
            try:
                self.peak = max(self.peak, int(self._proc.memory_info().rss))
            except Exception:
                pass
            self._stop.wait(self._interval)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=2.0)
        return False

    @property
    def delta_mb(self) -> float:
        return (self.peak - self.baseline) / 2 ** 20


def _repeat_count(n: int, repeats: int) -> int:
    """大尺寸自动降重复数: n>=65536 只跑 1 次 (预热 1 次仍保留)。"""
    return 1 if n >= 65536 else int(repeats)


def _impl_list(tag: str) -> list[str]:
    if tag == "both":
        return ["new", "old"]
    return [tag]


# --------------------------------------------------------------------------- #
# worker (子进程内执行一个格子)
# --------------------------------------------------------------------------- #
def _measure_cell(args: dict) -> dict:
    """在子进程内实测一个格子, 返回可序列化结果 dict。"""
    import numpy as np

    for p in (str(_SRC), str(_HERE)):
        if p not in sys.path:
            sys.path.insert(0, p)

    from signals import build_case, case_desc
    from quality import analyze

    from Modal_Decomposition import Class

    impl = args["impl"]
    case = args["case"]
    n = int(args["n"])
    trials = int(args["trials"])
    noise_width = float(args["noise_width"])
    spline_kind = args.get("spline_kind") or "CubicSpline"
    rich_info = bool(args.get("rich_info", False))
    seed = args.get("seed", _SEED)
    reps = int(args.get("repeats", 3))
    parallel = bool(args.get("parallel", False)) and impl == "new"

    S, modes = build_case(case, n)
    S0ref = S.copy()          # 诊断: 校验分解过程没有原地改动传入信号
    out = {
        "impl": impl,
        "impl_label": _IMPL_NAMES[impl],
        "case": case,
        "case_desc": case_desc(case),
        "n": n,
        "trials": trials,
        "noise_width": noise_width,
        "spline_kind": spline_kind,
        "spline_kind_pyemd": _SPLINE_TO_PYEMD.get(spline_kind),
        "rich_info": rich_info,
        "parallel": parallel,
        "seed": seed,
        "repeats": reps,
    }

    if impl == "new":
        factory = lambda: Class.EEMD(  # noqa: E731
            trials=trials,
            noise_width=noise_width,
            max_imf=-1,
            seed=seed,
            parallel=parallel,
            spline_kind=spline_kind,
            rich_info=rich_info,
        ).decompose(S)
    else:
        # 自 0.3.0 起 ``Class.EEMD`` 即原生实现; PyEMD 那一列改走过渡开关
        # ``pyemd=True`` (经 import_module 调 PyEMD)。
        if args.get("parallel"):
            out["parallel_downgraded"] = True
        factory = lambda: Class.EEMD(  # noqa: E731
            trials=trials,
            noise_width=noise_width,
            max_imf=-1,
            seed=seed,
            parallel=_OLD_PARALLEL,
            pyemd=True,
        ).decompose(S)

    # 预热 1 次 (首次 import / PyEMD EMD 对象构造不计入计时)
    t0 = time.perf_counter()
    warm = factory()
    out["warmup_s"] = time.perf_counter() - t0

    ts: list[float] = []
    aborted = None
    with _RssSampler() as sampler:
        for _ in range(reps):
            t0 = time.perf_counter()
            r = factory()
            dt = time.perf_counter() - t0
            ts.append(dt)
            if dt > 240.0 and reps > 1:
                # 单次就出格: 别让整格撞上 --time-cap-s 而丢掉全部已测数据。
                aborted = ("单次耗时 %.1fs 超过 240s 自中止阈值" % dt)
                break
    ts_arr = np.asarray(ts, dtype=np.float64)

    if aborted is not None:
        out["status"] = "aborted_slow"
        out["note"] = aborted
        out["t_all_s"] = [float(v) for v in ts_arr]
        out["t_median_s"] = None
        return out

    IMFs = np.asarray(r.IMFs, dtype=np.float64)
    if IMFs.ndim == 1:
        IMFs = IMFs.reshape(1, -1)
    Res = np.asarray(r.Res, dtype=np.float64)
    s_imf = IMFs.sum(axis=0) if IMFs.shape[0] else np.zeros_like(S)
    recon_arr = s_imf + Res
    recon_err = float(np.max(np.abs(recon_arr - S)))       # ΣIMFs + Res − S
    imf_sum_err = float(np.max(np.abs(s_imf - S)))         # ΣIMFs − S
    recon_rec = np.asarray(r.reconstruct(), dtype=np.float64)
    recon_rec_err = float(np.max(np.abs(recon_rec - S)))
    missing_residue = np.asarray(S, dtype=np.float64) - s_imf

    info = r.info or {}
    iters = info.get("iterations", None)
    iters_flat = [int(v) for lst in iters for v in lst] if iters else []

    # quality.analyze 的重构项只对"行"求和 (不含 Res), 故先取它的正交指数 /
    # 模式捕获两项; 重构类字段一律用本脚本口径 (含 Res) 覆盖。
    stack_all = IMFs
    q_all = analyze(S, stack_all, modes)
    analyze_recon = q_all.get("recon_max_abs_err")

    out["status"] = "ok"
    out["t_median_s"] = float(np.median(ts_arr))
    out["t_min_s"] = float(np.min(ts_arr))
    out["t_max_s"] = float(np.max(ts_arr))
    out["t_all_s"] = [float(v) for v in ts_arr]
    out["t_repeats_used"] = len(ts)
    out["rss_delta_mb"] = float(sampler.delta_mb)
    out["rss_peak_mb"] = sampler.peak / 2 ** 20
    out["n_rows"] = int(IMFs.shape[0])
    out["recon_max_abs_err"] = recon_err
    out["recon_exact"] = bool(recon_err <= 1e-9)
    out["imf_sum_max_abs_err"] = imf_sum_err
    out["res_abs_max"] = float(np.max(np.abs(Res))) if Res.size else float("nan")
    out["missing_residue_abs_max"] = float(np.max(np.abs(missing_residue)))
    out["missing_residue_rms"] = float(np.sqrt(np.mean(missing_residue ** 2)))
    out["missing_residue_rel"] = float(
        np.sqrt(np.mean(missing_residue ** 2)) / np.sqrt(np.mean(np.asarray(S) ** 2))
    )
    out["imf_abs_max"] = float(np.max(np.abs(IMFs))) if IMFs.size else float("nan")
    out["orthogonality_index"] = q_all.get("orthogonality_index")
    out["mode_recovery"] = q_all.get("mode_recovery")
    # 交叉校验: 自建 Σ+Res / reconstruct() / ΣIMFs 三者必须自洽 (1e-12 内)。
    out["xcheck_recon_err_from_reconstruct"] = recon_rec_err
    out["xcheck_recon_err_over_imf_rows"] = analyze_recon
    out["xcheck_rebuild_vs_reconstruct"] = float(np.max(np.abs(recon_arr - recon_rec)))
    out["xcheck_rebuild_vs_imfsum"] = float(np.max(np.abs(recon_arr - (s_imf + Res))))
    out["xcheck_max_diff"] = max(
        abs(recon_rec_err - recon_err),
        out["xcheck_rebuild_vs_reconstruct"],
        out["xcheck_rebuild_vs_imfsum"],
    )
    # Res 是否等于 S − ΣIMFs (新版 True / 旧版 False): Res 语义判别位。
    out["res_is_exact_remainder"] = bool(
        float(np.max(np.abs(Res - missing_residue))) <= 1e-9
    )
    # ``Res`` 语义。新版: ``Res = S − ΣIMFs`` (真实余量), 只有它能精确重构;
    # 旧版: ``Res`` 是 PyEMD 原始堆栈最后一行 IMF (逐行比对证据见报告), 故
    # 上方位为 False。缺陷证据是 ``recon_max_abs_err`` (~0.1) 与
    # ``missing_residue_*``, 不是这个布尔量本身。
    out["res_minus_true_residue_max"] = float(
        np.max(np.abs(Res - missing_residue))
    )

    # 平权口径 (前 k 行) 由驱动侧补齐 (``_fair_fill``): 单实现格先占位自检。
    out["cmp_rows"] = int(IMFs.shape[0])
    out["orthogonality_index_firstk"] = q_all.get("orthogonality_index")
    out["mode_recovery_firstk"] = q_all.get("mode_recovery")
    out["fair_basis"] = "self (no counterpart in run)"

    # info 诊断
    out["info_keys"] = sorted(info.keys())
    out["n_trials_effective"] = info.get("n_trials", None)
    out["workers"] = info.get("workers", None)
    out["iters_total"] = int(sum(iters_flat)) if iters_flat else None
    out["iters_mean"] = float(np.mean(iters_flat)) if iters_flat else None
    out["iters_max"] = int(max(iters_flat)) if iters_flat else None
    if iters:
        out["iters_per_order_mean"] = [float(np.mean(lst)) for lst in iters]
        out["iters_per_order_max"] = [int(max(lst)) for lst in iters]
        out["iters_per_order_n"] = [int(len(lst)) for lst in iters]
    if "ensemble_std" in info:
        es = np.asarray(info["ensemble_std"], dtype=np.float64)
        sig_rms = float(np.sqrt(np.mean(S ** 2)))
        out["ensemble_std_shape"] = list(es.shape)
        out["ensemble_std_row0_rms"] = float(np.sqrt(np.mean(es[0] ** 2))) if es.size else None
        out["ensemble_std_rms"] = float(np.sqrt(np.mean(es ** 2))) if es.size else None
        out["ensemble_std_rel"] = (
            out["ensemble_std_rms"] / sig_rms if sig_rms > 0 and es.size else None
        )
    if isinstance(warm.info, dict) and "ensemble_std" in (warm.info or {}):
        out["warmup_has_ensemble_std"] = True

    return out


def _worker(args: dict) -> int:
    """子进程入口: 任何异常都序列化成 status=error 行, 保证驱动侧能落地。"""
    try:
        out = _measure_cell(args)
    except Exception as exc:  # noqa: BLE001 - 需要把失败原因带回父进程
        import traceback
        out = {k: args.get(k) for k in
               ("impl", "case", "n", "trials", "noise_width", "spline_kind",
                "rich_info", "seed", "repeats")}
        out["status"] = "error"
        out["error"] = "%s: %s" % (type(exc).__name__, exc)
        out["traceback_tail"] = traceback.format_exc()[-1200:]
    print("@@RESULT@@" + json.dumps(out), flush=True)
    return 0


def _cell_meta(args: dict) -> dict:
    """格子标识字段 (可由键值参数完全重建, 故无需落盘整张信号)。"""
    keys = ("impl", "impl_label", "case", "n", "trials", "noise_width",
            "spline_kind", "spline_kind_pyemd", "rich_info", "seed", "repeats")
    m = {k: args.get(k) for k in keys}
    if not m.get("spline_kind_pyemd"):
        m["spline_kind_pyemd"] = _SPLINE_TO_PYEMD.get(m.get("spline_kind"))
    return m


def _weights_for(blob: dict, key: str, row: dict, S):
    """取该格的 IMF 行权重 (前 k 行比较用); 进程内缓存, 不落盘。"""
    import numpy as np

    memo = blob.setdefault("_w_memo", {})
    w = memo.get(key)
    if w is None:
        if row["impl"] == "new":
            from Modal_Decomposition import Class

            d = Class.EEMD(
                trials=int(row["trials"]), noise_width=float(row["noise_width"]),
                max_imf=-1, seed=row.get("seed", _SEED),
                spline_kind=row.get("spline_kind", "CubicSpline"),
            ).decompose(S)
        else:
            from Modal_Decomposition import Class

            d = Class.EEMD(
                trials=int(row["trials"]), noise_width=float(row["noise_width"]),
                max_imf=-1, seed=row.get("seed", _SEED), parallel=False,
            ).decompose(S)
        w = np.asarray(d.IMFs, dtype=np.float64)
        memo[key] = w
    return w


def _fair_fill(blob: dict, only_keys=None) -> None:
    """补平权 (前 k 行) 口径的质量指标 + 行数差: 驱动侧按可复现种子重算。

    主口径仍是各实现**自报的全部行** (``orthogonality_index`` /
    ``mode_recovery`` / ``recon_max_abs_err``); 这里额外填 ``*_firstk``:

    * ``cmp_rows = min(新版行数, 旧版行数)``, 两边都只取前 ``cmp_rows`` 行;
    * 意义: 旧版少报一阶, 用同阶数比较可把"少一阶"的影响从行级指标里摘掉,
      但它**同时**把新版真实余量 (Res, 可能不小) 排除在外 —— 因此
      ``recon_max_abs_err_firstk`` 两边都不接近 0 (新版缺余量、旧版缺一阶),
      这**不是**重构质量指标, 只是同阶数行级指标 (正交指数 / 模式捕获) 的口径。

    真正的重构指标是 ``recon_max_abs_err`` (含 Res): 新版 ~1e-17, 旧版 ~0.1。
    """
    import numpy as np

    for p in (str(_SRC), str(_HERE)):
        if p not in sys.path:
            sys.path.insert(0, p)
    from signals import build_case
    from quality import analyze

    cells = blob["cells"]
    groups: dict[tuple, dict[str, str]] = {}
    for key, row in cells.items():
        if only_keys is not None and key not in only_keys:
            continue
        if row.get("status") != "ok":
            continue
        gk = (row.get("case"), row.get("n"), row.get("trials"),
              row.get("noise_width"), row.get("spline_kind"))
        groups.setdefault(gk, {})[row.get("impl")] = key

    for gk in sorted(groups, key=lambda g: (str(g[0]), g[1] or 0, g[2] or 0, g[3] or 0)):
        pair = groups[gk]
        if "new" not in pair or "old" not in pair:
            for key in pair.values():
                if "fair_basis" not in cells[key]:
                    cells[key]["cmp_rows"] = cells[key].get("n_rows")
                    cells[key]["fair_basis"] = "self (no counterpart)"
            continue
        S, modes = build_case(gk[0], gk[1])
        row_new, row_old = cells[pair["new"]], cells[pair["old"]]
        k = min(int(row_new["n_rows"]), int(row_old["n_rows"]))
        if k <= 0:
            continue
        for key, row in ((pair["new"], row_new), (pair["old"], row_old)):
            w = _weights_for(blob, key, row, S)
            q = analyze(S, w[:k], modes)
            row["cmp_rows"] = k
            row["orthogonality_index_firstk"] = q.get("orthogonality_index")
            row["mode_recovery_firstk"] = q.get("mode_recovery")
            row["recon_max_abs_err_firstk"] = q.get("recon_max_abs_err")
            row["fair_basis"] = "first k rows of both sides (k=%d)" % k
        row_new["n_rows_gap_vs_old"] = int(row_new["n_rows"]) - int(row_old["n_rows"])


def run_cell(blob: dict, cell: dict, reps: int, time_cap_s: float) -> dict:
    """子进程跑一个格子; 超时/异常返回同名 status 行 (不中断整轮)。"""
    import psutil  # type: ignore
    del blob  # 仅供签名一致 (实测全部在子进程内完成)
    payload = json.dumps(cell)
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(_SRC), str(_HERE)])
    env["PYTHONIOENCODING"] = "utf-8"
    proc = subprocess.Popen(
        [sys.executable, "-B", str(Path(__file__).resolve()), "--worker", payload],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        encoding="utf-8", env=env,
    )
    t0 = time.perf_counter()
    verdict = None
    while proc.poll() is None:
        if time.perf_counter() - t0 > time_cap_s:
            verdict = "timeout"
            break
        time.sleep(0.05)
    if verdict:
        try:
            proc.kill()
        except Exception:
            pass
        proc.wait(timeout=30)
        out = {k: cell.get(k) for k in
               ("impl", "case", "n", "trials", "noise_width", "spline_kind", "seed")}
        out["status"] = f"timeout>{time_cap_s:.0f}s"
        out["repeats"] = reps
        return out
    stdout, stderr = proc.communicate()
    out = None
    for line in stdout.splitlines():
        if line.startswith("@@RESULT@@"):
            out = json.loads(line[len("@@RESULT@@"):])
    if out is None:
        out = {k: cell.get(k) for k in
               ("impl", "case", "n", "trials", "noise_width", "spline_kind", "seed")}
        out["status"] = "error"
        out["repeats"] = reps
        out["stderr_tail"] = (stderr or "")[-600:]
    out["wall_s"] = time.perf_counter() - t0
    return out


# --------------------------------------------------------------------------- #
# 计划 / 合并
# --------------------------------------------------------------------------- #
def cell_key(impl: str, case: str, n: int, trials: int, nw: float,
             kind: str, rich: bool) -> str:
    return f"{impl}|{case}|{n}|{trials}|{nw:.3g}|{kind}|{'rich' if rich else 'plain'}"


def build_plan(sweeps: list[str], impls: list[str], cases: list[str],
               rich: bool) -> tuple[dict[str, dict], dict[str, list[str]]]:
    """返回 (cells_by_key, sweep -> [cell_key, ...]); cell 内记 plan_keys。"""
    cells: dict[str, dict] = {}
    order: dict[str, list[str]] = {}
    for sw in sweeps:
        order[sw] = []
        for tag, case, n, trials, nw, kind, est in _SWEEPS[sw]["cells"]:
            if case not in cases:
                continue
            if n >= 65536 and "--allow-65536" not in sys.argv and sw != "headtohead":
                continue
            for impl in _impl_list(tag):
                if impl not in impls:
                    continue
                key = cell_key(impl, case, n, trials, nw, kind, False)
                if key not in cells:
                    cells[key] = dict(
                        impl=impl, case=case, n=n, trials=trials, noise_width=nw,
                        spline_kind=kind, rich_info=False, est_s=est,
                        seed=_SEED, plan_keys={},
                    )
                cells[key]["plan_keys"].setdefault(sw, key)
                if key not in order[sw]:
                    order[sw].append(key)
        if sw == "noise_sweep" and rich and "new" in impls:
            for spec in _RICH_CELLS:
                if spec["case"] not in cases:
                    continue
                key = cell_key("new", spec["case"], spec["n"], spec["trials"],
                               spec["noise_width"], spec["spline_kind"], True)
                if key not in cells:
                    cells[key] = dict(
                        impl="new", **spec, rich_info=True,
                        est_s=1.6 if spec["case"] == "A" else 2.0,
                        seed=_SEED, plan_keys={},
                    )
                cells[key]["plan_keys"].setdefault(sw, key)
                if key not in order[sw]:
                    order[sw].append(key)
    return cells, order


def load_raw() -> dict:
    if _RAW_JSON.exists():
        try:
            blob = json.loads(_RAW_JSON.read_text(encoding="utf-8"))
            if isinstance(blob, dict) and "cells" in blob:
                return blob
        except Exception as exc:  # pragma: no cover - 只提示, 不炸
            print(f"[warn] 无法解析既有 {_RAW_JSON}: {exc!r}", flush=True)
    return {"cells": {}}


def save_raw(blob: dict) -> None:
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": blob.get("meta", {}),
        "cell_count": len(blob["cells"]),
        "cells": blob["cells"],
    }
    _RAW_JSON.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def bind_sweeps(blob: dict, plan: dict[str, dict]) -> None:
    """把 sweep 归属写进已有行 (跨轮次复用时不丢字段)。"""
    for key, cell in plan.items():
        row = blob["cells"].get(key)
        if row is None:
            continue
        tags = row.setdefault("sweeps", [])
        for sw in cell.get("plan_keys", {}):
            if sw not in tags:
                tags.append(sw)


# --------------------------------------------------------------------------- #
# 打印
# --------------------------------------------------------------------------- #
def _fmt(v, spec="%.4g"):
    if v is None:
        return "—"
    try:
        if isinstance(v, float) and v != v:
            return "—"
        return spec % v
    except Exception:
        return str(v)


def _is_pair_group(cells: dict, keys: list[str]) -> bool:
    """该组内是否同时存在同 (case,n,trials,nw,spline) 的 new 与 old 行。"""
    sig = set()
    for key in keys:
        c = cells.get(key)
        if c is None:
            continue
        sig.add((c.get("case"), c.get("n"), c.get("trials"),
                 c.get("noise_width"), c.get("spline_kind"), c.get("impl")))
    for s in sig:
        if s[-1] == "new" and s[:-1] + ("old",) in sig:
            return True
    return False


def print_pair_table(blob: dict, cells_plan: dict, keys: list[str], title: str) -> None:
    """成对打印 new/old 及其同格比值 (比值只在同格内计算)。"""
    cells = blob["cells"]
    print(f"\n### {title}\n", flush=True)
    print("| case | n | trials | noise_width | impl | t 中位(s) | reps | 行数 | "
          "重构误差 | 行和误差 | 缺失余量RMS | 正交指数 | RSSΔ(MB) | 比值 old/new |",
          flush=True)
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|", flush=True)
    # 按 (case, n, trials, nw) 分组
    groups: dict[tuple, dict[str, dict]] = {}
    for key in keys:
        row = cells.get(key)
        if row is None:
            continue
        gk = (row.get("case"), row.get("n"), row.get("trials"), row.get("noise_width"))
        groups.setdefault(gk, {})[row.get("impl")] = row
    ratios: list[float] = []
    for gk in sorted(groups, key=lambda g: (str(g[0]), g[1] or 0, g[2] or 0, g[3] or 0)):
        grp = groups[gk]
        t_new = (grp.get("new") or {}).get("t_median_s")
        t_old = (grp.get("old") or {}).get("t_median_s")
        ratio = (t_old / t_new) if (t_new and t_old) else None
        for impl in ("new", "old"):
            row = grp.get(impl)
            if row is None:
                continue
            if ratio is not None and impl == "new":
                ratios.append(ratio)
            print("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
                gk[0], gk[1], gk[2], gk[3], impl,
                _fmt(row.get("t_median_s"), "%.3f"), row.get("repeats", "—"),
                row.get("n_rows", "—"),
                _fmt(row.get("recon_max_abs_err"), "%.2e"),
                _fmt(row.get("imf_sum_max_abs_err"), "%.2e"),
                _fmt(row.get("missing_residue_rms"), "%.2e"),
                _fmt(row.get("orthogonality_index"), "%.4f"),
                _fmt(row.get("rss_delta_mb"), "%.1f"),
                _fmt(ratio, "%.2f") if impl == "new" else "—",
            ), flush=True)
    if ratios:
        print(f"\n- 同格比值 old/new (n={len(ratios)} 格): "
              f"min {min(ratios):.2f}x, 中位 {sorted(ratios)[len(ratios)//2]:.2f}x, "
              f"max {max(ratios):.2f}x", flush=True)


def print_recovery_table(blob: dict, cells_plan: dict, keys: list[str], title: str) -> None:
    """模式捕获 (真值分量): 自报全行 + 平权前 k 行两个口径。"""
    cells = blob["cells"]
    rows = [(k, cells[k]) for k in keys
            if k in cells and (cells[k].get("mode_recovery") or cells[k].get("mode_recovery_firstk"))]
    if not rows:
        return
    names = sorted({n for _, r in rows
                    for src in ("mode_recovery", "mode_recovery_firstk")
                    for n in (r.get(src) or {})})
    print(f"\n### {title} (模式捕获)\n", flush=True)
    print("| case | n | trials | noise_width | impl | 口径 | " +
          " | ".join(f"corr({n})" for n in names) + " |", flush=True)
    print("|---|---|---|---|---|---|" + "---|" * len(names), flush=True)
    for key, row in sorted(rows, key=lambda kr: (
            str(kr[1].get("case")), kr[1].get("n") or 0, kr[1].get("trials") or 0,
            kr[1].get("noise_width") or 0, str(kr[1].get("impl")))):
        for src, basis in (("mode_recovery", "全部行"),
                           ("mode_recovery_firstk", "前 k=%s 行" % row.get("cmp_rows"))):
            rec = row.get(src) or {}
            if not rec:
                continue
            cells_txt = " | ".join(
                _fmt((rec.get(n) or {}).get("best_abs_corr"), "%.4f") for n in names)
            print("| %s | %s | %s | %s | %s | %s | %s |" % (
                row.get("case"), row.get("n"), row.get("trials"),
                row.get("noise_width"), row.get("impl"), basis, cells_txt), flush=True)


def print_single_table(blob: dict, cells_plan: dict, keys: list[str], title: str,
                       self_groups: list | None = None) -> None:
    cells = blob["cells"]
    print(f"\n### {title}\n", flush=True)
    print("| case | n | trials | noise_width | impl | spline | t 中位(s) | 行数 | "
          "重构误差 | 行和误差 | 正交指数 | 迭代合计 | 迭代均值 | 迭代最大 |", flush=True)
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|", flush=True)
    for key in keys:
        row = cells.get(key)
        if row is None:
            continue
        iters_total = row.get("iters_total")
        print("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            row.get("case"), row.get("n"), row.get("trials"), row.get("noise_width"),
            row.get("impl"), row.get("spline_kind"),
            _fmt(row.get("t_median_s"), "%.3f"),
            row.get("n_rows", "—"), _fmt(row.get("recon_max_abs_err"), "%.2e"),
            _fmt(row.get("imf_sum_max_abs_err"), "%.2e"),
            _fmt(row.get("orthogonality_index"), "%.4f"),
            iters_total if iters_total is not None else "—",
            _fmt(row.get("iters_mean"), "%.1f"),
            row.get("iters_max", "—"),
        ), flush=True)

    # 新版包络档自比 (同一 case/n/trials/nw 下三种 spline)
    if not self_groups:
        return
    print("\n| case | n | case 内基准档 | PCHIP / 基准 | linear / 基准 | 基准 t(s) |",
          flush=True)
    print("|---|---|---|---|---|---|", flush=True)
    for case, n, trials, nw in self_groups:
        base = cells.get(cell_key("new", case, n, trials, nw, "CubicSpline", False))
        if not base or base.get("status") != "ok":
            continue
        tb = base.get("t_median_s")
        out = []
        for kind in ("PCHIP", "linear"):
            r = cells.get(cell_key("new", case, n, trials, nw, kind, False))
            out.append(_fmt((r or {}).get("t_median_s", None) and
                            r["t_median_s"] / tb, "%.2f"))
        print("| %s | %d | CubicSpline | %s | %s | %.3f |" % (
            case, n, out[0], out[1], tb), flush=True)


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description="EEMD 原生 vs PyEMD 包装 对比基准")
    ap.add_argument("--worker", default=None, help="内部: 子进程单格入口 (JSON)")
    ap.add_argument("--sweeps", nargs="*", default=["headtohead"],
                    choices=["headtohead", "trials_sweep", "noise_sweep",
                             "spline_sweep", "all"])
    ap.add_argument("--impls", nargs="*", default=["new", "old"],
                    choices=["new", "old"])
    ap.add_argument("--cases", nargs="*", default=["A", "B", "C"],
                    choices=["A", "B", "C"])
    ap.add_argument("--repeats", type=int, default=3, help="预热 1 次后的计时重复数")
    ap.add_argument("--time-cap-s", type=float, default=420.0, help="单格墙钟上限")
    ap.add_argument("--budget-s", type=float, default=1800.0,
                    help="本轮预算 (仅作提示与中止闸门, 默认 30 min)")
    ap.add_argument("--no-rich", action="store_true", help="跳过 rich_info 诊断格")
    ap.add_argument("--force", action="store_true", help="重跑已存在的格子")
    ap.add_argument("--allow-65536", action="store_true",
                    help="允许非 headtohead sweep 使用 n=65536 (默认关闭)")
    ap.add_argument("--estimate", action="store_true",
                    help="只打印计划与估算预算, 不跑任何分解")
    args = ap.parse_args()

    if args.worker:
        return _worker(json.loads(args.worker))

    sweeps = list(args.sweeps)
    if "all" in sweeps:
        sweeps = ["headtohead", "trials_sweep", "noise_sweep", "spline_sweep"]
    if "old" in args.impls and args.impls == ["old"] and "spline_sweep" in sweeps:
        print("[warn] spline_sweep 只定义新版格; 已忽略 --impls old", flush=True)
        sweeps = [s for s in sweeps if s != "spline_sweep"]

    cells, order = build_plan(sweeps, args.impls, args.cases, not args.no_rich)
    blob = load_raw()
    blob["_w_memo"] = {}
    blob.setdefault("meta", {})
    import numpy as np
    blob["meta"].update({
        "script": "tests/comparison/bench_eemd_native.py",
        "report": "docs/EEMD_Native_vs_PyEMD_Report.md",
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "seed_default": _SEED,
        "old_parallel_constant": _OLD_PARALLEL,
        "old_parallel_sent_to_pyemd": _OLD_PARALLEL,
        "old_parallel_note": "旧列一律以 parallel=False 调 PyEMD (覆盖其默认 True)",
        "new_parallel_unverified": True,
        "spline_map": _SPLINE_TO_PYEMD,
    })
    try:
        import scipy
        blob["meta"]["scipy"] = scipy.__version__
    except Exception:
        pass
    try:
        import PyEMD
        blob["meta"]["pyemd"] = getattr(PyEMD, "__version__", "?")
    except Exception:
        pass

    print(f"# EEMD 对比基准 · sweeps={sweeps} · impls={args.impls} · "
          f"cases={args.cases} · repeats={args.repeats}", flush=True)
    print(f"# 计划 {len(cells)} 格 (含跨 sweep 复用); 预算 {args.budget_s:.0f}s", flush=True)

    if args.estimate:
        total_est = 0.0
        print("\n| sweep | case | n | trials | noise_width | impl | spline | 估算(s) |",
              flush=True)
        print("|---|---|---|---|---|---|---|---|", flush=True)
        for sw in sweeps:
            for key in order.get(sw, []):
                c = cells[key]
                done_ok = (not args.force
                           and blob["cells"].get(key, {}).get("status") == "ok")
                est = 0.0 if done_ok else float(c["est_s"])
                total_est += est
                print("| %s | %s | %s | %s | %s | %s | %s | %s |" % (
                    sw, c["case"], c["n"], c["trials"], c["noise_width"], c["impl"],
                    c["spline_kind"], ("已算" if done_ok else "%.1f" % est)), flush=True)
        print(f"\n# 估算总耗时 {total_est:.0f}s ({total_est / 60:.1f} min); "
              f"预算 {args.budget_s:.0f}s", flush=True)
        return 0

    t_start = time.perf_counter()
    done = skipped = 0
    total = len(cells)
    for sw in sweeps:
        keys = order.get(sw, [])
        if not keys:
            continue
        pending = [k for k in keys
                   if args.force or blob["cells"].get(k, {}).get("status") != "ok"]
        print(f"\n===== sweep {sw}: {_SWEEPS[sw]['desc']} "
              f"({len(keys)} 格, 待算 {len(pending)}) =====", flush=True)
        for key in keys:
            todo = cells[key]
            est_tot = float(todo["est_s"])
            used = time.perf_counter() - t_start
            if used + est_tot > args.budget_s:
                print(f"[budget] 跳过 {key} (已用 {used:.0f}s, 估算还需 "
                      f"{est_tot:.0f}s > 预算 {args.budget_s:.0f}s)", flush=True)
                skipped += 1
                continue
            if key in blob["cells"] and not args.force:
                row = blob["cells"][key]
                if row.get("status") == "ok":
                    print(f"[cached] {key} t={_fmt(row.get('t_median_s'), '%.3f')}s",
                          flush=True)
                    done += 1
                    continue
                print(f"[retry] {key} (此前 status={row.get('status')})", flush=True)
            reps_needed = _repeat_count(todo["n"], args.repeats)
            cell = {k: v for k, v in todo.items()
                    if k in ("impl", "case", "n", "trials", "noise_width",
                             "spline_kind", "rich_info", "seed")}
            cell["repeats"] = reps_needed
            cell["worker"] = str(Path(__file__).resolve())
            t0 = time.perf_counter()
            row = run_cell(blob, cell, reps_needed, args.time_cap_s)
            row["est_budget_s"] = todo["est_s"]
            blob["cells"][key] = row
            if row.get("status") == "ok":
                # 平权口径 (前 k 行) 需要对手行数, 因此每次落盘前统一补齐。
                try:
                    _fair_fill(blob)
                except Exception as exc:  # noqa: BLE001 - 诊断失败不阻断测量
                    print(f"[warn] fair_fill 失败: {exc!r}", flush=True)
            save_raw(blob)          # 崩了也留下已完成的部分
            dt = time.perf_counter() - t0
            print(f"[{done + 1}/{total}] {key} status={row.get('status')} "
                  f"t={_fmt(row.get('t_median_s'), '%.3f')}s rows={row.get('n_rows', '—')} "
                  f"recon={_fmt(row.get('recon_max_abs_err'), '%.2e')} "
                  f"wall={dt:.1f}s", flush=True)
            done += 1

    cells, order = build_plan(sweeps, args.impls, args.cases, not args.no_rich)
    bind_sweeps(blob, cells)
    try:
        _fair_fill(blob)
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] fair_fill 失败: {exc!r}", flush=True)
    save_raw(blob)
    print(f"\n# 本轮完成 {done} 格, 跳过 {skipped} 格, 总耗时 "
          f"{time.perf_counter() - t_start:.0f}s", flush=True)

    # ---- 打印表 (按 sweep 分组) ----
    for sw in sweeps:
        keys = order.get(sw, [])
        if not keys:
            continue
        print(f"\n<!-- sweep:{sw} -->", flush=True)
        if _is_pair_group(cells, keys):
            print_pair_table(blob, cells, keys, f"{sw}: {_SWEEPS[sw]['desc']}")
        else:
            print_single_table(blob, cells, keys, f"{sw}: {_SWEEPS[sw]['desc']}",
                               _SWEEPS[sw].get("self_groups"))
        print_recovery_table(blob, cells, keys, f"{sw}: {_SWEEPS[sw]['desc']}")

    # ---- 缺陷证据 + 总计 ----
    ok = [r for r in blob["cells"].values() if r.get("status") == "ok"]
    old_ok = [r for r in ok if r.get("impl") == "old"]
    new_ok = [r for r in ok if r.get("impl") == "new"]
    print("\n### 残差语义自检 (res_is_exact_remainder) 与缺陷代价\n", flush=True)
    print("| impl | 格数 | res==S−ΣIMFs (res_is_exact_remainder) | 重构误差范围 | "
          "行和误差范围 | "
          "缺失余量 RMS 范围 | 相对 RMS |", flush=True)
    print("|---|---|---|---|---|---|---|", flush=True)
    for name, rows in (("new", new_ok), ("old", old_ok)):
        if not rows:
            continue
        flag = [bool(r.get("res_is_exact_remainder")) for r in rows]
        e = [r["recon_max_abs_err"] for r in rows if r.get("recon_max_abs_err") is not None]
        s = [r["imf_sum_max_abs_err"] for r in rows
             if r.get("imf_sum_max_abs_err") is not None]
        mr = [r["missing_residue_rms"] for r in rows
              if r.get("missing_residue_rms") is not None]
        mq = [r["missing_residue_rel"] for r in rows
              if r.get("missing_residue_rel") is not None]
        print("| %s | %d | %d/%d | %.2e – %.2e | %.2e – %.2e | %.2e – %.2e | "
              "%.4f – %.4f |" % (
                  name, len(rows), sum(flag), len(flag),
                  min(e), max(e), min(s), max(s), min(mr), max(mr),
                  min(mq), max(mq)), flush=True)
    gaps = [r.get("n_rows_gap_vs_old") for r in new_ok
            if r.get("n_rows_gap_vs_old") is not None]
    if gaps:
        print(f"\n- 成对格的 (新版行数 − 旧版行数): {sorted(set(gaps))} "
              f"(>0 ⇒ 旧版少报阶数)", flush=True)
    print(f"\nraw results ({len(blob['cells'])} cells) -> {_RAW_JSON}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
