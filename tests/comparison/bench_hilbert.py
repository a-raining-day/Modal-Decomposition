"""Hilbert 后端基准 (scipy vs FFT vs 自适应 FHT vs 纯 SAO 内核)。

产出 ``docs/test/Hilbert_Backend_Results.json`` —— 与
``docs/test/Hilbert_Backend_Report.md`` 一一对应。五个被测对象见
``_analytic()``; 简述:

* ``scipy``      : ``scipy.signal.hilbert``, 基线 (两次复数 FFT);
* ``fft``        : ``hilbert(S, "FFT")``, rfft/irfft 实半谱 (两次实变换);
* ``fht``        : ``hilbert(S, "FHT")``, **自适应**: 2 的幂且 n <= 512 走
                   SAO C 内核, 其余走实半谱 (默认后端);
* ``fht-kernel`` : ``hilbert_transform(S)``, 纯 SAO C 内核 (零填充到 2 的幂),
                   用来单独量化填充伪影;
* ``fht-pure``   : 编译内核缺失时的纯 NumPy 参考实现 (同零填充 / 相位约定)。

计时口径
--------
每条记录 = 同进程内 ``reps`` 次独立调用的 **best**(min) 与 median;
所有调用前先跑一次 warm-up, 避免首次 import / 页错误污染。

内存口径
--------
``tracemalloc`` 的峰值增量(已实测能看到 NumPy 的缓冲区分配;
1 GB 输入 ``rfft`` 的峰值与 ``bench_fft.py`` 的提交内存同量级)。
它只覆盖 Python/NumPy 分配器, 不含 C 内核里的 ``malloc`` ——
FHT 原生内核用 ``np.zeros`` 申请缓冲区, 所以仍在统计范围内。

准确性口径
----------
参考实现 = ``scipy.signal.hilbert``(同 MATLAB 相位约定)。记录
``max|z_ref - z|``、相对 L2、包络最大相对误差, 并把误差按"边缘 1%"与"中段"
分开统计 —— 零填充伪影在边界最大、向中段衰减, 必须区分。

用法
----
    python tests/comparison/bench_hilbert.py                     # 默认全量
    python tests/comparison/bench_hilbert.py --sizes 256 4096 65536
    python tests/comparison/bench_hilbert.py --configs scipy fft fht
    python tests/comparison/bench_hilbert.py --out docs/test/Hilbert_Backend_Results.json
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import sys
import time
import tracemalloc
from datetime import datetime

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.Modal_Decomposition.Utils.Hilbert import (  # noqa: E402
    fht_forward,
    fht_inverse,
    hilbert,
    hilbert_transform,
)
from src.Modal_Decomposition.Utils._Hilbert import _fht  # noqa: E402

ALL_CONFIGS = ("scipy", "fft", "fht", "fht-kernel", "fht-pure")
DEFAULT_SIZES = (256, 1024, 4096, 16384, 65536, 262144, 1048576, 4194304)
MEM_SIZES = (1024, 65536, 1048576, 4194304)


def _analytic(config: str, x: np.ndarray) -> np.ndarray:
    """按后端求解析信号(统一 MATLAB 相位约定)。

    * ``scipy``      — ``hilbert(x, "scipy")``, 基线;
    * ``fft``        — ``hilbert(x, "FFT")``, rfft/irfft 实半谱;
    * ``fht``        — ``hilbert(x, "FHT")``, 自适应 (小 2 的幂走 SAO 内核,
                       其余走实半谱);
    * ``fht-kernel`` — ``hilbert_transform(x, matlab_phase=True)``, 纯 SAO
                       内核 (会零填充, 用来量化填充伪影);
    * ``fht-pure``   — ``_fht._hilbert_transform_pure``, 无编译器时的参考实现。
    """
    if config == "scipy":
        return hilbert(x, "scipy")
    if config == "fft":
        return hilbert(x, "FFT")
    if config == "fht":
        return hilbert(x, "FHT")
    if config == "fht-kernel":
        return hilbert_transform(x, matlab_phase=True)
    if config == "fht-pure":
        return _fht._hilbert_transform_pure(x, matlab_phase=True)
    raise ValueError(f"unknown config {config!r}")


def _reps_for(n: int) -> int:
    """小尺寸多次、大尺寸少量, 让每条记录的耗时大致可比。"""
    return int(np.clip(2e7 // max(n, 1), 3, 50))


def _time_call(fn, reps: int) -> tuple[float, float]:
    """返回 (best_s, median_s); 调用方负责 warm-up。"""
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return float(np.min(ts)), float(np.median(ts))


def _peak_mb(fn) -> float:
    """单次调用的 tracemalloc 峰值增量 (MB)。"""
    gc.collect()
    tracemalloc.start()
    tracemalloc.reset_peak()
    base = tracemalloc.get_traced_memory()[0]
    fn()
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    return float((peak - base) / 2**20)


# --------------------------------------------------------------------------- #
# 1. 正确性
# --------------------------------------------------------------------------- #
def correctness(sizes, non_pow2_sizes) -> list[dict]:
    import scipy.signal as ss

    out = []
    rng = np.random.default_rng(20261018)
    for n in list(sizes) + list(non_pow2_sizes):
        x = rng.standard_normal(n)
        ref = ss.hilbert(x)
        edge = max(1, n // 100)
        for cfg in ALL_CONFIGS:
            z = _analytic(cfg, x)
            err = np.abs(z - ref)
            env_ref = np.abs(ref)
            env_err = np.abs(np.abs(z) - env_ref) / np.maximum(env_ref, 1e-12)
            out.append({
                "config": cfg,
                "n": int(n),
                "is_pow2": bool(n & (n - 1) == 0),
                "max_abs_err": float(err.max()),
                "rel_l2_err": float(np.linalg.norm(z - ref) / max(np.linalg.norm(ref), 1e-30)),
                "mid_max_abs_err": float(err[edge:n - edge].max()) if n > 2 * edge else 0.0,
                "edge_max_abs_err": float(max(err[:edge].max(), err[-edge:].max())),
                "max_env_rel_err": float(env_err.max()),
                "edge_frac_err_gt_1e6": float(
                    np.mean(np.r_[err[:edge], err[-edge:]] > 1e-6)),
                "mid_frac_err_gt_1e6": float(
                    np.mean(err[edge:n - edge] > 1e-6)) if n > 2 * edge else 0.0,
            })
    return out


# --------------------------------------------------------------------------- #
# 2. 计时 / 内存
# --------------------------------------------------------------------------- #
def timing(sizes, configs) -> list[dict]:
    out = []
    rng = np.random.default_rng(7)
    for n in sizes:
        x = rng.standard_normal(n)
        reps = _reps_for(n)
        for cfg in configs:
            _analytic(cfg, x)  # warm-up
            for op, fn in (
                ("analytic", lambda c=cfg, xx=x: _analytic(c, xx)),
                ("envelope", lambda c=cfg, xx=x: np.abs(_analytic(c, xx))),
            ):
                best, med = _time_call(fn, reps)
                out.append({
                    "config": cfg, "n": int(n), "op": op, "reps": reps,
                    "best_s": best, "median_s": med,
                    "best_us": best * 1e6, "median_us": med * 1e6,
                })
    return out


def memory(sizes) -> list[dict]:
    out = []
    rng = np.random.default_rng(11)
    for n in sizes:
        x = rng.standard_normal(n)
        for cfg in ALL_CONFIGS:
            for op, fn in (
                ("analytic", lambda c=cfg, xx=x: _analytic(c, xx)),
                ("envelope", lambda c=cfg, xx=x: np.abs(_analytic(c, xx))),
            ):
                fn()  # warm-up (分配器/页面)
                out.append({"config": cfg, "n": int(n), "op": op,
                            "peak_mb": _peak_mb(fn)})
    return out


def fht_primitives(sizes) -> list[dict]:
    """FHT 原语: 前向/逆变换, 原生 C 内核 vs 纯 NumPy 参考实现。

    前向变体的参考 = 纯 NumPy 前向; 逆变体的参考 = 由前向结果还原出的原信号
    (2 的幂长度时应逐点还原), 并额外记录往返误差。
    """
    out = []
    rng = np.random.default_rng(3)
    for n in sizes:
        x = rng.standard_normal(n)
        reps = _reps_for(n)
        fwd_ref = _fht._fht_forward_pure(x)
        variants = {
            "fht-forward-native": (lambda xx=x: fht_forward(xx), fwd_ref, "forward"),
            "fht-forward-pure": (lambda xx=x: _fht._fht_forward_pure(xx), fwd_ref, "forward"),
            "fht-inverse-native": (lambda xx=x: fht_inverse(fwd_ref), x, "inverse"),
            "fht-inverse-pure": (lambda xx=x: _fht._fht_inverse_pure(fwd_ref), x, "inverse"),
        }
        for name, (fn, ref, kind) in variants.items():
            fn()
            best, med = _time_call(fn, reps)
            got = np.asarray(fn())
            if kind == "forward":
                roundtrip = float(np.max(np.abs(fht_inverse(got) - x))) \
                    if got.size == x.size else float("nan")
            else:
                roundtrip = float(np.max(np.abs(got - x))) if got.size == x.size else float("nan")
            out.append({
                "variant": name, "n": int(n), "reps": reps,
                "best_us": best * 1e6, "median_us": med * 1e6,
                "n_out": int(got.size),
                "max_abs_err_vs_ref": float(np.max(np.abs(got - ref)))
                if got.size == ref.size else float("nan"),
                "roundtrip_max_abs_err": roundtrip,
            })
    return out


# --------------------------------------------------------------------------- #
# 3. 效果 (解调质量)
# --------------------------------------------------------------------------- #
def _quadrature_metrics(z: np.ndarray, phase_true: np.ndarray,
                        amp_true: np.ndarray, trim: float = 0.05) -> dict:
    """对比真实相位/幅度: 幅度纹波、相位误差、瞬时频率误差。

    ``trim`` 比例的两端被剔除 —— 解析信号在序列端点本来就有边界效应
    (scipy 同样有), 不剔除会把"方法差异"和"边界效应"混在一起。
    """
    n = z.size
    k = int(n * trim)
    sl = slice(k, n - k) if n - 2 * k > 8 else slice(0, n)
    amp = np.abs(z)
    amp_rel = np.abs(amp - amp_true) / np.maximum(amp_true, 1e-12)
    phase = np.unwrap(np.angle(z))
    dphi = phase - phase_true
    dphi = dphi - np.mean(dphi)          # 解析信号相位允许有常值偏置
    if_err = (np.diff(phase) - np.diff(phase_true)) / (2 * np.pi)
    return {
        "amp_ripple_max": float(amp_rel[sl].max()),
        "amp_ripple_rms": float(np.sqrt(np.mean(amp_rel[sl] ** 2))),
        "phase_err_max_rad": float(np.abs(dphi[sl]).max()),
        "phase_err_rms_rad": float(np.sqrt(np.mean(dphi[sl] ** 2))),
        "if_err_rms": float(np.sqrt(np.mean(if_err[sl] ** 2))),
        "if_err_max": float(np.abs(if_err[sl]).max()),
    }


def _vs_scipy(z: np.ndarray, ref: np.ndarray) -> dict:
    """与 scipy 参考的实现差异(全段 / 边缘 1% / 中段)。"""
    err = np.abs(z - ref)
    n = err.size
    edge = max(1, n // 100)
    return {
        "max_abs_diff_vs_scipy": float(err.max()),
        "edge_abs_diff_vs_scipy": float(max(err[:edge].max(), err[-edge:].max())),
        "mid_abs_diff_vs_scipy": float(err[edge:n - edge].max()) if n > 2 * edge else 0.0,
        "env_rel_diff_vs_scipy": float(np.max(
            np.abs(np.abs(z) - np.abs(ref)) / np.maximum(np.abs(ref), 1e-12))),
    }


def effect(sizes_pow2, sizes_non_pow2) -> dict:
    rng = np.random.default_rng(42)
    cases = {}

    def tone(n, cycles):
        ph = 2 * np.pi * cycles * np.arange(n, dtype=np.float64) / n
        return np.sin(ph), ph, np.ones(n)

    def am(n, fc, fm, depth=0.5):
        t = np.arange(n, dtype=np.float64)
        ph = 2 * np.pi * fc * t / n
        env = 1.0 + depth * np.cos(2 * np.pi * fm * t / n)
        return env * np.cos(ph), ph, env

    def mixed(n):
        """两音 + 噪声 (LMD 里真实的宽带输入), 无解析真值。"""
        t = np.arange(n, dtype=np.float64)
        ph = 2 * np.pi * (n // 64) * t / n
        x = (np.sin(ph) + 0.4 * np.sin(2 * np.pi * (n // 9) * t / n)
             + 0.05 * rng.standard_normal(n))
        return x, ph, np.ones(n)

    specs = [("tone", tone), ("am", am), ("mixed", mixed)]
    for name, gen in specs:
        rows = []
        for n in sizes_pow2:
            x, ph, amp = gen(n) if name == "mixed" else (
                gen(n, n // 64) if name == "tone" else gen(n, n // 8, n // 512))
            ref = _analytic("scipy", x)
            for cfg in ALL_CONFIGS:
                z = _analytic(cfg, x)
                m = _quadrature_metrics(z, ph, amp)
                m.update(_vs_scipy(z, ref))
                m.update({"config": cfg, "n": int(n)})
                rows.append(m)
        cases[name] = rows

    # 非 2 的幂: 同一正弦, 量化零填充边界伪影
    rows = []
    for n in sizes_non_pow2:
        ph = 2 * np.pi * (n // 64) * np.arange(n, dtype=np.float64) / n
        x = np.sin(ph)
        ref = _analytic("scipy", x)
        for cfg in ALL_CONFIGS:
            z = _analytic(cfg, x)
            m = _quadrature_metrics(z, ph, np.ones(n))
            m.update(_vs_scipy(z, ref))
            m.update({"config": cfg, "n": int(n)})
            rows.append(m)
    cases["tone_non_pow2"] = rows
    return cases


# --------------------------------------------------------------------------- #
# 4. 规模扫描: 误差随"零填充比例"的关系
# --------------------------------------------------------------------------- #
def pad_scan() -> list[dict]:
    """在 2 的幂基础上逐步拉长 n, 量化 FHT 零填充引入的边界伪影宽度。"""
    rows = []
    rng = np.random.default_rng(99)
    for base in (1024, 8192):
        for extra in (0, 1, 16, 128, 512, 1024):
            n = base + extra
            x = rng.standard_normal(n)
            ref = _analytic("scipy", x)
            z = _analytic("fht-kernel", x)
            err = np.abs(z - ref)
            edge = max(1, n // 100)
            bad_1e6 = np.flatnonzero(err > 1e-6)
            rows.append({
                "n": int(n), "base": base, "pad_ratio": float(n / base - 1.0),
                "max_abs_err": float(err.max()),
                "rel_l2_err": float(np.linalg.norm(z - ref) / max(np.linalg.norm(ref), 1e-30)),
                "edge_max_abs_err": float(max(err[:edge].max(), err[-edge:].max())),
                "mid_max_abs_err": float(err[edge:n - edge].max()),
                "frac_err_gt_1e6": float(np.mean(err > 1e-6)),
                "frac_err_gt_1e3": float(np.mean(err > 1e-3)),
                "affected_edge_samples": int(bad_1e6.size),
            })
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sizes", type=int, nargs="*", default=list(DEFAULT_SIZES))
    ap.add_argument("--non-pow2", type=int, nargs="*",
                    default=[1000, 10000, 100000])
    ap.add_argument("--configs", nargs="*", default=list(ALL_CONFIGS))
    ap.add_argument("--out", default=os.path.join(
        _ROOT, "docs", "test", "Hilbert_Backend_Results.json"))
    ap.add_argument("--skip", nargs="*", default=[],
                    choices=["correctness", "timing", "memory", "primitives",
                             "effect", "pad_scan"])
    args = ap.parse_args()

    sizes = list(args.sizes)
    result = {
        "meta": {
            "generated": datetime.now().isoformat(timespec="seconds"),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
            "processor": platform.processor(),
            "cpu_count": os.cpu_count(),
            "native_kernel_active": _fht._native is not None,
            "native_module": getattr(_fht._native, "__file__", None),
            "fht_forward_is_native": fht_forward is getattr(
                _fht._native, "fht_forward", None),
            "sizes": [int(s) for s in sizes],
            "non_pow2_sizes": [int(s) for s in args.non_pow2],
            "configs": list(args.configs),
        },
    }

    if "correctness" not in args.skip:
        result["correctness"] = correctness(sizes, args.non_pow2)
    if "timing" not in args.skip:
        result["timing"] = timing(sizes, args.configs)
    if "memory" not in args.skip:
        result["memory"] = memory([s for s in sizes if s in MEM_SIZES] or
                                  sizes[:3])
    if "primitives" not in args.skip:
        result["primitives"] = fht_primitives(sizes)
    if "effect" not in args.skip:
        result["effect"] = effect([s for s in sizes if s <= 65536][:3] or sizes[:3],
                                  args.non_pow2)
    if "pad_scan" not in args.skip:
        result["pad_scan"] = pad_scan()

    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)
    print(f"[hilbert-bench] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
