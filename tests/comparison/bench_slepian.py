"""
Slepian (DPSS) 后端基准脚本 —— 生成 docs/Slepian_Backend_Report.md 中的表格数据。

用法::

    python tests/comparison/bench_slepian.py            # 默认规模组
    python tests/comparison/bench_slepian.py --quick    # 小规模快速版

三项内容:
  1) 正确性: 各后端与 scipy.signal.windows.dpss 的偏差 (序列逐元素 + 集中比);
  2) 性能: 稳态重复调用耗时 (best-of-rep) 与 tracemalloc 峰值, 逐 N 对比;
  3) 冷启动: 单次调用的整进程 wall time (含 import scipy.signal 的代价)。

内存约束: numpy 后端的稠密分解只到 SLEPIAN_NUMPY_MAX_BYTES 预算 (默认 N<=11584),
更大 N 直接跳过 (脚本不会尝试 GB 级分配)。
"""

import argparse
import os
import subprocess
import sys
import time
import tracemalloc
import warnings

warnings.simplefilter("ignore")

import numpy as np
from scipy.signal.windows import dpss

from Modal_Decomposition.Base.ConstDefine import SLEPIAN_NUMPY_MAX_BYTES
from Modal_Decomposition.Utils.Slepian import available_backends, slepian
from Modal_Decomposition.Utils._Slepian.numpy_slepian import generate_slepian_numpy

#: numpy 后端可用的样本上限 (由稠密矩阵预算反推)。
NUMPY_N_MAX = 2 * int(np.sqrt(SLEPIAN_NUMPY_MAX_BYTES // 8))

_QUICK = False


def _numpy_ok(N: int) -> bool:
    return N <= NUMPY_N_MAX


def _call(backend, N, NW, K, ratios=True):
    return slepian(N, NW, K, mod=backend, return_ratios=ratios, norm=2)


def bench_accuracy(sizes, nw_list, k_list):
    """各后端 vs scipy: 序列逐元素最大偏差与集中比最大偏差。"""
    print("\n## 1) 正确性 (vs scipy.signal.windows.dpss)\n")
    print("| 后端 | 组合数 | 序列 max\\|Δ\\| | 集中比 max\\|Δ\\| | 备注 |")
    print("|---|---:|---:|---:|---|")
    for backend in ("numpy", "C", "scipy"):
        if backend not in available_backends() or not available_backends()[backend]:
            print(f"| {backend} | - | - | - | 不可用 |")
            continue
        worst_v = worst_r = 0.0
        n = skipped = 0
        for N in sizes:
            if backend == "numpy" and not _numpy_ok(N):
                skipped += 1
                continue
            for NW in nw_list:
                if NW >= N / 2:                       # scipy 自身要求 NW < M/2
                    continue
                for K in k_list:
                    K = int(min(K, N))
                    try:
                        ref, ref_r = dpss(N, NW, K, norm=2, return_ratios=True)
                    except Exception:
                        skipped += 1
                        continue
                    try:
                        got, got_r = _call(backend, N, NW, K)
                    except Exception:
                        skipped += 1
                        continue
                    worst_v = max(worst_v, float(np.max(np.abs(got - ref))))
                    worst_r = max(worst_r, float(np.max(np.abs(got_r - ref_r))))
                    n += 1
        note = f"跳过 {skipped} 组" if skipped else "全覆盖"
        if backend == "numpy":
            note = f"N<={NUMPY_N_MAX}; " + note
        print(f"| {backend} | {n} | {worst_v:.2e} | {worst_r:.2e} | {note} |")


def bench_speed(sizes, NW=4.0, K=5, rep=3):
    """稳态耗时 (best-of-rep) 与峰值内存。"""
    print(f"\n## 2) 性能与峰值内存 (NW={NW}, K={K}, best-of-{rep})\n")
    print("| N | scipy ms | numpy ms | C ms | scipy MB | numpy MB | C MB |")
    print("|---:|---:|---:|---:|---:|---:|---:|")

    def run(fn, N):
        best = float("inf")
        for _ in range(rep):
            t0 = time.perf_counter()
            fn(N)
            best = min(best, time.perf_counter() - t0)
        tracemalloc.start()
        fn(N)
        peak = tracemalloc.get_traced_memory()[1] / 1024 ** 2
        tracemalloc.stop()
        return best * 1e3, peak

    for N in sizes:
        s_ms, s_mb = run(lambda n: dpss(n, NW, K, norm=2), N)
        if _numpy_ok(N):
            n_ms, n_mb = run(lambda n: generate_slepian_numpy(n, NW, K, norm=2), N)
            n_s, n_p = f"{n_ms:.2f}", f"{n_mb:.1f}"
        else:
            n_s, n_p = "—", "—"
        if available_backends()["C"]:
            c_ms, c_mb = run(lambda n: _call("C", n, NW, K, ratios=False), N)
            c_s, c_p = f"{c_ms:.2f}", f"{c_mb:.1f}"
        else:
            c_s, c_p = "—", "—"
        print(f"| {N} | {s_ms:.2f} | {n_s} | {c_s} | {s_mb:.1f} | {n_p} | {c_p} |")


def bench_cold_start(N=100000, NW=4.0, K=5):
    """冷启动: 新进程里 import + 单次调用 的全程 wall time。"""
    print(f"\n## 3) 冷启动单次调用 (N={N}, 整进程 wall ms)\n")
    print("| 后端 | 首次 wall | 第二次 wall | 说明 |")
    print("|---|---:|---:|---|")
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [os.path.join(os.path.dirname(__file__), "..", "..", "src")]
        + [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p]
    )
    for backend in ("C", "scipy"):
        if backend == "C" and not available_backends()["C"]:
            continue
        walls = []
        for _ in range(2):
            code = (
                "from Modal_Decomposition.Utils.Slepian import slepian;"
                f"slepian({N}, {NW}, {K}, mod={backend!r}, return_ratios=True)"
            )
            t0 = time.perf_counter()
            subprocess.run([sys.executable, "-c", code], capture_output=True, env=env)
            walls.append((time.perf_counter() - t0) * 1e3)
        print(f"| {backend} | {walls[0]:.0f} | {walls[1]:.0f} | "
              f"{'ctypes 装载共享库' if backend == 'C' else '首次需 import scipy.signal'} |")


def bench_cache(sizes, NW=4.0, K=5, rep=5):
    """Tier-1 缓存: 计算(miss) vs 命中(hit) 耗时与缓存占用。"""
    from Modal_Decomposition.Utils.Slepian import cache_info, clear_cache

    print(f"\n## 4) Tier-1 进程内缓存 (NW={NW}, K={K}, 命中为 best-of-{rep})\n")
    print("| N | 后端 | 计算(miss) ms | 命中(hit) ms | 加速比 | 缓存条目字节 |")
    print("|---:|---|---:|---:|---:|---:|")
    for backend in ("scipy", "C"):
        if backend == "C" and not available_backends()["C"]:
            continue
        for N in sizes:
            clear_cache()
            _call(backend, N, NW, K, ratios=False)                 # miss
            info = cache_info()
            entry_bytes = info["bytes"]
            best = float("inf")
            for _ in range(rep):
                t0 = time.perf_counter()
                _call(backend, N, NW, K, ratios=False)             # hit
                best = min(best, time.perf_counter() - t0)
            hit_ms = best * 1e3
            t0 = time.perf_counter()
            slepian(N, NW, K, mod=backend, return_ratios=False, cache=False)
            miss_ms = (time.perf_counter() - t0) * 1e3
            print(f"| {N} | {backend} | {miss_ms:.2f} | {hit_ms:.3f} | "
                  f"{miss_ms / max(hit_ms, 1e-9):.0f}x | {entry_bytes} |")
    clear_cache()


def bench_cache_workload(NW=4.0, K=5):
    """模拟真实负载: 对等长信号反复调用 -> 缓存命中率与其收益。"""
    from Modal_Decomposition.Utils.Slepian import cache_info, clear_cache

    print("\n## 5) 缓存命中率 (等长信号批量负载)\n")
    print("| 场景 | 调用次数 | 命中 | 未命中 | 命中率 | 总耗时 ms(开缓存) | 总耗时 ms(关缓存) | 加速 |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for label, N, calls in (("EWT 典型 (N=16384)", 16384, 20),
                            ("中规模 (N=65536)", 65536, 10),
                            ("不同长度 (N 变化)", None, 10)):
        clear_cache()
        sizes = [16384] * calls if N else [4096 * (i + 1) for i in range(calls)]
        t0 = time.perf_counter()
        for n in sizes:
            slepian(n, NW, K, mod="scipy", return_ratios=False)
        with_cache = (time.perf_counter() - t0) * 1e3
        info = cache_info()
        t0 = time.perf_counter()
        for n in sizes:
            slepian(n, NW, K, mod="scipy", return_ratios=False, cache=False)
        without = (time.perf_counter() - t0) * 1e3
        total = info["hits"] + info["misses"]
        rate = info["hits"] / total if total else 0.0
        print(f"| {label} | {calls} | {info['hits']} | {info['misses']} | {rate:.0%} | "
              f"{with_cache:.1f} | {without:.1f} | {without / max(with_cache, 1e-9):.1f}x |")
    clear_cache()


def main():
    global _QUICK
    ap = argparse.ArgumentParser(description="Slepian backend benchmark")
    ap.add_argument("--quick", action="store_true", help="小规模快速版")
    args = ap.parse_args()
    _QUICK = args.quick

    print("# Slepian 后端基准 (自动生成)")
    print(f"\n后端可用性: {available_backends()} | numpy 上限 N<={NUMPY_N_MAX}")
    sizes = [512, 1024, 2048, 4096, 8192] if _QUICK else \
            [1024, 4096, 8192, 16384, 65536, 262144, 1048576]
    bench_accuracy([256, 1024, 4096] if _QUICK else [257, 1024, 4096, 8192],
                   [1.0, 2.0, 4.0], [1, 3, 5])
    bench_speed(sizes)
    bench_cold_start(100000 if not _QUICK else 8192)
    bench_cache([1024, 4096, 16384, 65536] if _QUICK else [1024, 4096, 16384, 65536, 262144])
    bench_cache_workload()


if __name__ == "__main__":
    main()
