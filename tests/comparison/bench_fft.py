"""
FFT 后端基准: numpy / scipy / pyfftw / tiled / cupy × {1,10,100,500,1024}MB.

设计与安全约束 (上一轮"采样 RSS 做上限"失效的教训, 见
``docs/VMD_Large_Array_Iteration_Report.md``):

- **跑前先过预算闸门**: ``Utils.Memory.memory_budget(extra=预计峰值)``; 不通过就记
  ``over_budget`` 并**不分配**, 而不是"先分配再被杀";
- **监控提交内存** (``private`` / ``vms``) 而不是工作集 RSS —— Windows 会主动裁剪
  工作集, 只看 RSS 会让"提交 10GB 而物理只剩 3GB"的进程被判为安全;
- 每个 (后端 × 尺寸) 组合跑在**独立子进程**里, 超预算/超时即杀;
- 结果打印 Markdown 表并写 ``docs/FFT_Backend_Results.json``。

用法
----
::

    python tests/comparison/bench_fft.py --sizes 1 10
    python tests/comparison/bench_fft.py --sizes 100 500 1024 --time-cap-s 600
    python tests/comparison/bench_fft.py --worker '<json>'      # 内部使用

Python version: 3.10
Only accessed by: ``docs/FFT_Backend_Report.md`` (实验复现)
Modify:
    2026.3.4
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_SRC = _ROOT / "src"
_RESULT_JSON = _ROOT / "docs" / "FFT_Backend_Results.json"

#: 后端配置: 名称 → (op, kwargs)。kwargs 直接透传给 Utils.FFT。
CONFIGS = {
    "numpy": dict(op="rfft", kw=dict(mod="numpy")),
    "scipy-w1": dict(op="rfft", kw=dict(mod="scipy", workers=1)),
    "scipy-w4": dict(op="rfft", kw=dict(mod="scipy", workers=4)),
    "scipy-w8": dict(op="rfft", kw=dict(mod="scipy", workers=8)),
    "scipy-all": dict(op="rfft", kw=dict(mod="scipy", workers=-1)),
    "pyfftw-w1": dict(op="rfft", kw=dict(mod="pyfftw", workers=1)),
    "pyfftw-w8": dict(op="rfft", kw=dict(mod="pyfftw", workers=8)),
    "pyfftw-all": dict(op="rfft", kw=dict(mod="pyfftw", workers=-1)),
    "pyfftw-measure": dict(op="rfft", kw=dict(mod="pyfftw", workers=-1,
                                             planner_effort="FFTW_MEASURE")),
    "tiled": dict(op="rfft", kw=dict(mod="tiled")),
    "cupy": dict(op="rfft", kw=dict(mod="cupy")),
    "auto": dict(op="rfft", kw=dict(mod=None)),
}

#: 复数 fft 对照组 (用于说明 tiled 只在复数变换上有意义)。
CONFIGS_FFT = {
    "fft-numpy": dict(op="fft", kw=dict(mod="numpy")),
    "fft-scipy-all": dict(op="fft", kw=dict(mod="scipy", workers=-1)),
    "fft-tiled": dict(op="fft", kw=dict(mod="tiled")),
    "fft-pyfftw-all": dict(op="fft", kw=dict(mod="pyfftw", workers=-1)),
}


def _n_samples(size_mb: float) -> int:
    """float64 输入在给定 MB 下的样本数 (取 2 的幂附近, 便于对比 FFT 效率)。"""
    n = int(size_mb * 1024 ** 2 // 8)
    return n - (n % 2)


def _projected_peak(n: int, dtype_bytes: int = 8) -> int:
    """峰值估计: 输入 + 半谱输出 + 10% 余量 (FFT 内部临时量)。

    估得**偏紧**是有意的: 这道闸门只用于"明显超预算就别开跑", 真正兜底的是运行期
    的提交内存监控; 估得过松会把本来可行的 1GB 用例误判为 over_budget。
    """
    raw = n * dtype_bytes + (n // 2 + 1) * dtype_bytes * 2
    return int(raw * 1.1)


class _CommitSampler:
    """worker 内部按 5ms 采样**提交内存** (vms/private) 的峰值。

    为什么不用父进程采样: 小尺寸的 FFT 只要十几毫秒, 父进程 50ms 的轮询会完全错过
    峰值; 而 Windows 会主动裁剪工作集, 只看 RSS 又会低估大尺寸的真实占用。
    """

    def __init__(self, interval: float = 0.005):
        import threading
        import psutil

        self._proc = psutil.Process()
        self._interval = interval
        self.peak_commit = 0
        self.peak_rss = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _loop(self):
        while not self._stop.is_set():
            try:
                mi = self._proc.memory_info()
                self.peak_commit = max(self.peak_commit, int(mi.vms))
                self.peak_rss = max(self.peak_rss, int(mi.rss))
            except Exception:
                pass
            self._stop.wait(self._interval)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=1.0)
        return False


# --------------------------------------------------------------------------- #
# worker
# --------------------------------------------------------------------------- #
def _worker(args: dict) -> int:
    import numpy as np

    sys.path.insert(0, str(_SRC))
    from Modal_Decomposition.Utils import Memory
    from Modal_Decomposition.Utils.FFT import fft as FFT

    rfft, fft = FFT.rfft, FFT.fft

    name = args["config"]
    size_mb = args["size_mb"]
    n = _n_samples(size_mb)
    spec = dict(CONFIGS)
    spec.update(CONFIGS_FFT)
    if name not in spec:
        print("@@RESULT@@" + json.dumps({"config": name, "size_mb": size_mb, "status": "unknown-config"}))
        return 0
    op = spec[name]["op"]
    kw = dict(spec[name]["kw"])

    out = {"config": name, "size_mb": size_mb, "n_samples": n, "op": op}
    # 预算闸门: 预测峰值先过 Utils.Memory, 不通过就不分配。
    projected = _projected_peak(n)
    budget = Memory.memory_budget(projected, force=True)
    out["projected_mb"] = projected / 2 ** 20
    out["budget_mb"] = (budget["budget"] or 0) / 2 ** 20
    out["limited_by"] = budget["limited_by"]
    if not budget["ok"]:
        out["status"] = "over_budget"
        print("@@RESULT@@" + json.dumps(out))
        return 0

    proc_start_commit = Memory.get_process_memory(force=True)["private"]

    # 数据准备 (不计时, 但计入内存峰值 —— 真实使用中输入必然存在)
    rng = np.random.default_rng(0)
    fn = fft if op == "fft" else rfft
    with _CommitSampler() as sampler:
        x = rng.standard_normal(n)
        if op == "fft":
            x = x.astype(np.complex128)

        # 导入/规划开销单列 (cupy 首次导入、pyfftw 首次 plan 都很贵)
        t0 = time.perf_counter()
        try:
            y = fn(x, **kw)
        except ImportError as exc:
            out["status"] = "unavailable"
            out["detail"] = str(exc).splitlines()[0][:120]
            out["pip_hint"] = str(exc).splitlines()[-1].strip() if len(str(exc).splitlines()) > 1 else ""
            print("@@RESULT@@" + json.dumps(out))
            return 0
        except Exception as exc:
            out["status"] = "%s" % type(exc).__name__
            out["detail"] = str(exc)[:140]
            print("@@RESULT@@" + json.dumps(out))
            return 0
        first = time.perf_counter() - t0

        # reps: 全部尺寸都按 --reps 跑 —— 后端首调含 import/plan(cold), 之后为 warm。
        # 报告里两者都要看: 库内单次调用付 cold, 批量调用付 warm。
        reps = max(1, int(args["reps"]))
        best = first
        for _ in range(reps - 1):
            t0 = time.perf_counter()
            y = fn(x, **kw)
            best = min(best, time.perf_counter() - t0)

        out["peak_commit_mb"] = sampler.peak_commit / 2 ** 20
        out["peak_rss_mb"] = sampler.peak_rss / 2 ** 20
        # 相对"导入后基线"的增量才是这次变换真正吃的内存 (绝对值含解释器/DLL 基线)
        out["peak_commit_delta_mb"] = (sampler.peak_commit - proc_start_commit) / 2 ** 20

    # 正确性校验放在采样区**之外**: 参考数组/差值数组会额外占一份全量内存,
    # 混进来会污染"FFT 本身的峰值"这一指标。
    # 大尺寸不做全量对比 (会再造一份全量数组), 改为**同一段切片喂两个后端**比对 ——
    # 注意不能用"输出前缀 vs 前缀的 FFT", 那在数学上就不相等。
    try:
        if n <= (64 << 20) // 8:
            ref = np.fft.rfft(x) if op == "rfft" else np.fft.fft(x)
            err = float(np.abs(y - ref).max())
            out["verify"] = "full(n=%d)" % n
        else:
            m = 1 << 20
            seg = np.ascontiguousarray(x[:m])
            if op == "rfft":
                ref = np.fft.rfft(seg)
                got = fn(seg, **kw)
            else:
                ref = np.fft.fft(seg)
                got = fn(seg, **kw)
            err = float(np.abs(np.asarray(got) - ref).max())
            out["verify"] = "slice(n=%d)" % m
        out["max_abs_err"] = err
    except Exception as exc:
        out["max_abs_err"] = None
        out["detail"] = "verify failed: %s" % str(exc)[:80]

    proc = Memory.get_process_memory(force=True)
    out.update({
        "status": "ok",
        "first_s": first,
        "best_s": best,
        "reps": reps,
        "process_commit_mb": (proc["private"] - proc_start_commit) / 2 ** 20,
        "process_commit_abs_mb": proc["private"] / 2 ** 20,
        "dtype_out": str(np.asarray(y).dtype),
    })
    print("@@RESULT@@" + json.dumps(out))
    return 0


# --------------------------------------------------------------------------- #
# 父进程
# --------------------------------------------------------------------------- #
def run_one(size_mb: float, config: str, reps: int, commit_cap_gb: float,
            time_cap_s: float) -> dict:
    import psutil

    payload = json.dumps({"config": config, "size_mb": size_mb, "reps": reps})
    env = dict(os.environ)
    env["PYTHONPATH"] = str(_SRC)
    env["PYTHONIOENCODING"] = "utf-8"
    proc = subprocess.Popen(
        [sys.executable, str(Path(__file__).resolve()), "--worker", payload],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, env=env,
    )
    mon = psutil.Process(proc.pid)
    cap = commit_cap_gb * 1024 ** 3
    t_start = time.perf_counter()
    verdict = None
    peak_commit = 0
    while proc.poll() is None:
        try:
            mi = mon.memory_info()
            commit = max(int(mi.vms), int(mi.rss))
            peak_commit = max(peak_commit, commit)
            if commit > cap:
                verdict = "commit_cap"
                break
        except psutil.Error:
            break
        if time.perf_counter() - t_start > time_cap_s:
            verdict = "timeout"
            break
        time.sleep(0.05)

    if verdict:
        try:
            mon.kill()
        except psutil.Error:
            pass
        proc.wait(timeout=30)
        return {"config": config, "size_mb": size_mb, "status": verdict,
                "seconds": time.perf_counter() - t_start,
                "peak_commit_mb": peak_commit / 2 ** 20}

    stdout, _ = proc.communicate()
    row = None
    for line in stdout.splitlines():
        if line.startswith("@@RESULT@@"):
            row = json.loads(line[len("@@RESULT@@"):])
    if row is None:
        row = {"config": config, "size_mb": size_mb, "status": "error",
               "seconds": time.perf_counter() - t_start}
    row.setdefault("peak_commit_mb", peak_commit / 2 ** 20)
    return row


def fmt(row: dict) -> str:
    st = row.get("status")
    base = "| %-13s | %-8g | %-13s |" % (row["config"], row["size_mb"], st)
    if st != "ok":
        extra = row.get("detail", "") or row.get("pip_hint", "")
        return base + " %-9s | %-9s | — | %s |" % ("", "", extra[:44])
    return base + " %9.3f | %9.0f | %.1e |  |" % (
        row["best_s"], row.get("peak_sampled_mb", row.get("peak_commit_delta_mb", float("nan"))),
        row.get("max_abs_err") or 0.0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", default=None)
    ap.add_argument("--sizes", type=float, nargs="*", default=[1, 10, 100, 500, 1024])
    ap.add_argument("--configs", nargs="*", default=list(CONFIGS))
    ap.add_argument("--with-fft", action="store_true", help="附加复数 fft 对照组")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--commit-cap-gb", type=float, default=3.0)
    ap.add_argument("--time-cap-s", type=float, default=300.0)
    args = ap.parse_args()

    if args.worker:
        return _worker(json.loads(args.worker))

    configs = list(args.configs) + (list(CONFIGS_FFT) if args.with_fft else [])
    print("| config | size(MB) | status | best(s) | peakCommit(MB) | max|err| | note |")
    print("|---|---|---|---|---|---|---|")

    # 与已有结果**合并** (键 = config + size_mb): 分多次跑 (例如单独补 cupy) 不会
    # 把别的行覆盖掉, 最终 JSON 始终是完整矩阵。
    merged = {}
    if _RESULT_JSON.exists():
        try:
            for row in json.loads(_RESULT_JSON.read_text(encoding="utf-8")):
                merged[(row.get("config"), row.get("size_mb"), row.get("op", "rfft"))] = row
        except Exception:
            pass

    rows = []
    for size_mb in args.sizes:
        for config in configs:
            row = run_one(size_mb, config, args.reps, args.commit_cap_gb, args.time_cap_s)
            rows.append(row)
            merged[(row.get("config"), row.get("size_mb"), row.get("op", "rfft"))] = row
            print(fmt(row), flush=True)

    _RESULT_JSON.parent.mkdir(exist_ok=True)
    all_rows = sorted(merged.values(), key=lambda r: (r.get("size_mb", 0), str(r.get("config"))))
    _RESULT_JSON.write_text(json.dumps(all_rows, indent=2), encoding="utf-8")
    print("\nraw results (%d rows) -> %s" % (len(all_rows), _RESULT_JSON))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
