"""
VMD 换版对比实验: 旧版 (vmdpy 包装) vs 新版 (原生实现, 已落到 ``VMD.py``)。

- 旧版行为 = 直接调 ``vmdpy.VMD`` (旧 ``VMD.py`` 就是它的薄包装, 代码可由 git 取回);
- 新版 = ``Modal_Decomposition.VMD.VMD`` (原生 ADMM, 不依赖 vmdpy);
- 指标: 用时(含/不含首次导入) / 提交内存峰值 / 中心频率精度 / 残差比 /
  ``reconstruct()`` 误差 / 与旧版模态的一致性;
- 结果打印 Markdown 表并合并写入 ``docs/VMD_Native_vs_vmdpy_Results.json``。

用法::

    python tests/comparison/bench_vmd_native.py --sizes 1024 4096 16384
    python tests/comparison/bench_vmd_native.py --worker '<json>'

Python version: 3.10
Only accessed by: ``docs/VMD_Native_vs_vmdpy_Report.md`` (实验复现)
Modify: 2026.3.4
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
_RESULT_JSON = _ROOT / "docs" / "VMD_Native_vs_vmdpy_Results.json"

#: 真值音 (cycles/sample) —— 与 N 无关, 便于跨尺寸对照。
_TONES = ((0.010, 1.0), (0.050, 0.7), (0.120, 0.4))
_K = 3
_NOISE = 0.05


def _signal(n: int):
    import numpy as np
    rng = np.random.default_rng(0)
    idx = np.arange(n, dtype=np.float64)
    x = np.zeros(n)
    for f, a in _TONES:
        x += a * np.cos(2.0 * np.pi * f * idx)
    return x + _NOISE * rng.standard_normal(n)


class _CommitSampler:
    """按 5ms 采样**提交内存** (vms) 的峰值 —— 只看 RSS 会被 Windows 的工作集裁剪骗到。"""

    def __init__(self, interval: float = 0.005):
        import threading
        import psutil
        self._proc = psutil.Process()
        self._interval = interval
        self.peak = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _loop(self):
        while not self._stop.is_set():
            try:
                self.peak = max(self.peak, int(self._proc.memory_info().vms))
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


def _worker(args: dict) -> int:
    import numpy as np

    sys.path.insert(0, str(_SRC))
    from Modal_Decomposition.Utils import Memory

    tag = args["impl"]
    n = int(args["n"])
    K = int(args["k"])
    x = _signal(n)
    out = {"impl": tag, "n": n, "k": K}

    t_import = 0.0
    if tag == "vmdpy":
        t0 = time.perf_counter()
        from vmdpy import VMD as vmdpy_VMD
        t_import = time.perf_counter() - t0
    else:
        from Modal_Decomposition.VMD import VMD as NativeVMD

    budget = Memory.memory_budget(n * 8 * 12, force=True)
    out["budget_ok"] = bool(budget["ok"])
    if not budget["ok"]:
        out["status"] = "over_budget"
        print("@@RESULT@@" + json.dumps(out))
        return 0

    before = Memory.get_process_memory(force=True)["private"]
    with _CommitSampler() as sampler:
        t0 = time.perf_counter()
        if tag == "vmdpy":
            u, u_hat, omega = vmdpy_VMD(x, 2000.0, 0.0, K, 0, 1, 1e-7)
            first = time.perf_counter() - t0
            best = first
            for _ in range(args["reps"] - 1):
                t0 = time.perf_counter()
                u, u_hat, omega = vmdpy_VMD(x, 2000.0, 0.0, K, 0, 1, 1e-7)
                best = min(best, time.perf_counter() - t0)
            modes = np.asarray(u)
            resid = x - modes.sum(axis=0)
            recon_err = float(np.max(np.abs(modes.sum(axis=0) - x)))
            om = np.asarray(omega)[-1]
            n_iter = int(np.asarray(omega).shape[0])
            u_hat_info = np.asarray(u_hat)
        else:
            r = NativeVMD(num_imf=K, epsilon=1e-7, fs=float(n)).decompose(x)
            first = time.perf_counter() - t0
            best = first
            for _ in range(args["reps"] - 1):
                t0 = time.perf_counter()
                r = NativeVMD(num_imf=K, epsilon=1e-7, fs=float(n)).decompose(x)
                best = min(best, time.perf_counter() - t0)
            modes = r.IMFs
            resid = r.Res
            recon_err = float(np.max(np.abs(r.reconstruct() - x)))
            om = r.info["omega"]
            n_iter = int(r.info["n_iter"])
            u_hat_info = r.info["u_hat"]
        import_err = None
        if tag == "vmdpy":
            import_err = "n/a"

    # 质量: 中心频率与真值的偏差 (按模态匹配真值频率)
    true_f = np.array([t[0] for t in _TONES])
    omega_hz = np.sort(om * n) / n          # 归一化循环频率 (cycles/sample)
    hit = []
    for f in true_f:
        hit.append(float(np.min(np.abs(omega_hz - f))))
    out.update({
        "status": "ok",
        "import_s": t_import,
        "first_s": first,
        "best_s": best,
        "reps": args["reps"],
        "peak_commit_mb": sampler.peak / 2 ** 20,
        "commit_delta_mb": (sampler.peak - before) / 2 ** 20,
        "n_iter": n_iter,
        "omega_hz": [float(v) for v in omega_hz],
        "omega_err_by_tone": hit,
        "omega_err_max": float(np.max(hit)),
        "residual_ratio": float(np.linalg.norm(resid) / np.linalg.norm(x)),
        "recon_err": recon_err,
        "u_hat_shape": list(np.shape(u_hat_info)),
        "imf_shape": list(np.shape(modes)),
    })
    print("@@RESULT@@" + json.dumps(out))
    return 0


def run_one(n: int, impl: str, reps: int, commit_cap_gb: float, time_cap_s: float) -> dict:
    import psutil

    payload = json.dumps({"impl": impl, "n": n, "k": _K, "reps": reps})
    env = dict(os.environ)
    env["PYTHONPATH"] = str(_SRC)
    env["PYTHONIOENCODING"] = "utf-8"
    proc = subprocess.Popen(
        [sys.executable, str(Path(__file__).resolve()), "--worker", payload],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, env=env,
    )
    mon = psutil.Process(proc.pid)
    cap = commit_cap_gb * 1024 ** 3
    t0 = time.perf_counter()
    verdict = None
    while proc.poll() is None:
        try:
            if max(mon.memory_info().vms, mon.memory_info().rss) > cap:
                verdict = "commit_cap"
                break
        except psutil.Error:
            break
        if time.perf_counter() - t0 > time_cap_s:
            verdict = "timeout"
            break
        time.sleep(0.05)
    if verdict:
        try:
            mon.kill()
        except psutil.Error:
            pass
        proc.wait(timeout=30)
        return {"impl": impl, "n": n, "status": verdict, "seconds": time.perf_counter() - t0}
    stdout, _ = proc.communicate()
    for line in stdout.splitlines():
        if line.startswith("@@RESULT@@"):
            return json.loads(line[len("@@RESULT@@"):])
    return {"impl": impl, "n": n, "status": "error"}


def fmt(row: dict) -> str:
    if row.get("status") != "ok":
        return "| %-6s | %-6d | %-12s | — | — | — | — | — |" % (
            row["impl"], row["n"], row.get("status"))
    return "| %-6s | %-6d | %8.3f | %9.0f | %9.2e | %9.2e | %6d | %.2e |" % (
        row["impl"], row["n"], row["best_s"], row.get("commit_delta_mb", float("nan")),
        row["omega_err_max"], row["residual_ratio"], row["n_iter"], row["recon_err"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", default=None)
    ap.add_argument("--sizes", type=int, nargs="*", default=[1024, 4096, 16384, 65536])
    ap.add_argument("--impls", nargs="*", default=["vmdpy", "native"])
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--commit-cap-gb", type=float, default=3.6)
    ap.add_argument("--time-cap-s", type=float, default=600.0)
    args = ap.parse_args()

    if args.worker:
        return _worker(json.loads(args.worker))

    print("| impl | N | best(s) | commitΔ(MB) | ω err | res ratio | n_iter | recon err |")
    print("|---|---|---|---|---|---|---|---|")
    merged = {}
    if _RESULT_JSON.exists():
        try:
            for row in json.loads(_RESULT_JSON.read_text(encoding="utf-8")):
                merged[(row.get("impl"), row.get("n"), row.get("k"))] = row
        except Exception:
            pass
    for n in args.sizes:
        for impl in args.impls:
            row = run_one(n, impl, args.reps, args.commit_cap_gb, args.time_cap_s)
            merged[(row.get("impl"), row.get("n"), row.get("k"))] = row
            print(fmt(row), flush=True)

    _RESULT_JSON.parent.mkdir(exist_ok=True)
    rows = sorted(merged.values(), key=lambda r: (r.get("n", 0), str(r.get("impl"))))
    _RESULT_JSON.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print("\nraw results (%d rows) -> %s" % (len(rows), _RESULT_JSON))

    # 版本间一致性: 同 N 下 native 与 vmdpy 的模态/频谱
    by = {(r.get("n"), r.get("impl")): r for r in rows if r.get("status") == "ok"}
    print("\n| N | ω 最大差 (排序后) | 残差比 vmdpy/native | 迭代数 vmdpy/native |")
    print("|---|---|---|---|")
    for n in args.sizes:
        a, b = by.get((n, "vmdpy")), by.get((n, "native"))
        if not a or not b:
            continue
        w = max(abs(x - y) for x, y in zip(a["omega_hz"], b["omega_hz"]))
        print("| %d | %.2e | %.4e / %.4e | %d / %d |" % (
            n, w, a["residual_ratio"], b["residual_ratio"], a["n_iter"], b["n_iter"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
