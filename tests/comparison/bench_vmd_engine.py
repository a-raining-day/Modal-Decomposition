"""
VMD 引擎与"大数组不崩"实验: ram vs chunked vs auto (含外存)。

目的: 证明 ``engine="auto"`` / ``"chunked"`` + ``out_of_core`` 能在**受限内存**下完成
RAM 路径做不完的输入, 并量化代价。全部配置跑在独立子进程里, 带
**预算闸门 + 提交内存硬上限**, 因此本脚本本身不会把机器内存吃满。

用法::

    python tests/comparison/bench_vmd_engine.py --sizes 2097152 4194304 --commit-cap-gb 0.8
    python tests/comparison/bench_vmd_engine.py --worker '<json>'

Python version: 3.10
Only accessed by: ``docs/VMD_Refactor_and_Validation_Report.md`` (实验复现)
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
_RESULT_JSON = _ROOT / "docs" / "VMD_Engine_Results.json"

_TONES = ((0.010, 1.0), (0.050, 0.7), (0.120, 0.4))
_K = 3
_NOISE = 0.05

#: (标签, 构造参数): ram 强制全内存, auto/chunked 由策略分流
CONFIGS = {
    "ram": dict(engine="ram"),
    "chunked": dict(engine="chunked"),
    "auto": dict(engine="auto"),
    "auto+ooc": dict(engine="auto", out_of_core=True),
}


def _signal(n: int):
    import numpy as np
    rng = np.random.default_rng(0)
    idx = np.arange(n, dtype=np.float64)
    x = np.zeros(n)
    for f, a in _TONES:
        x += a * np.cos(2.0 * np.pi * f * idx)
    return x + _NOISE * rng.standard_normal(n)


def _worker(args: dict) -> int:
    import numpy as np
    import threading
    import psutil

    sys.path.insert(0, str(_SRC))
    from Modal_Decomposition.Utils import Memory
    from Modal_Decomposition.VMD import VMD

    n, label = int(args["n"]), args["config"]
    kw = dict(CONFIGS[label])
    if args.get("budget_mb"):                       # 模拟"内存受限的机器"
        Memory.set_absolute_limit(int(args["budget_mb"]) * 1024 ** 2)

    x = _signal(n)
    out = {"config": label, "n": n, "input_mb": n * 8 / 2 ** 20,
           "budget_mb": args.get("budget_mb"), "limit": Memory.get_memory_policy()}

    proc = psutil.Process()
    before = proc.memory_info().vms
    peak = {"v": 0}
    stop = threading.Event()

    def sample():
        while not stop.is_set():
            try:
                peak["v"] = max(peak["v"], proc.memory_info().vms)
            except Exception:
                pass
            stop.wait(0.005)

    th = threading.Thread(target=sample, daemon=True)
    th.start()
    t0 = time.perf_counter()
    try:
        r = VMD(num_imf=_K, fs=float(n), **kw).decompose(x)
    except MemoryError as exc:
        stop.set(); th.join(timeout=1)
        out.update({"status": "MemoryError", "detail": str(exc)[:80]})
        print("@@RESULT@@" + json.dumps(out)); return 0
    except Exception as exc:
        stop.set(); th.join(timeout=1)
        out.update({"status": type(exc).__name__, "detail": str(exc)[:120]})
        print("@@RESULT@@" + json.dumps(out)); return 0
    dt = time.perf_counter() - t0
    stop.set(); th.join(timeout=1)

    out.update({
        "status": "ok",
        "seconds": dt,
        "peak_commit_mb": peak["v"] / 2 ** 20,
        "commit_delta_mb": (peak["v"] - before) / 2 ** 20,
        "engine": r.info["engine"],
        "chunk_size": int(r.info["chunk_size"]),
        "out_of_core": bool(r.info["out_of_core"]),
        "n_iter": int(r.info["n_iter"]),
        "residual_ratio": float(r.info["residual_ratio"]),
        "recon_err": float(np.max(np.abs(r.reconstruct() - x))),
        "projected_total_mb": r.info["projected_bytes"]["total"] / 2 ** 20,
        "projected_modes_mb": r.info["projected_bytes"]["modes"] / 2 ** 20,
    })
    print("@@RESULT@@" + json.dumps(out))
    return 0


def run_one(n: int, label: str, budget_mb: int, cap_gb: float, time_cap_s: float) -> dict:
    import psutil

    payload = json.dumps({"n": n, "config": label, "budget_mb": budget_mb})
    env = dict(os.environ); env["PYTHONPATH"] = str(_SRC); env["PYTHONIOENCODING"] = "utf-8"
    proc = subprocess.Popen(
        [sys.executable, str(Path(__file__).resolve()), "--worker", payload],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, env=env,
    )
    mon = psutil.Process(proc.pid)
    cap = cap_gb * 1024 ** 3
    t0 = time.perf_counter()
    verdict = None
    while proc.poll() is None:
        try:
            if mon.memory_info().vms > cap:
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
        return {"config": label, "n": n, "status": verdict,
                "seconds": time.perf_counter() - t0}
    stdout, _ = proc.communicate()
    for line in stdout.splitlines():
        if line.startswith("@@RESULT@@"):
            return json.loads(line[len("@@RESULT@@"):])
    return {"config": label, "n": n, "status": "error"}


def fmt(r: dict) -> str:
    if r.get("status") != "ok":
        return "| %-8s | %-9d | %-11s | %7.2f | — | — | — | — |" % (
            r["config"], r["n"], r.get("status"), r.get("seconds", float("nan")))
    return "| %-8s | %-9d | %-11s | %7.2f | %9.0f | %-8s | %-8d | %.2e |" % (
        r["config"], r["n"], "ok", r["seconds"], r["peak_commit_mb"],
        r["engine"] + ("+ooc" if r["out_of_core"] else ""), r["chunk_size"], r["residual_ratio"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", default=None)
    ap.add_argument("--sizes", type=int, nargs="*", default=[2097152, 4194304])
    ap.add_argument("--configs", nargs="*", default=list(CONFIGS))
    ap.add_argument("--budget-mb", type=int, default=256, help="模拟的内存预算 (绝对策略)")
    ap.add_argument("--commit-cap-gb", type=float, default=0.9)
    ap.add_argument("--time-cap-s", type=float, default=600.0)
    args = ap.parse_args()

    if args.worker:
        return _worker(json.loads(args.worker))

    print("| config | N | status | 秒 | 提交峰值(MB) | 生效引擎 | chunk | 残差比 |")
    print("|---|---|---|---|---|---|---|---|")
    merged = {}
    if _RESULT_JSON.exists():
        try:
            for row in json.loads(_RESULT_JSON.read_text(encoding="utf-8")):
                merged[(row.get("config"), row.get("n"), row.get("budget_mb"))] = row
        except Exception:
            pass
    for n in args.sizes:
        for label in args.configs:
            row = run_one(n, label, args.budget_mb, args.commit_cap_gb, args.time_cap_s)
            row["budget_mb"] = args.budget_mb
            merged[(row.get("config"), row.get("n"), args.budget_mb)] = row
            print(fmt(row), flush=True)

    _RESULT_JSON.parent.mkdir(exist_ok=True)
    rows = sorted(merged.values(), key=lambda r: (r.get("n", 0), str(r.get("config"))))
    _RESULT_JSON.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print("\nraw results (%d rows) -> %s" % (len(rows), _RESULT_JSON))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
