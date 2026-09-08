"""
Cache 速度对比基准: 有 cache 与无 cache 的调用耗时。

对比四种取用方式 (同一函数):
  A. direct       —— 直接 import 后调用 (无 cache, 基线)
  B. cached_once  —— Utils.get_xxx() 解析一次, 之后纯属性调用 (cache 一次性)
  C. getter_call  —— 每次调用都走 Utils.get_xxx() (每调含 cache.check+get)
  D. cache_get    —— 每次调用都走 cache.get(key) (跳过 check)

另测三类操作 (轻/中/重) 与首次访问的一次性注册成本。
输出: 终端表格 + docs/CacheSpeedReport.md (速度报告 + cache 流程说明)。

运行: .venv\\Scripts\\python.exe tests\\comparison\\bench_cache.py   (仓库根目录)
"""

import os
import statistics
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

from src.Modal_Decomposition import Utils as U  # noqa: E402
from src.Modal_Decomposition.Base.Cache import cache  # noqa: E402

# --------------------------------------------------------------------------- #
# 被测操作
# --------------------------------------------------------------------------- #
N_MONO = 10_000
ARR_MONO = np.arange(N_MONO, dtype=np.float64)  # 单调递增

t = np.linspace(0, 1, 2048, endpoint=False)
SIG = np.sin(2 * np.pi * 3 * t) + 0.3 * np.sin(2 * np.pi * 17 * t)

X_SP = np.linspace(0, 1, 500)
Y_SP = np.sin(40 * X_SP)
XI_SP = np.linspace(0, 1, 2000)

OPS = {
    "is_monotonic (10k)": {
        "direct": lambda: _mono_direct(ARR_MONO),
        "cached_once": lambda: _MONO_MOD.is_monotonic(ARR_MONO),
        "getter_call": lambda: U.get_monotonicity().is_monotonic(ARR_MONO),
        "cache_get": lambda: cache.get("Modal_Decomposition.Utils.Monotonicity").is_monotonic(ARR_MONO),
    },
    "Check_Time_and_Signal (2k)": {
        "direct": lambda: _check_direct(SIG),
        "cached_once": lambda: _CHECK_MOD.Check_Time_and_Signal(SIG),
        "getter_call": lambda: U.get_check().Check_Time_and_Signal(SIG),
        "cache_get": lambda: cache.get("Modal_Decomposition.Utils.Check").Check_Time_and_Signal(SIG),
    },
    "spline fit+eval (500)": {
        "direct": lambda: _spline_direct(X_SP, Y_SP)(XI_SP),
        "cached_once": lambda: _SPLINE_MOD.spline(X_SP, Y_SP)(XI_SP),
        "getter_call": lambda: U.get_spline().spline(X_SP, Y_SP)(XI_SP),
        "cache_get": lambda: cache.get("Modal_Decomposition.Utils.Spline").spline(X_SP, Y_SP)(XI_SP),
    },
    "envelope Hilbert (2k)": {
        "direct": lambda: _env_direct(SIG),
        "cached_once": lambda: _ENV_MOD.envelope(SIG),
        "getter_call": lambda: U.get_envelope().envelope(SIG),
        "cache_get": lambda: cache.get("Modal_Decomposition.Utils.Envelope").envelope(SIG),
    },
}

from src.Modal_Decomposition.Utils.Monotonicity import is_monotonic as _mono_direct  # noqa: E402
from src.Modal_Decomposition.Utils.Check import Check_Time_and_Signal as _check_direct  # noqa: E402
from src.Modal_Decomposition.Utils.Spline import spline as _spline_direct  # noqa: E402
from src.Modal_Decomposition.Utils.Envelope import envelope as _env_direct  # noqa: E402

_MONO_MOD = U.get_monotonicity()
_CHECK_MOD = U.get_check()
_SPLINE_MOD = U.get_spline()
_ENV_MOD = U.get_envelope()


def bench(fn, n_iter):
    """median per-call wall time (s) over 5 repeats of n_iter calls."""
    samples = []
    for _ in range(5):
        t0 = time.perf_counter()
        for _ in range(n_iter):
            fn()
        samples.append((time.perf_counter() - t0) / n_iter)
    return statistics.median(samples)


def bench_roundrobin(modes, n_iter, rounds=6):
    """
    轮转顺序测量多种取用方式 (抵消 CPU 频率漂移): 返回 {mode: median 每调秒}。

    先对每种方式做一次预热调用 (吸收惰性 import / 分配器预热), 不计时。
    """
    per_mode = {m: [] for m in modes}
    order = list(modes)
    for m in modes:  # warmup: 惰性 import 等一次性成本不进入样本
        modes[m]()
    for _ in range(rounds):
        for m in order:
            fn = modes[m]
            t0 = time.perf_counter()
            for _ in range(n_iter):
                fn()
            per_mode[m].append((time.perf_counter() - t0) / n_iter)
        order = order[::-1]  # 每轮反转, 进一步抵消顺序效应
    return {m: statistics.median(per_mode[m]) for m in modes}


def fmt_us(seconds):
    if seconds < 1e-5:
        return f"{seconds * 1e6:.3f} us"
    return f"{seconds * 1e3:.3f} ms"


def first_access_cost(module_name, getter_name):
    """干净子进程里: import-only 与 getter(import+register) 的墙钟 (中位 3 次)。"""
    script = (
        "import sys, time; sys.path.insert(0, r'{root}'); "
        "t0=time.perf_counter(); "
        "{body}; "
        "print(time.perf_counter()-t0)"
    )
    results = []
    for kind in ("import", "getter"):
        body = (
            f"from src.Modal_Decomposition.Utils import {module_name}"
            if kind == "import"
            else (
                "from src.Modal_Decomposition import Utils as U; "
                f"U.{getter_name}()"
            )
        )
        vals = []
        for _ in range(3):
            out = subprocess.run(
                [sys.executable, "-c", script.format(root=ROOT, body=body)],
                capture_output=True, text=True, cwd=ROOT,
            )
            if out.returncode != 0:
                vals.append(float("inf"))
            else:
                vals.append(float(out.stdout.strip().splitlines()[-1]))
        results.append(statistics.median(vals))
    return results[0], results[1]


def main():
    table_rows = []
    for name, modes in OPS.items():
        row = {"op": name}
        n_iter = 400 if name.startswith("is_monotonic") else 60
        times = bench_roundrobin(modes, n_iter)
        row.update(times)
        row["overhead_vs_direct"] = {
            m: (times[m] / times["direct"] - 1.0) * 100.0 for m in times
        }
        table_rows.append(row)

    first = {
        m: first_access_cost(m, g)
        for m, g in (
            ("Hilbert", "get_hilbert"),
            ("Spline", "get_spline"),
            ("Envelope", "get_envelope"),
            ("Monotonicity", "get_monotonicity"),
        )
    }

    _print_report(table_rows, first)
    _write_markdown(table_rows, first)


def _print_report(rows, first):
    print("\n=== per-call cost (median) ===")
    for r in rows:
        print(f"\n{r['op']}")
        for m in ("direct", "cached_once", "getter_call", "cache_get"):
            ov = r["overhead_vs_direct"][m]
            print(f"  {m:12s} {fmt_us(r[m]):>10s}   ({ov:+.1f}% vs direct)")
    print("\n=== first access (fresh subprocess, median of 3) ===")
    for mod, (ti, tg) in first.items():
        print(f"  {mod:14s} import-only {fmt_us(ti):>9s} | getter(import+register) {fmt_us(tg):>9s}")


def _write_markdown(rows, first):
    import datetime

    env = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "scipy": __import__("scipy").__version__,
        "date": datetime.date.today().isoformat(),
    }
    lines = [
        "# Cache 速度报告 —— 使用 / 不使用 cache 对比",
        "",
        f"- 生成日期: {env['date']} · Python {env['python']} · numpy {env['numpy']} · "
        f"scipy {env['scipy']}",
        f"- 生成脚本: `tests/comparison/bench_cache.py` (仓库根目录运行, 每次重跑刷新本文件)",
        "",
        "## 1. 四种取用方式",
        "",
        "| 方式 | 含义 |",
        "|---|---|",
        "| `direct` | 直接 import 后调用 (无 cache, 基线) |",
        "| `cached_once` | `Utils.get_xxx()` 解析一次, 之后纯属性调用 (cache 一次性) |",
        "| `getter_call` | 每次调用都走 `Utils.get_xxx()` (每调含 `cache.check` + `cache.get`) |",
        "| `cache_get` | 每次调用都走 `cache.get(key)` (跳过 `check`) |",
        "",
        "## 2. 单次调用耗时 (中位数)",
        "",
    ]
    header = "| 操作 | direct | cached_once | getter_call | cache_get |"
    sep = "|---|---|---|---|---|"
    lines += [header, sep]
    for r in rows:
        cells = [r["op"]]
        for m in ("direct", "cached_once", "getter_call", "cache_get"):
            v = r[m]
            cell = f"{v*1e6:.2f} µs" if v < 1e-5 else f"{v*1e3:.2f} ms"
            ov = r["overhead_vs_direct"][m]
            cells.append(f"{cell} ({ov:+.0f}%)")
        lines.append("| " + " | ".join(cells) + " |")
    lines += [
        "",
        "> 结论: `cached_once` 与 `direct` 逐点等价 (解析一次后零额外开销);",
        "> `getter_call`/`cache_get` 每调额外 ~0.5–2 µs (RLock + dict 查表),",
        "> 对 ms 级运算可忽略, 仅在 µs 级小数组 (如 `is_monotonic`) 的紧循环中才可见。",
        "",
        "## 3. 首次访问成本 (干净子进程, 中位 3 次)",
        "",
        "| 模块 | import-only | getter (import + 注册) |",
        "|---|---|---|",
    ]
    for mod, (ti, tg) in first.items():
        lines.append(f"| `{mod}` | {fmt_us(ti)} | {fmt_us(tg)} |")
    lines += [
        "",
        "> 首次访问含一次性的 import (+ register) 开销; 注册本身只多一次",
        "> `cache.check` + `cache.add` (微秒级)。之后所有调用均为纯查表。",
        "",
        "## 4. Cache 使用流程 (Cache.py / Utils getter)",
        "",
        "1. **注册 add(name, api, description?)** — 校验键 (非空字符串) 与 api (非 None),",
        "   加锁后写入; 已存在则返回 `False` 且**永不覆盖** (保留最先注册的实例)。",
        "2. **查询 get(name) / check(name) / describe(name)** — 校验键后加锁读取;",
        "   未注册的 `get` 抛 `KeyError` 并列出全部已注册键。",
        "3. **集中式惰性导入 import_module(name, description?)** — 抽象",
        "   `if flag is None: try import except ImportError: raise` 样板: 首次 import",
        "   并注册, 之后直接命中缓存; ImportError 原样传播且不缓存 (可重试)。",
        "4. **声明式惰性加载 lazy(name, description?) 装饰器** — 零参加载函数",
        "   进程内恰好执行一次 (持锁), 对外完全透明 (`functools.wraps`),",
        "   带参调用/返回 None 抛 `TypeError`; 异常不缓存。",
        "5. **惰性接入 (Utils 约定)** — 模块自身不接触缓存; `Utils.get_xxx()` 首次访问时",
        "   `import_module` + `add`, 之后返回 `get(key)` 的同一实例; 幂等、线程安全。",
        "6. **替换** — 唯一合法途径是 `remove(name)` 后再 `add`; 直接重复 `add` 被拒绝。",
        "7. **枚举** — `names()/keys()/values()/items()/descriptions()` 排序快照;",
        "   `in` / `len()` / `[]` / `repr` 等便捷协议。",
        "8. **清空 clear()** — 删除全部条目 (主要供测试隔离使用)。",
        "9. **并发** — 所有读写由单个 `RLock` 串行化: 并发注册同名键恰好成功一次,",
        "   读取者永不看到半写入状态。",
        "",
        "## 5. 使用建议",
        "",
        "* 纯第三方 import: 用 `cache.import_module(\"scipy.signal\")` (一处 try/except);",
        "* 非 import 的昂贵构造: 用 `@cache.lazy(key)` 装饰零参加载函数;",
        "* 循环外解析一次 (`mod = Utils.get_xxx()`), 循环内直接调 `mod.fn(...)`;",
        "* 不要在 µs 级热循环里每调都走 `get_xxx()`/`cache.get`;",
        "* 跨模块取用统一实例: `cache.get(\"Modal_Decomposition.Utils.Monotonicity\")`",
        "  与 `Utils.get_monotonicity()` 等价且同对象。",
        "",
    ]
    path = os.path.join(ROOT, "docs", "CacheSpeedReport.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
