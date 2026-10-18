# `tests/comparison/results/eemd_new/` — EEMD 原生引擎对照实验原始数据

本目录存放 EEMD（原生）对照轮的四份原始结果。**分析与结论见
[`docs/test/EEMD_Native_Engine_Report.md`](../../../../docs/test/EEMD_Native_Engine_Report.md)**，
本文件只说明数据本身。

| 文件 | 实验 | M | α | 行数 | 产出脚本 |
|---|---|---:|---:|---:|---|
| `compare.json` | 实验 1：四方对比（本库新 / 本库旧原生 / PyEMD / PySDKit + wheel 包装作参考） | 50 | 0.2 | 28 | `.scratch/eemd_new/bench.py` |
| `inner_tiers.json` | 实验 2：内层 EMD 档位消融（速度档 vs 质量档 i100 / i300） | 20 | 0.2 | 12 | `.scratch/eemd_new/bench.py` |
| `pareto_sweep.json` | 实验 3：`M` × `alpha` 二维 Pareto 扫描 | 10/25/50/100/200 | 0.05–0.40 | 25 | `.scratch/eemd_new/pareto.py` |
| `defaults.json` | 实验 4：开箱默认对比（每家用它自己的默认参数） | 各自默认 | 各自默认 | 16 | `.scratch/eemd_new/defaults_cmp.py` |

`α` 的含义：`alpha` **就是本库的幅值系数 `noise_amp`**，实际噪声 `σ̂ = α · std(S)`。各引擎的传参换算见报告 §4
（本库 `noise_amp = α = σ̂ / std(S)`；PyEMD / PySDKit / 旧原生都是峰峰值口径 `noise_width = σ̂ / |max(S)−min(S)|`）。

> ⚠️ 实验 1/2/3 是**统一 α 的同口径对照**；**实验 4 不是** —— 它让每家走自己的默认
> （本库 `alpha=0.3`、另三家 `noise_width=0.05` 峰峰值），所以 `defaults.json` 每行额外记了
> `sigma`（该实现实际注入的噪声标准差，绝对值）供核对。

---

## 文件结构

`compare.json` / `inner_tiers.json` 的顶层结构相同：

```jsonc
{
  "env":  { ... },     // 环境快照 + scipy 修复状态 + 旧基线构建来源
  "M":    50,          // 本实验的集成长度
  "alpha": 0.2,        // σ̂ = alpha * std(S)
  "rows": [ ... ]      // 结果行，见下表
}
```

`pareto_sweep.json` 与 `defaults.json` 的结构见本文末尾的「另两份 JSON」一节。

### `rows[]` 每行的字段

| 字段 | 类型 | 含义 |
|---|---|---|
| `case` | `str` | 测试信号代号：`A`（强+弱音调+噪声）、`B`（AM-FM+音调+趋势）、`C`（白噪声） |
| `n` | `int` | 采样点数 |
| `impl` | `str` | 实现标签，取值见下 |
| `alpha` | `float` | 本行使用的 `α` |
| `t_ms` | `float` | 耗时（毫秒）。**实验 1 为单次计时；实验 2 未预热**，只作量级参考 |
| `k` | `int` | IMF 行数（不含残差行） |
| `valid` | `str` | `"满足 \|zc−ext\| ≤ 1 的行数 / k"` |
| `valid_frac` | `float` | 上式的比值 |
| `c1` | `float \| null` | 主分量最佳单行 \|corr\|：case A = 37 Hz、case B = AM-FM、case C = `null` |
| `c2` | `float \| null` | 次分量最佳单行 \|corr\|：case A = 113 Hz、case B = 89 Hz、case C = `null` |
| `io` | `float` | 正交性指数 `2·Σ_{i<j}⟨c_i,c_j⟩ / Σ‖c_i‖²`，越接近 0 越好 |
| `recon` | `float` | `max\|ΣIMFs + Res − S\|`，重建恒等式的最大偏离 |
| `M` | `int` | 本行使用的集成长度 |

### `impl` 的取值

**`compare.json`**

| 标签 | 是什么 |
|---|---|
| `EEMD (原生/新)` | `src/Modal_Decomposition/EEMD.py`，本轮转正的原生实现 |
| `EEMD_OLD (原生/旧)` | `git show HEAD:src/Modal_Decomposition/EEMD.py` 的旧原生实现，经 `.scratch/eemd_new/legacy_eemd.py` 注入等价 `resolve_workers` 后构建到 `Legacy_MD/EEMD_native.py`；**内部用 0.3.0 的旧 EMD** |
| `PyEMD 1.9` | `PyEMD.EEMD`；残差由 `S − ΣIMFs` 另行计算（不用它自己的包装） |
| `PySDKit 0.5` | `pysdkit.EEMD`；已按峰峰值口径换算传参（见下方「前提 1」的留档） |
| `wheel 包装 (参考)` | `dist/modal_decomposition-0.3.0-*.whl` 里的 `EEMD.py`，只是 PyEMD 的薄包装且残差有缺陷（`IMFs=result[:-1]`、`Res=result[-1]`），仅作参考 |

**`inner_tiers.json`**

| 标签 | 内层 EMD 参数 |
|---|---|
| `内层=速度档 (默认)` | `emd_faster=True`（默认即 EMD 轮的 Pareto 拐点：`nbsym=4, internal_delta=0.2, internal_iter=50`） |
| `内层=质量档 i100` | `emd_faster=False, emd_internal_delta=0.002, emd_internal_iter=100` |
| `内层=质量档 i300` | `emd_faster=False, emd_internal_delta=0.002, emd_internal_iter=300` |

---

## ⚠️ 读数据前必看的两个前提

1. **四列已是同口径数据（口径）**。`compare.json` 的四方对比统一取 `σ̂ = α·std(S)`：
   本库 EEMD 的 `noise_amp` 是**相对 `std(S)` 的幅值系数**（`σ̂ = noise_amp · std(S)`），
   而旧原生 / PyEMD / PySDKit 的 `noise_width` 是**峰峰值口径**，须传 `σ̂ / |max(S)−min(S)|`。
   > 留档：本文件**首版**的 `run_pysdkit` 曾误传 `noise_width = σ̂`，而 pysdkit
   > （`.venv/Lib/site-packages/pysdkit/_emd/eemd.py:244`）内部会再乘一次峰峰值
   > （`scale = noise_width * |max−min|`），导致该列噪声被放大 |max−min| 倍、分解提前停，
   > `IO` 出现 0.49 / 0.80 / 0.35 的假象。该 bug 已修正并整组重跑，**当前文件里的 PySDKit
   > 列是修正后的同口径数据**。若你手上有更早的副本，请以本目录为准。

2. **四份文件的耗时不可互相比较。**
   `compare.json` / `pareto_sweep.json` / `defaults.json` 都是单次计时；`inner_tiers.json`
   **未预热**，只用于比较同一实验内部的相对量级。跨日 / 跨脚本的绝对毫秒数也不可比。

---

## 另两份 JSON

### `pareto_sweep.json`（实验 3）

顶层字段除 `rows` 外还有：

| 字段 | 含义 |
|---|---|
| `M_grid` | 扫描的集成长度网格 `[10, 25, 50, 100, 200]` |
| `alpha_grid` | 扫描的噪声系数网格 `[0.05, 0.1, 0.2, 0.3, 0.4]` |
| `cost_case` | 成本轴所用的算例，固定 `["A", 16384]` |
| `leak_max` | 可行域阈值 `0.05`（`leak ≤ 5%`） |
| `rows` | 25 行结果，字段见下 |
| `front` | Pareto 前沿点（`{alpha, M}` 列表，4 个点） |
| `knee` | 无约束弦距拐点 `{alpha, M, Q, t_ms, leak}` |
| `knee_feasible` | 加 `leak ≤ 5%` 后的弦距拐点 |
| `best_feasible` | 可行域内 `Q` 最高点 |

`rows[]` 的字段：`M`、`alpha`、`t_ms`（成本轴 A/16384）、`q_A`、`q_C`、`q_B`、`Q`、
`k_A4`（A/4096 的 IMF 行数）、`io_A4`、`c113_A4`（A/4096 的 113 Hz 单行 \|corr\|，
即报告里的 `mode2`）、`leak_A4`、`leak_A16`、`valid_C4`、`valid_C16`。

### `defaults.json`（实验 4）

顶层只有 `rows`（16 行 = 4 个算例 × 4 个实现）。字段：

| 字段 | 含义 |
|---|---|
| `case` / `n` | 算例与长度 |
| `impl` | 实现标签，形如 `EEMD (原生/新) 默认 M=50, alpha=0.3`、`EEMD_OLD (原生/旧) 默认 trials=100, nw=0.05` |
| `t_ms`、`k`、`valid`、`c1`、`c2`、`io`、`recon` | 与其它 JSON 同义 |
| `leak` | `‖Res‖/‖S‖`（A/4k 口径）。⚠️ case B 的残差含真实二次趋势，其 `leak` 0.25–0.28 **不是噪声** |
| `sigma` | 该实现**实际注入**的噪声标准差（绝对值），用于核对口径 |

> ⚠️ 本文件**不是等口径数据**：每家走自己的默认（本库 `alpha=0.3`、另三家
> `noise_width=0.05` 峰峰值），`sigma` 列就是差异的证据。

---

## 如何重跑

```powershell
$env:PYTHONPATH = "$PWD\.scratch\emd_new"   # 以下脚本都依赖其中的 bootstrap.py / legacy.py
.\.venv\Scripts\python.exe -X utf8 -u .scratch\eemd_new\bench.py         # 实验 1 + 2
.\.venv\Scripts\python.exe -X utf8 -u .scratch\eemd_new\pareto.py        # 实验 3
.\.venv\Scripts\python.exe -X utf8 -u .scratch\eemd_new\defaults_cmp.py  # 实验 4
```

- 会**覆盖**本目录的两份 JSON，并重新打印表格。
- `bench.py` 依赖链：`.scratch/emd_new/bootstrap.py`（stub 包 + 修本机 scipy 的
  `_codata.py`）、`.scratch/emd_new/legacy.py`（0.3.0 wheel 重建）、
  `.scratch/eemd_new/legacy_eemd.py`（旧原生 EEMD 重建）、`tests/comparison/signals.py`
  与 `tests/comparison/harness.py`（信号与指标）。
- 只重画图不重跑基准：

  ```powershell
  .\.venv\Scripts\python.exe -X utf8 -u .scratch\eemd_new\figs.py
  ```
  （输出到 `fig/eemd_speed.png`、`fig/eemd_quality.png`、`fig/eemd_recon.png`、`fig/eemd_inner.png`）
- 并行路径需要能创建命名管道；受限沙箱下会 `PermissionError: [WinError 5]`，属环境限制。
