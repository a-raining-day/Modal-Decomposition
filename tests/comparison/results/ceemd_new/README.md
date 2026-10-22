# `tests/comparison/results/ceemd_new/`

CEEMD（原生）验证轮的原始数据。配套报告：`docs/test/methods/CEEMD_Native_Engine_Report.md`。

## 文件

| 文件 | 产出脚本 | 内容 |
|---|---|---|
| `compare.json` | `.scratch/eemd_new/ceemd_bench.py` | 开箱默认对比：5 个实现 × 4 个信号格 = 20 行 |
| `pareto_sweep.json` | `.scratch/eemd_new/pareto.py` | Pareto 扫描：`M × alpha` 16 组 + 前沿 + 选定点 |

出图脚本是 `.scratch/eemd_new/fig_ceemd.py`（只读 JSON，不重跑基准），产物在
`fig/methods/ceemd/ceemd_compare.png`、`fig/methods/ceemd/ceemd_pareto.png`。

## `compare.json`

顶层键：`env`（环境快照）、`rows`（数组）。

`rows[]` 每条字段：

| 字段 | 含义 |
|---|---|
| `case` / `n` | 信号代号（`A`/`B`/`C`）与长度 |
| `impl` | 实现标签（见下表） |
| `t_ms` | 单次实测墙钟（毫秒，**未做重复取中位**） |
| `k` | IMF 行数（不含残差行） |
| `valid` | 满足 `\|zc − ext\| ≤ 1` 的 IMF 行数，形如 `"2/7"` |
| `valid_frac` | 上式的浮点形式 |
| `c37` / `c113` | case A 的 37 Hz / 113 Hz 音调最佳单行绝对相关 |
| `amfm` / `c89` | case B 的 AM-FM / 89 Hz 音调最佳单行绝对相关 |
| `io` | 正交性指数，越接近 0 越好 |
| `leak` | `‖Res‖ / ‖S‖`；**只在真实残差≈0 的信号上是噪声代理**（case B 含真实趋势，不能当噪声读） |
| `sigma` | 该实现**实际注入**的噪声标准差（不是名义参数） |
| `recon` | `max\|ΣIMFs + Res − S\|` |

`impl` 取值与含义：

| `impl` | 说明 |
|---|---|
| `CEEMD (新, 默认)` | 本轮对象 `src/Modal_Decomposition/CEEMD.py`（原型阶段叫 `CEEMD_NEW.py`），默认 `num_imf=-1, noise_amp=None→系数 0.3, M=25` |
| `CEEMD (旧, 默认)` | 旧 `CEEMD.py` 的可运行副本（另一套算法，见报告 §2.1 / §4），由 `.scratch/eemd_new/legacy_ceemd.py` 动态构建 |
| `EEMD (同库, 默认)` | 同库 EEMD，作为"互补对到底带来多少收益"的参照 |
| `PyEMD.EEMD (默认)` | `PyEMD.EEMD()`，`trials=100`、`noise_width=0.05`（峰峰值口径） |
| `PySDKit.EEMD (默认)` | `pysdkit.EEMD()`，同上 |

> **必看前提 1**：这一组是**各家用自己的默认参数**，**不是等口径对照**。新 CEEMD 的
> `noise_amp` 是**幅值系数**（实际 `σ̂ = noise_amp · std(S)`），另三家的 `noise_width` 是
> **峰峰值**口径，所以 `sigma` 列各不相同。报告 §5 表注写明了这一点。
>
> **必看前提 2**：旧 CEEMD 是**另一套算法**（逐阶只剥 1 个 IMF），它的 `k`=14–16 与
> `leak`=0.87–0.89 不能当成新版的退化版本。

## `pareto_sweep.json`

顶层键：

| 键 | 含义 |
|---|---|
| `env` | 环境快照 |
| `M_grid` | `[10, 25, 50, 100]` |
| `alpha_grid` | `[0.1, 0.2, 0.3, 0.4]`，`alpha` 就是幅值系数 `noise_amp`（`σ̂ = alpha · std(S)`） |
| `cost_case` | `["A", 16384]` —— 成本轴的信号 |
| `leak_max` | `0.05`，可行性约束 |
| `rows` | 16 组结果 |
| `front` | Pareto 前沿（按耗时升序的 `{alpha, M}` 列表） |
| `best_feasible` | 可行域内最高 `Q` 的点 |
| `pick` | 最终选定的默认点 |

`rows[]` 每条字段：

| 字段 | 含义 |
|---|---|
| `alpha` / `M` | 噪声强度（相对 `std(S)`）与互补对数 |
| `Q` | `(q_A + q_C + q_B) / 3` |
| `q_A` | A/4096 与 A/16384 上 `0.5·(c37 + c113)` 的均值 |
| `q_C` | C/4096 的 IMF 行级合规率（**只有 0/7、1/7 这类粗分辨率**） |
| `q_B` | B/4096 的 `0.5·(amfm + c89)` |
| `t_ms` | A/16384 的单次实测墙钟（成本轴） |
| `k_A4` | A/4096 的 `k` |
| `io_A4` | A/4096 的 `io` |
| `c113_A4` | A/4096 的 `c113`（报告里的 `mode2` 列） |
| `leak_A4` / `leak_A16` | A/4096 与 A/16384 的 `leak` |

> **读 `Q` 的提醒**：`q_C` 只有 8 个可能取值，跳一格就是 0.1429，在 `Q` 里等于 0.0476。
> 因此 `Q` 会出现明显非单调（例如 `alpha=0.2/M=25` 的 `Q` 低于 `alpha=0.1/M=25`，但同组
> `q_A`、`q_B` 都是上升的）。判断配置好坏请优先看 `mode2`（=`c113_A4`）与 `q_A`。

## 重跑

```powershell
# 开箱默认对比（覆盖 compare.json）
.\.venv\Scripts\python.exe -X utf8 -u .scratch\eemd_new\ceemd_bench.py

# Pareto 扫描（覆盖 pareto_sweep.json；约数分钟）
.\.venv\Scripts\python.exe -X utf8 -u .scratch\eemd_new\pareto.py

# 只重画图（读 JSON，不重跑基准）
.\.venv\Scripts\python.exe -X utf8 -u .scratch\eemd_new\fig_ceemd.py
```

运行前提：脚本依赖 `.scratch/emd_new/bootstrap.py`（stub 包）与 `legacy.py`（0.3.0 wheel
重建），二者同时负责修复本机 scipy 1.15.3 的 `_codata.py`。并行路径需要能创建命名管道，
受限沙箱下会 `WinError 5`。

## 正确性与稳健性不在这两个 JSON 里

`test_ceemd.py`（串行/并行与手工算术平均逐位对比、崩溃注入、互补性收益）与
`test_ceemd_robust.py`（退化输入、`dtype`、契约、参数校验）只输出到控制台，**不落盘**；
它们的结果直接写进了报告 §3.3。要复核就重跑这两个脚本。
