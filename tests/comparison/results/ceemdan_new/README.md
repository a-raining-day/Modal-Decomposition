# `tests/comparison/results/ceemdan_new/`

CEEMDAN（原生）验证轮的原始数据。配套报告：`docs/test/CEEMDAN_Native_Engine_Report.md`。

## 文件

| 文件 | 产出脚本 | 内容 |
|---|---|---|
| `compare.json` | `.scratch/eemd_new/ceemdan_bench.py` | 四方**等口径**对比：4 个实现 × 4 个信号格 = 16 行 |
| `pareto_sweep.json` | `.scratch/eemd_new/pareto.py` | Pareto 扫描：`M × eps` 16 组 + 前沿 + 选点 |

出图脚本是 `.scratch/eemd_new/fig_ceemdan.py`（只读 JSON，不重跑基准），产物在
`fig/ceemdan_compare.png`、`fig/ceemdan_pareto.png`。

## `compare.json`

顶层键：

| 键 | 值 / 含义 |
|---|---|
| `env` | 环境快照（python / numpy / scipy / PyEMD / pysdkit / CPU …） |
| `M` | `25` —— 本组所有实现的 trials / 集成次数 |
| `eps` | `0.2` —— 噪声**幅值系数**（实际 `σ = eps · std(S)`） |
| `rows` | 16 行结果 |

`rows[]` 每条字段：

| 字段 | 含义 |
|---|---|
| `case` / `n` | 信号代号（`A`/`B`/`C`）与长度 |
| `impl` | 实现标签，见下表 |
| `t_ms` | 单次实测墙钟（毫秒，未做重复取中位） |
| `k` | IMF 行数（不含残差行） |
| `valid` | 满足 `\|zc − ext\| ≤ 1` 的 IMF 行数，形如 `"1/10"` |
| `valid_frac` | 上式的浮点形式 |
| `c37` / `c113` | case A 的 37 Hz / 113 Hz 音调最佳单行绝对相关 |
| `amfm` / `c89` | case B 的 AM-FM / 89 Hz 音调最佳单行绝对相关 |
| `io` | 正交性指数，越接近 0 越好 |
| `leak` | `‖Res‖ / ‖S‖`。**各家残差契约不同，不可横比**（见下方前提 2） |
| `sigma` | 该实现**实际注入**的噪声标准差 |
| `recon` | `max\|ΣIMFs + Res − S\|` |
| `eps` / `M` | 本行使用的系数与集成次数（冗余，便于单行自洽） |

`impl` 取值：

| `impl` | 说明 |
|---|---|
| `CEEMDAN (新)` | 本轮对象 `src/Modal_Decomposition/CEEMDAN_NEW.py`（481 行，**尚未转正**） |
| `CEEMDAN (旧原生)` | 旧 `CEEMDAN.py`（595 行）的可运行副本，由 `.scratch/eemd_new/legacy_ceemdan.py` 动态构建 |
| `PyEMD.CEEMDAN` | `PyEMD.CEEMDAN`，`trials=25`、`epsilon=0.2` |
| `PySDKit.CEEMDAN` | `pysdkit.CEEMDAN`，同上 |

> **必看前提 1（等口径）**：这一组**是**等口径的 —— 四家的噪声都是「系数 × `N(0, std(S))`」
> （旧原生 docstring 明确 `noise_width ≈ epsilon`，PyEMD / PySDKit 用 `epsilon`）。
> 与 `eemd_new/`、`ceemd_new/` 那两个目录不同（那两处的开源对照是峰峰值口径）。

> **必看前提 2（残差契约不一致）**：**PyEMD / PySDKit 的 `Res ≈ 0`**（实测
> `‖Res‖/‖S‖ = 1.7e-16 – 2.2e-16`），它们把趋势也吸进了 IMF；**新 / 旧原生把趋势留在 `Res`**
> （B/4096 上 `leak` 0.258 / 0.257 就是那条二次趋势）。因此 `leak`、`k`、`IO` 三家之间
> **不可直接横比**，只有 `mode1` / `mode2` 可比。

> **必看前提 3（`valid` 低不是 bug）**：四家的 `valid` 都只有 1–5 行。CEEMDAN 每阶只做一次
> `E_1`（`EMD(num_imf=1)`）再集平均，阶段 IMF 天然偏宽带，不满足经典 IMF 的窄带条件。
> 详见报告 §6。

## `pareto_sweep.json`

顶层键：

| 键 | 含义 |
|---|---|
| `env` | 环境快照 |
| `M_grid` | `[10, 25, 50, 100]` |
| `eps_grid` | `[0.1, 0.2, 0.3, 0.4]`，`eps` 就是噪声幅值系数 `noise_amp` |
| `leak_max` | `0.05`，可行性约束 |
| `rows` | 16 组结果 |
| `front` | Pareto 前沿（按耗时升序的 `{eps, M}` 列表，7 点） |
| `best_feasible` | 可行域内最高 `Q` 的点 |
| `pick` | 最终选定的默认点 |
| `pick_rule` | 选点规则的文字说明：**可行域（`leak ≤ 0.05`）内 `Q` 达到最优 99% 的最省点** |

`rows[]` 每条字段：

| 字段 | 含义 |
|---|---|
| `eps` / `M` | 噪声幅值系数与集成次数 |
| `Q` | `(q_A + q_C + q_B) / 3` |
| `q_A` | A/4096 与 A/16384 上 `0.5·(c37 + c113)` 的均值 |
| `q_C` | C/4096 的 IMF 行级合规率 |
| `q_B` | B/4096 的 `0.5·(amfm + c89)` |
| `t_ms` | A/16384 的单次实测墙钟（成本轴） |
| `k_A4` | A/4096 的 `k` |
| `io_A4` | A/4096 的 `io` |
| `c113_A4` | A/4096 的 `c113`（报告里的 `mode2` 列） |
| `leak_A4` | A/4096 的 `leak` |

> **读 `Q` 的提醒**：`q_C` 只有 `0/10`、`1/10`… 这类粗分辨率（跳一格 = 0.1，在 `Q` 里
> 等于 0.033），所以 `Q` 会有明显非单调（例如 `(0.2, 100)` 的 `Q` 反而低于 `(0.2, 50)`）。
> **CEEMDAN 的主导旋钮是 `M` 而不是噪声系数** —— 系数固定 0.3 时 `Q` 随 `M` 单调上升
> （0.6832 → 0.7028 → 0.7551 → 0.7618），而在 `M=50` 处换系数只动 0.047。判断配置好坏
> 优先看 `mode2`（=`c113_A4`）与 `q_A`。

## 重跑

```powershell
# 四方等口径对比（覆盖 compare.json）
.\.venv\Scripts\python.exe -X utf8 -u .scratch\eemd_new\ceemdan_bench.py

# Pareto 扫描（覆盖 pareto_sweep.json；约数分钟）
.\.venv\Scripts\python.exe -X utf8 -u .scratch\eemd_new\pareto.py

# 只重画图（读 JSON，不重跑基准）
.\.venv\Scripts\python.exe -X utf8 -u .scratch\eemd_new\fig_ceemdan.py
```

运行前提：脚本依赖 `.scratch/emd_new/bootstrap.py`（stub 包）与 `legacy.py`（0.3.0 wheel
重建），二者同时负责修复本机 scipy 1.15.3 的 `_codata.py`。并行路径需要能创建命名管道，
受限沙箱下会 `WinError 5`。

## 正确性与性能不在这两个 JSON 里

- `test_ceemdan.py`：串行与**手算**对比、`pre_decompose_noise=True/False` 等价性、重构恒等式、
  退化输入、`dtype`、状态重置。
- `test_ceemdan_parallel.py`：并行与串行对比、切分、可复现性、崩溃注入。
- `test_ceemdan_speed.py`：串行 vs 并行的速度实测（结论：**并行当前无收益**，见报告 §9.2）。

这三个只输出到控制台，**不落盘**；结果已写进报告 §4 与 §9.2。要复核就重跑它们。
