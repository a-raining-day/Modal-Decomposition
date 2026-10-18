# `tests/comparison/results/iceemdan_new/`

ICEEMDAN（原生）验证轮的原始数据。配套报告：`docs/test/ICEEMDAN_Native_Engine_Report.md`。

## 文件

| 文件 | 产出脚本 | 内容 |
|---|---|---|
| `compare.json` | `.scratch/eemd_new/iceemdan_bench.py`（实验 1） | 四方**同类/参照**对比：4 个实现 × 4 个信号格 = 16 行 |
| `pareto_sweep.json` | `.scratch/eemd_new/iceemdan_bench.py`（实验 2） | Pareto 扫描：`eps × M` 16 组 + 前沿 + 选点（成本轴 A/16384） |
| `local_mean_probe.json` | `.scratch/eemd_new/probe_local_mean.py` + `probe_mirror.py` | 局部均值算子 `M(·)` 的口径探针：5 组对照 |

出图脚本是 `.scratch/eemd_new/fig_iceemdan.py`（只读 JSON，不重跑基准），产物在
`fig/iceemdan_compare.png`、`fig/iceemdan_pareto.png`。

> `compare.json` 与 `pareto_sweep.json` 由同一个脚本、同一进程产出（实验 1 + 实验 2），
> 所以两者的 `env` / `c1` 是同一份；两者都是**单次计时**。`local_mean_probe.json` 是口径探针的
> 记录（`probe_local_mean.py` / `probe_mirror.py` 打印后人工归档），**不含计时**，也不与上面两个
> JSON 共享运行批次。跨日 / 跨脚本的绝对毫秒数不可比。

## `compare.json`

顶层键：

| 键 | 值 / 含义 |
|---|---|
| `env` | 环境快照（python / numpy / scipy / PyEMD / pysdkit / CPU …，含 `legacy` 的 0.3.0 wheel 信息） |
| `M` | `25` —— 本组所有实现的 trials / 集成次数 |
| `eps` | `0.2` —— 噪声**幅值系数**（新/旧的 `noise_amp` / `epsilon_0`，第 1 阶注入 `eps · std(S)`） |
| `c1` | `0.8329` —— 诊断量 `std(E_1(单位白噪声)) / std(白噪声)`。**β 缩放后会约掉，不参与幅值换算**，只留作记录 |
| `rows` | 16 行结果 |

`rows[]` 每条字段：

| 字段 | 含义 |
|---|---|
| `case` / `n` | 信号代号（`A`/`B`/`C`）与长度（A 有 4096 与 16384 两格） |
| `impl` | 实现标签，见下表 |
| `t_ms` | 单次实测墙钟（毫秒，未做重复取中位） |
| `k` | IMF 行数（不含残差行） |
| `valid` | 满足 `\|zc − ext\| ≤ 1` 的 IMF 行数，形如 `"4/8"` |
| `valid_frac` | 上式的浮点形式 |
| `c37` / `c113` | case A 的 37 Hz / 113 Hz 音调最佳单行绝对相关（报告里的 `mode1` / `mode2`） |
| `amfm` / `c89` | case B 的 AM-FM / 89 Hz 音调最佳单行绝对相关（报告里的 `mode1` / `mode2`） |
| `io` | 正交性指数，越接近 0 越好 |
| `leak` | `‖Res‖ / ‖S‖`。**各家残差契约不同，不可横比**（见下方前提 3） |
| `sigma` / `sigma_eff` | 该实现**实际注入**的噪声标准差（本组两者相同：`eps · std(S)`） |
| `recon` | `max\|ΣIMFs + Res − S\|` |
| `eps` / `M` | 本行使用的系数与集成次数（冗余，便于单行自洽） |

`impl` 取值：

| `impl` | 说明 |
|---|---|
| `ICEEMDAN (新)` | `src/Modal_Decomposition/ICEEMDAN.py`（584 行，提交 `833c1f4` 由 `ICEEMDAN_NEW.py` 转正） |
| `ICEEMDAN (旧原生)` | 旧 255 行实现的可运行副本（源码取自提交 `d3eec7e`），由 `.scratch/eemd_new/legacy_iceemdan.py` 动态构建（参数名 `ensemble_size` / `epsilon_0`） |
| `CEEMDAN (新)` | `src/Modal_Decomposition/CEEMDAN.py`，作为**算法参照**（改进前身） |
| `PyEMD.CEEMDAN` | `PyEMD.CEEMDAN(trials=25, epsilon=0.2)`，外部参照 |

> **必看前提 1（没有外部 ICEEMDAN）**：PyEMD 1.9.0 只有 `CEEMDAN`，pysdkit 0.5.0 没有 ICEEMDAN。
> 所以 **`ICEEMDAN (旧原生)` 是唯一同类对照**，`CEEMDAN (新)` 与 `PyEMD.CEEMDAN` 只能作
> 算法参照，不参与「谁更准」的结论。

> **必看前提 2（口径只有前两列等口径）**：**新实现与旧原生同为 `ε₀` 语义**
> （`β_k = ε₀ · std(r_{k−1}) / std(E_k(w))`，第 1 阶注入恰为 `ε₀ · std(S)`），两列可等口径对比。
> **CEEMDAN / PyEMD 不是**：它们第 1 阶注入**原始噪声**（系数即相对 `std`），其后注入噪声 IMF，
> 没有 `ε₀` 缩放。

> **必看前提 3（残差契约不一致）**：**PyEMD.CEEMDAN 的 `Res` 不是它自己给的** —— PyEMD 把全部
> 数据塞进 IMFs，bench 按 `Res = S − ΣIMFs` 反算，所以它的 `leak` 恒 ≈ 2e-16 是**定义出来的**；
> **新 / 旧原生把趋势留在 `Res`**（B/4096 的 `leak` 0.1545 / 0.2043 就是那条二次趋势）。
> 因此 `leak`、`k`、`IO` 不可直接横比，**只有 `mode1` / `mode2` 可比**。

> **必看前提 4（`valid` 低不是 bug）**：四家的 `valid` 都只有 1–5 行。ICEEMDAN 每阶的 IMF 是
> `r_{k−1} − ⟨M(·)⟩`，集平均会把 M 条不同分解平滑在一起，平滑后的行不再严格满足经典 IMF 的
> 窄带必要条件。详见报告 §6.5。

## `pareto_sweep.json`

顶层键：

| 键 | 含义 |
|---|---|
| `env` | 环境快照 |
| `M_grid` | `[10, 25, 50, 100]` |
| `eps_grid` | `[0.1, 0.2, 0.3, 0.4]`，`eps` 就是噪声幅值系数 `noise_amp`（即 ε₀） |
| `leak_max` | `0.05`，可行性约束（作用在 A/4096 的 `leak_A4` 上） |
| `c1` | 同 `compare.json` 的诊断量，不参与换算 |
| `rows` | 16 组结果 |
| `front` | Pareto 前沿（按耗时升序的 `{eps, M}` 列表，**4 点**） |
| `best_feasible` | 可行域内最高 `Q` 的点（`{eps, M, Q, t_ms, leak, mode2}`） |
| `pick` | 最终选定的默认点，同上结构 |

本 JSON **没有** `pick_rule` 字段；选点规则写在代码 `iceemdan_bench.pick()` 里：
**可行域（`leak ≤ 0.05`）内 `Q` 达到最优 99.5% 的最省点**。本次由 `pick()` 算出的
`best_feasible` 与 `pick` 恰好是同一点（`eps=0.2, M=100`）。

`rows[]` 每条字段：

| 字段 | 含义 |
|---|---|
| `eps` / `M` | 噪声幅值系数与集成次数 |
| `Q` | `(q_A + q_C + q_B) / 3` |
| `q_A` | A/4096 与 A/16384 上 `0.5·(c37 + c113)` 的均值 |
| `q_C` | C/4096 的 IMF 行级合规率 |
| `q_B` | B/4096 的 `0.5·(amfm + c89)` |
| `t_ms` | **A/16384 的单次实测墙钟（成本轴）** |
| `k_A4` | A/4096 的 `k` |
| `c113_A4` | A/4096 的 `c113`（报告里的 `mode2` 列） |
| `io_A4` | A/4096 的 `io` |
| `leak_A4` | A/4096 的 `leak` |

> **读 `Q` 的提醒**：`q_C` 只有 `0.1111 / 0.1250 / 0.2222 / 0.2500 / 0.3333` 这类粗分辨率
> （跳一格折进 `Q` 约 ±0.037，跳到 0.3333 相当于两格），所以 `Q` 会明显非单调。**`(0.2,100)` 的「最优」几乎
> 全部来自 `q_C` 从 0.1250 跳到 0.3333**（这一项就贡献 `+0.0694` 的 `Q`，而 `Q` 总共只涨
> 0.0718）；A / B 两轴在 `ε₀ ≥ 0.2` 之后基本饱和，`M` 从 50 加到 100 耗时 ×2 而 `Q` 只差 5%。
> 详见报告 §7.4。判断配置好坏优先看 `c113_A4`（=`mode2`）与 `q_A`。

## `local_mean_probe.json`

顶层键：

| 键 | 含义 |
|---|---|
| `_comment` | 探针条件与三种 `M(·)` 实现的定义（`envelope` / `envelope+mirror` / `sift`） |
| `env` | `case=A`、`n=4096`、`M=25`、`eps=0.2`、`seed=0` |
| `rows` | 5 组对照：`local_mean ∈ {envelope, envelope+mirror, sift}` × `beta ∈ {none, paper}` |
| `old_native_reference` | 旧原生实现的一次记录（`k=7`、`mode1=0.9977`、`mode2=0.9829`、`io=0.1051`，含 `_source` 说明「改口径前的一次实验 1」） |
| `conclusion` | 文字结论：β 缩放不是主因、边界镜像不是主因，差别在「包络平均只做一次还是筛分到收敛」 |
| `decision` | 采用 `sift` 口径（`M(x)=x−E_1(x)`）+ 单位方差噪声池 + `noise_amp` 即 ε₀；默认 `(0.2, 100)` |
| `caveat` | 实现陷阱：手工 `mirror()` 的 `nbsym` 必须按实际极值数 clamp，否则 `ValueError: nbsym must be <= extrema-1`；本实现不做手工 mirror |

`rows[]` 每条字段：

| 字段 | 含义 |
|---|---|
| `local_mean` | `M(·)` 的实现：`envelope`（单次 `(上+下)/2`）、`envelope+mirror`（同上但先 `mirror(nbsym=4)`）、`sift`（`x − E_1(x)`） |
| `beta` | `none` = 不做 `β_k` 缩放；`paper` = 论文 `β_k` |
| `k` / `mode1` / `mode2` / `io` | 该组的阶数、A 的 37 Hz / 113 Hz 单行相关、正交性指数 |

> 注意：`local_mean_probe.json` 是**探针记录**，行里的 `k/mode1/mode2/io` 由脚本打印后归档；
> 它和 `compare.json` 不是同一次运行（`compare.json` 的 `ICEEMDAN (旧原生)` 是
> 0.9976 / 0.9846 / 0.1104）。两者同档，差异来自判据实现与运行批次，不要混用。

## 重跑

```powershell
# 四方对比 + Pareto 扫描（覆盖 compare.json / pareto_sweep.json；约数分钟）
.\.venv\Scripts\python.exe -u .scratch\eemd_new\iceemdan_bench.py

# 只重画图（读 JSON，不重跑基准）
.\.venv\Scripts\python.exe -u .scratch\eemd_new\fig_iceemdan.py

# 正确性（串行/手算、pre=True/False、退化输入、dtype、状态复用）
.\.venv\Scripts\python.exe -u .scratch\eemd_new\test_iceemdan.py

# 并行验证（等价性 / num_imf=-1 / workers>M / 可复现 / 崩溃注入）
.\.venv\Scripts\python.exe -u .scratch\eemd_new\test_iceemdan_parallel.py

# 局部均值口径探针（打印到控制台，不落盘；JSON 里的行是人工归档的）
.\.venv\Scripts\python.exe -u .scratch\eemd_new\probe_local_mean.py

# 旧 ICEEMDAN 基线能否复活
.\.venv\Scripts\python.exe -u .scratch\eemd_new\legacy_iceemdan.py
```

运行前提：脚本依赖 `.scratch/emd_new/bootstrap.py`（stub 包）与 `legacy.py`（0.3.0 wheel
重建），二者同时负责修复本机 scipy 1.15.3 的 `_codata.py`。并行路径需要能创建命名管道，
受限沙箱下会 `WinError 5`；写 `tests/comparison/results/` 与 `fig/` 也需要写权限。

## 正确性与性能不在这三个 JSON 里

- `test_iceemdan.py`：`pre_decompose_noise=True/False` 等价性、`num_imf` 各档、重构恒等式、
  退化输入、`dtype`、状态重置、直接调用 worker 与进程内参考的对比。
- `test_iceemdan_parallel.py`：并行 vs 串行、`num_imf=-1`、`num_workers > M`、可复现性、崩溃注入。

这两个只输出到控制台，**不落盘**；结果已写进报告 §5 与 §4.3。要复核就重跑它们。
