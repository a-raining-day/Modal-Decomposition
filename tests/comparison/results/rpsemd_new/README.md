# `tests/comparison/results/rpsemd_new/`

RPSEMD（原生）验证轮的原始数据。配套报告：`docs/test/RPSEMD_Native_Engine_Report.md`。

> **本目录的 JSON 是"算法规范修正后"的产物**：`amp`/`freq` 只在第一阶初步 IMF 的
> **极值点**上取（规范①），相位网格是 `θ_i = 2πi/I`（规范②）。修正项同时记录在
> `pareto_sweep.json` 的 `algorithm_corrections` 数组里，机制见报告 §3.3 / §3.4。

## 文件

| 文件 | 产出脚本 | 内容 |
|---|---|---|
| `compare.json` | `.scratch/eemd_new/rpsemd_bench.py`（实验 1） | 四方对比：4 个实现 × 4 个信号格 = 16 行 |
| `pareto_sweep.json` | `.scratch/eemd_new/rpsemd_bench.py`（实验 2） | Pareto 扫描：`separation_ratio × sin_smaple_num × max_clusters` 27 组 + 前沿 + 选点（成本轴 A/16384） |

出图脚本是 `.scratch/eemd_new/fig_rpsemd.py`（只读 JSON，不重跑基准），产物在
`fig/rpsemd_compare.png`、`fig/rpsemd_pareto.png`。

> 两个 JSON 由同一个脚本、同一进程产出（实验 1 + 实验 2），所以两者的 `env` / `params` 是同一份；
> 两者都是**单次计时**。跨日 / 跨脚本的绝对毫秒数不可比。

## `compare.json`

顶层键：

| 键 | 值 / 含义 |
|---|---|
| `env` | 环境快照（python / numpy / scipy / PyEMD / pysdkit / CPU …，含 `legacy` 的 0.3.0 wheel 信息） |
| `params` | `num_imf=10, separation_ratio=1.5, max_clusters=4, sin_smaple_num=8` —— 本组所有实现共用的配置 |
| `rows` | 16 行结果 |

`rows[]` 每条字段：

| 字段 | 含义 |
|---|---|
| `case` / `n` | 信号代号（`A`/`B`/`C`）与长度（A 有 4096 与 16384 两格） |
| `impl` | 实现标签，见下表 |
| `t_ms` | 单次实测墙钟（毫秒，未做重复取中位） |
| `k` | IMF 行数（不含残差行） |
| `valid` | 满足 `\|zc − ext\| ≤ 1` 的 IMF 行数，形如 `"2/8"` |
| `valid_frac` | 上式的浮点形式 |
| `c37` / `c113` | case A 的 37 Hz / 113 Hz 音调最佳单行绝对相关（报告里的 `mode1` / `mode2`） |
| `amfm` / `c89` | case B 的 AM-FM / 89 Hz 音调最佳单行绝对相关（报告里的 `mode1` / `mode2`） |
| `io` | 正交性指数，越接近 0 越好（本引擎可以接近 0 或为负，正负号不能当判据） |
| `leak` | `‖Res‖ / ‖S‖`。**各家停止判据不同，不可横比**（见下方前提 3） |
| `recon` | `max\|ΣIMFs + Res − S\|` |
| `num_imf` / `separation_ratio` / `max_clusters` / `sin_smaple_num` | 本行使用的配置（冗余，便于单行自洽） |

`impl` 取值：

| `impl` | 说明 |
|---|---|
| `RPSEMD (新)` | `src/Modal_Decomposition/RPSEMD.py`（473 行，**已由 `RPSEMD_NEW.py` 转正**） |
| `RPSEMD (旧原生)` | 旧 149 行实现（含 `_Registry`，源码只在提交 `833c1f4`）的可运行副本，由 `.scratch/eemd_new/legacy_rpsemd.py` **自动回落到该提交取源** + 8 条补丁后动态构建（参数名 `M` / `f` / `max_imf` / `fs` / `spline_kind` / `nbsym`） |
| `EMD (新)` | `src/Modal_Decomposition/EMD.py`，**RPSEMD 的基座**（无辅助正弦），作算法参照 |
| `ICEEMDAN (新)` | `src/Modal_Decomposition/ICEEMDAN.py`，同族**噪声辅助**方法，作横向参照 |

> **必看前提 1（没有外部 RPSEMD）**：PyEMD 1.9.0 与 pysdkit 0.5.0 **都没有 RPSEMD**。
> 所以 **`RPSEMD (旧原生)` 是唯一同类对照**；`EMD (新)` / `ICEEMDAN (新)` 只能作算法参照，
> 不参与「谁更准」的结论。

> **必看前提 2（新旧不是同一算法）**：旧版在 `f=None` 时用 **FFT 谱峰**定辅助频率、辅助正弦是
> **单位幅值**、没有 `separation_ratio` / `max_clusters` / `hilbert_mod` / `cluster_weight` 任何
> 聚类参数；新版用**瞬时频率的层次聚类中心（只在极值点上取）+ 簇的平均幅值**。两者只作
> **量级参照**，`k` / `leak` / `IO` 都不可横比，**只有 `mode1` / `mode2`（单行最佳相关）可比**。

> **必看前提 3（`leak` 语义各家不同）**：B/4096 上 0.270（新）/ 0.415（旧）里装的是**真实的二次趋势**，
> 不是噪声；C/4096 上新版 0.012 与旧版 0.017 的语义也不同。判配置好坏优先看 `mode1` / `mode2`。

> **必看前提 4（`valid` 低不是 bug）**：新版只有 **0–3 行**合规（A/16384 是 0/10）。每阶 IMF 是
> 多个相位上单阶 EMD 的**平均**，平均后的行不再严格满足经典 IMF 的窄带必要条件。详见报告 §8.5。

## `pareto_sweep.json`

顶层键：

| 键 | 含义 |
|---|---|
| `env` | 环境快照 |
| `ratio_grid` | `[1.2, 1.5, 2.0]`（`separation_ratio`） |
| `sin_grid` | `[4, 8, 16]`（`sin_smaple_num`） |
| `cluster_grid` | `[4, 8, 16]`（`max_clusters`） |
| `num_imf` | `10`，固定 |
| `leak_max` | `0.05`，可行性约束（作用在 A/4096 的 `leak_A4` 上） |
| `rows` | 27 组结果 |
| `front` | Pareto 前沿（按耗时升序；**本次 2 个点**：`{ratio:1.2, sin_num:4}` 与 `{ratio:1.2, sin_num:8}`） |
| `best_feasible` | 可行域内最高 `Q` 的点（`{ratio, sin_num, max_clusters, Q, t_ms, leak, mode2}`） |
| `pick` | 最终选定的默认点，同上结构 |
| `pick_rule` | 选点规则原文 |
| `inert_knobs` | 惰性旋钮说明（`separation_ratio` / `max_clusters` 的 Q 逐位相同） |
| `algorithm_corrections` | 本轮两处**算法规范修正**的文字记录（极值点取特征 / 相位网格 `2πi/I`） |
| `speedup` | 加速记录（极值点 + `cluster_max_samples=512` 兜底；A/16384 `35121 → 190.7 ms`，184×） |

`rows[]` 每条字段：

| 字段 | 含义 |
|---|---|
| `ratio` / `sin_num` / `max_clusters` | 本组配置 |
| `Q` | `(q_A + q_C + q_B) / 3` |
| `q_A` | A/4096 与 A/16384 上 `0.5·(mode1 + mode2)` 的**均值** |
| `q_C` | C/4096 的 IMF 行级合规率（`valid_frac`） |
| `q_B` | B/4096 的 `0.5·(mode1 + mode2)` |
| `t_ms` | **A/16384 的单次实测墙钟（成本轴）** |
| `k_A4` | A/4096 的 `k` |
| `c113_A4` | A/4096 的 `c113`（报告里的 `mode2` 列） |
| `io_A4` | A/4096 的 `io` |
| `leak_A4` | A/4096 的 `leak` |

选点规则（`pick_rule`）：**可行域（`leak ≤ 5%`）内 `Q` 达到最优 99.5% 的最省点**；
`separation_ratio` 与 `max_clusters` 的 Q 逐位相同（前者因 2 簇即刻满足分离、后者因上界从未触到），
故 `ratio` 取论文推荐 **1.5**、`mc` 取最省的 **4**。本次：

| 锚点 | ratio | sin_num | max_clusters | Q | t_ms | leak | mode2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `best_feasible` | 1.2 | 8 | 4 | 0.7613 | 189.7 | 0.0025 | 0.9550 |
| `pick` | 1.5 | 8 | 4 | 0.7613 | 190.5 | 0.0025 | 0.9550 |

> **读 `Q` 的提醒**：`q_C` 只有 `0.10 / 0.33` 这种粗分辨率（跳一格折进 `Q` 约 ±0.07），
> 所以 `Q` 明显非单调。`sin_num=8` 胜出的关键是它同时拿到 `q_C=0.3333`；
> `sin_num=16` 的 `q_B=0.9955` 其实是全表最高（`q_A=0.9749` 也不差），但 `q_C` 掉到 0.10，
> 把 `Q` 拉到 0.6901。详见报告 §9.5。

## 重跑

```powershell
# 四方对比 + Pareto 扫描（覆盖 compare.json / pareto_sweep.json；约数分钟）
.\.venv\Scripts\python.exe -u .scratch\eemd_new\rpsemd_bench.py

# 只重画图（读 JSON，不重跑基准）-> fig/rpsemd_{compare,pareto}.png
.\.venv\Scripts\python.exe -u .scratch\eemd_new\fig_rpsemd.py

# 契约验证（默认构造 / num_imf 各档 / 退化输入 / dtype / 状态复用 / hilbert / cluster_weight）
.\.venv\Scripts\python.exe -u .scratch\eemd_new\test_rpsemd.py

# 近邻频率分离目标用例（§8.6，打印 4 个频率对 × 4 实现）
.\.venv\Scripts\python.exe -u .scratch\eemd_new\probe_rpsemd_close.py

# 旧 RPSEMD 基线能否复活（打印构造与一次小分解自检；会打印「旧 RPSEMD 来源: 提交 833c1f4」）
.\.venv\Scripts\python.exe -u .scratch\eemd_new\legacy_rpsemd.py
```

运行前提：脚本依赖 `.scratch/emd_new/bootstrap.py`（stub 包）与 `legacy.py`（0.3.0 wheel 重建），
二者同时负责修复本机 scipy 1.15.3 的 `_codata.py`。**本引擎没有并行路径**，所以不需要创建命名管道；
写 `tests/comparison/results/` 与 `fig/` 需要写权限。

> `fig_rpsemd.py` 生成的 Pareto 图面板 (c) 标题是 `no configuration meets the 5% leak limit`，
> 那是**修好 `amp`/`freq` 传参之前**的旧注记；按本 JSON 的实测 27 组 `leak` 全部 ≤ **0.0026**，
> **可行域非空且极宽**。标题字符串尚未同步，读数以 JSON / 报告表格为准。

## 正确性与性能不在这两个 JSON 里

- `test_rpsemd.py`：`num_imf` 各档（1/3/10/-1；注意 `num_imf=10` 在 A/512 上因停止判据只出
  **7 阶**、`-1` 出 **4 阶**）、重构恒等式、退化输入（常量/全零/ramp/x²）、`dtype=float32`、
  状态复用与 `max_clusters` 不被改写、`hilbert_mod ∈ {FFT, FHT}`、`cluster_weight` 0 vs 0.05 不同。
- `probe_rpsemd_speed.py`：一次分解的 `EMD` / `hilbert` / `linkage` / `fcluster` 分项耗时，
  以及 `linkage` 单点曲线（512 → 4096 点，O(N²)）。规范修正后聚类只占 13%（原来 92%）。
- `probe_rpsemd_if.py`：瞬时频率估计器对照。规范修正后只有 `base` / `clamp` 两条能跑
  （`gate` / `sub` 变体会截短特征数组，与"用极值点索引回原数组"的新实现冲突，抛 `IndexError`）。
- `_probe_rpsemd_weight_q.py`（临时）：`cluster_weight` 在 A/B/C 三例上的综合 Q（0.7610 → 0.7650），
  以及第一阶 IMF 的极值点数（A/4096 1203、A/16384 4774、C/4096 2980）。
- `probe_rpsemd_weight.py`：`cluster_weight` ∈ {0, 0.01, 0.05, 0.1, 0.3, 0.5} 的逐档扫描，
  含"纯双音 N=1024"那组（`cw=0 → 0.25684`、`cw≥0.05 → 0.11002`）。
- `probe_rpsemd_close.py`：**近邻频率分离目标用例**（37/40、37/44、37/50、37/74 Hz，两个等幅
  音调 + 1% 噪声，N=2048；判据 `|corr| > 0.9`）。结论：3–13 Hz 内四家都分不开（物理限制），
  八度级上新版 RPSEMD **捕获率最高**（0.9941/0.9675）。

这些脚本**只输出到控制台，不落盘**；结果已写进报告 §4–§8.6。要复核就重跑它们。
