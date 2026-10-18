# 实验与报告索引（tests/ ↔ docs/）

本索引把仓库 `tests/` 下的实验脚本、原始产物与 `docs/` 下的报告一一对应，
便于复现与追责。所有计时口径为**同进程/同脚本中位计时**（预热后测量），
跨日绝对耗时仅作参考。

## 1. EMD 家族（EMD / EEMD / CEEMD / CEEMDAN / ICEEMDAN / RPSEMD）原生引擎验证

六份引擎各有一轮独立验证，报告与数据一一对应：

| 引擎 | 验证内容 | 实验脚本 | 原始产物 | docs 报告 |
|---|---|---|---|---|
| **EMD** | 基线重建与可复现性、三方（EMD_OLD / PyEMD）质量与计时、架构与返回契约、`faster` 门、`nbsym` 口径对齐、`mirror_mod` 评测、内存轴 | `.scratch/emd_new/t00`–`t28`（脚本清单见数据目录 `README.md`） | `tests/comparison/results/emd_new/*.json` | `docs/test/EMD_Native_Engine_Report.md` |
| **EEMD** | 四方对照（本库新 / 旧原生 / PyEMD / PySDKit）、噪声口径对齐、残差恒等式与 0.3.0 wheel 包装缺陷、内层 EMD 档位消融、Pareto 调参与默认值选择、开箱默认对比、并行实现 | `.scratch/eemd_new/bench.py`、`legacy_eemd.py`、`figs.py`、`pareto.py`、`fig_pareto.py`、`defaults_cmp.py`、`verify_fixes.py`、`test_worker_crash.py` | `tests/comparison/results/eemd_new/{compare,inner_tiers,pareto_sweep,defaults}.json` | `docs/test/EEMD_Native_Engine_Report.md` |
| **CEEMD** | 旧版可运行性重建与**算法差异辨析**、并行实现补齐、退化输入与 `dtype` 修复、开箱默认对比、互补性收益、Pareto 调参与默认值选择、并行崩溃检测 | `.scratch/eemd_new/ceemd_bench.py`、`legacy_ceemd.py`、`fig_ceemd.py`、`pareto.py`、`test_ceemd.py`、`test_ceemd_robust.py` | `tests/comparison/results/ceemd_new/{compare,pareto_sweep}.json` | `docs/test/CEEMD_Native_Engine_Report.md` |
| **CEEMDAN** | 半成品补齐（10 项修复 + 3 处注释位）、**每阶 spawn 的并行分支**、四方等口径对比、`valid` 偏低成因、**主导旋钮为 `M` 的 Pareto 调参与默认值选择**、并行性能实测 | `.scratch/eemd_new/ceemdan_bench.py`、`legacy_ceemdan.py`、`fig_ceemdan.py`、`pareto.py`、`test_ceemdan.py`、`test_ceemdan_parallel.py`、`test_ceemdan_speed.py` | `tests/comparison/results/ceemdan_new/{compare,pareto_sweep}.json` | `docs/test/CEEMDAN_Native_Engine_Report.md` |
| **ICEEMDAN** | **局部均值算子 `M(·)` 口径探针**（`x − E_1(x)` vs 单次包络平均）、半成品补齐（10 项修复 + 3 处补齐）、**每阶 spawn 的并行分支与一处竞态修复**、四方对比（旧原生是唯一同类对照）、Pareto 调参与默认值选择（`ε₀=0.2`、`M=100`）、`peaks_backend` 补齐 | `.scratch/eemd_new/iceemdan_bench.py`、`legacy_iceemdan.py`、`fig_iceemdan.py`、`test_iceemdan.py`、`test_iceemdan_parallel.py`、`probe_local_mean.py`、`probe_mirror.py` | `tests/comparison/results/iceemdan_new/{compare,pareto_sweep,local_mean_probe}.json` | `docs/test/ICEEMDAN_Native_Engine_Report.md` |
| **RPSEMD** | **半成品补齐（13 项修复，含 `amp`/`freq` 传反这个根因）**、**「为什么这么慢」的拆解计时与聚类抽稀加速**（4096 点 `linkage` 155 → 1.8 ms、A/16384 端到端 35121 → 237.7 ms）、**瞬时频率估计器评估**（试过、无稳定收益、未采用）、**加权选簇评估**（干净双音有效、基准三例综合 Q 反降，默认不启用）、四方对比（PyEMD / pysdkit 都没有 RPSEMD，旧原生是唯一同类对照）、Pareto 调参与默认值选择（`max_clusters=4`、`sin_smaple_num=8`） | `.scratch/eemd_new/rpsemd_bench.py`、`legacy_rpsemd.py`、`fig_rpsemd.py`、`test_rpsemd.py`、`probe_rpsemd_speed.py`、`probe_rpsemd_speedup.py`、`probe_rpsemd_if.py`、`probe_rpsemd_weight.py` | `tests/comparison/results/rpsemd_new/{compare,pareto_sweep}.json` | `docs/test/RPSEMD_Native_Engine_Report.md` |

> 六份报告开头都有互换链接，且各含一节「同族一致性」，统一列出跨六个类的已知不一致
> （噪声参数口径、`num_imf` 是否必填、`info` 键名、`register_class` 缺失、并行能力）。

> 本表其余行引用的 EMD 脚本与报告（`bench_emd_faster.py`、`bench_emd_stopping_variants.py`、
> `bench_emd_vs_pyemd_alignment.py`、`bench_emd_validation.py`、`refresh_emd_validation_md.py`、
> `bench_emd_new.py`，以及 `docs/EMD_*.md` 全部报告）已在本次重构中删除；其历史结论仅存于
> git 历史与 `comparison/results/_legacy_pyemd_wrapper/`，**不再作为现行依据**。计时与内存轴的
> 现行数字并入 `EMD_Native_Engine_Report.md`（§3 计时、§9 内存）。
>
> EEMD 一侧同理：`comparison/bench_eemd_native.py`、`comparison/results/eemd_native_raw.json`、
> `comparison/results/_eemd_*`、`tests/test_eemd.py` 与 `docs/EEMD_Native_vs_PyEMD_Report.md`
> 已删除，现行数字并入 `EEMD_Native_Engine_Report.md`。
>
> CEEMD 此前没有独立的基准脚本或报告，因此**没有删除记录**；旧 `CEEMD.py` 仍在仓库里
> （跑不起来，作为新旧算法对照的基线使用，见 `CEEMD_Native_Engine_Report.md` §2.1）。
>
> CEEMDAN 同理**没有删除记录**：旧 `CEEMDAN.py`（595 行）跑不起来，同样作为
> 新旧算法对照的基线使用（见 `CEEMDAN_Native_Engine_Report.md` §2.1）；新引擎已由
> `CEEMDAN_NEW.py` **转正为 `CEEMDAN.py`**，旧实现可从提交 **`dd547f2`** 取回。
>
> ICEEMDAN 同理**没有删除记录**：旧的 `ICEEMDAN.py`（255 行、含 `_Registry`）在当前代码树上
> 直接 import 就跑不起来（`ImportError`、抽象方法未实现），需
> `.scratch/eemd_new/legacy_iceemdan.py` 打补丁复活，才能作为新旧算法对照的基线（见
> `ICEEMDAN_Native_Engine_Report.md` §11.2）；旧源码只在提交 **`d3eec7e`**，脚本会自动回落到那里。
> 新引擎已由 `ICEEMDAN_NEW.py` **转正为 `ICEEMDAN.py`**（584 行，提交 `833c1f4`）。
>
> RPSEMD 同理**没有删除记录**：旧 `RPSEMD.py`（**149 行、含 `_Registry`**）**仍在工作区**，
> 在当前代码树上直接 import 就跑不起来，需 `.scratch/eemd_new/legacy_rpsemd.py` 打补丁复活，
> 才能作为新旧算法对照的基线（见 `RPSEMD_Native_Engine_Report.md` §11.2）；补丁脚本优先读工作区
> 这份源码，若工作区已被替换则回落到提交 **`833c1f4`**。**新引擎仍叫 `RPSEMD_NEW.py`（473 行），
> 尚未转正** —— 也就是说 `RPSEMD.py` / `RPSEMD_NEW.py` 现在并存，旧的跑不起来、新的是现役实验对象。
>
> **旧原生基线取源注意**：EEMD / CEEMD 的旧原生实现要从未转正前的提交
> **`dd547f2`** 取回（`git show dd547f2:src/Modal_Decomposition/{EEMD,CEEMD}.py`，
> 分别 479 / 149 行、含 `_Registry`）。`d3eec7e` 及之后的 HEAD 里 `EEMD.py` / `CEEMD.py`
> 已是转正后的新引擎，按 HEAD 取源**无法复现**旧基线。CEEMDAN 同理：旧 595 行实现在
> **`dd547f2`**（`git show dd547f2:src/Modal_Decomposition/CEEMDAN.py`），转正后工作区已无旧文件。

## 2. 其它方法实验

| 主题 | 实验脚本 | 原始产物 | docs 报告 |
|---|---|---|---|
| VMD（MD-VMD / vmdpy / PySDKit 三列，parity 参数） | `comparison/bench_timing.py --method VMD`、`comparison/summarize_vmd.py` | `comparison/results/vmd/*` | `docs/Decomposition_Methods_Timing_Report.md`（**VMD §1 为原生重构前**：脚本当时用的是已废弃参数名，见下行） |
| **VMD 三方重测**（原生实现 vs vmdpy vs PySDKit，3 次重复中位 + 首次 PySDKit 质量列） | `comparison/bench_timing.py --method VMD`、`comparison/summarize_vmd_pysdkit.py` | `comparison/results/vmd_pysdkit/*`、`docs/VMD_Native_vs_vmdpy_vs_PySDKit_Results.json` | `docs/VMD_Three_Way_Comparison_Report.md` |
| LMD（Hilbert-scipy / midpoint / Hilbert-FHT / PySDKit 四列） | `comparison/bench_timing.py --method LMD`、`comparison/summarize_lmd.py` | `comparison/results/lmd_fixed/*` | 同上 |
| FMD（MD-FMD / PySDKit-FMD） | `comparison/bench_timing.py --method FMD` | `comparison/results/fmd/*` | 同上 |
| EFD（MD-EFD / PySDKit-EFD） | `comparison/bench_timing.py --method EFD` | `comparison/results/efd_fixed/*` | 同上 |
| SSA 信号级（stride 1/4/16 × 多音调/宽带/非平稳） | `ssa/test_ssa_signal.py` | `ssa/results/ssa_signal_results.{json,md}` | `docs/SSA_Stride_and_Signal_Validation_Report.md` |
| SSA 规则/边界（秩规则、窗、groups、参数校验） | `ssa/test_ssa_rules.py`、`ssa/test_ssa_stride.py` | —（断言级） | 同上（覆盖说明） |

## 3. 工具层实验（已报告）

| 主题 | 实验脚本 | docs 报告 |
|---|---|---|
| Cache（缓存层）速度与约定 | `comparison/bench_cache.py`、`test_cache.py` | `docs/CacheSpeedReport.md`、`docs/CacheConventions.md` |
| Chunk（分块/内存策略） | `test_chunk.py` | `docs/ChunkReport.md` |
| Peaks（极值检测三后端） | `test_peaks.py` | `docs/PeaksReport.md` |
| **Hilbert 后端**（scipy / FFT 实半谱 / 自适应 FHT / SAO 内核：正确性、计时、内存、零填充伪影、契约不一致、2026-10-19 优化记录） | `comparison/bench_hilbert.py`、`test_hilbert.py` | `docs/Hilbert_Backend_Report.md`、`docs/Hilbert_Backend_Results.json` |
| 组件统计 | `comparison/bench_timing.py`（metrics） | `docs/ComponentStatisticsReport.md` |

## 4. 未成报告 / 推迟的项

| 项 | 状态 |
|---|---|
| `comparison/verify_envelope_modes.py`、`comparison/verify_lmd_recon.py` | 控制台验证脚本（无独立结果文件）；结论并入 `docs/Decomposition_Methods_Timing_Report.md` |
| 3GB 与 ≥1GB 随机噪声内存格 | 推迟记录：`comparison/POSTPONED_3GB.md`；内存轴现行为见 `docs/test/EMD_Native_Engine_Report.md` §9（原 `docs/EMD_Large_Signal_Memory_Report.md` 已删除） |
| 历史（PyEMD 包装时代）计时/内存结论 | 数据归档于 `comparison/results/_legacy_pyemd_wrapper/`；结论文档 `comparison/REPORT_EMD_MD_vs_PySDKit.md`（头部已加归档路径说明） |
| `ssa/results/*`（SSA 结果快照） | 已报告：`docs/SSA_Stride_and_Signal_Validation_Report.md` |
| `test_memory` 的 `_runs/*.report.json`（逐格原始报告） | 作为内存轴的底层证据保留，不单独出报告；现行结论见 `docs/test/EMD_Native_Engine_Report.md` §9（原 `docs/EMD_Large_Signal_Memory_Report.md` 已删除） |

## 5. tests/ 清理记录

本轮清理（无效/过时文件）:

* 删除 `tests/test_path.py` —— 引用不存在的 `PathDefine.ROOT`，pytest 收集即报错；
* 删除 `tests/test_optimize_decomposition/` —— 早期 EMD 原型草稿
  （该目录下的 `EMD_new.py`：`decompose` 无返回值、依赖已删除接口），无测试价值。
  **注意：它是另一份废弃草稿，与后来转正的引擎无关** —— 后者原为
  `src/Modal_Decomposition/EMD_NEW.py`，现已转正为 `src/Modal_Decomposition/EMD.py`；
* 删除 EMD 对照实验的脚本与报告（`comparison/bench_emd_faster.py`、
  `comparison/bench_emd_stopping_variants.py`、`comparison/bench_emd_vs_pyemd_alignment.py`、
  `comparison/bench_emd_validation.py`、`comparison/refresh_emd_validation_md.py`、
  `comparison/bench_emd_new.py` 与全部 `docs/EMD_*.md`），由本轮 `.scratch/emd_new/`
  实验与 `docs/test/EMD_Native_Engine_Report.md` 取代；
* 删除全部 `__pycache__/` 与 `.pytest_cache/`；
* 删除 `comparison/results/lmd_test{1..5}.log`（临时日志）；
* 删除 `comparison/results/efd/`、`comparison/results/lmd/` —— 被修正版
  `efd_fixed/`、`lmd_fixed/` 取代的旧数据；
* 归档（非删除）`comparison/results/{timing_*.csv,json,md, summary_timing.md,
  summary_memory.md, memory/, memory_after_opt/, vmd/, fmd/, efd_fixed/,
  lmd_fixed/}` → `comparison/results/_legacy_pyemd_wrapper/`，计时类由重跑
  生成新数据；**内存类按用户要求沿用归档数据不重跑**（报告写明适用边界）；
* 旧 `test_memory/result_for_each_decomposition/EMD.{json,csv}` 归档为
  `EMD_legacy_pyemd_wrapper.{json,csv}`（同样沿用，不重跑）；
* 删除 **`tests/test_fht.py`**（旧的 Hilbert/FHT 用例：只验 DHT 定义与自逆性，未覆盖
  scipy 对照、非 2 的幂、内存与契约），由 `tests/test_hilbert.py` +
  `tests/comparison/bench_hilbert.py` 取代；其历史结论不再作为依据。
  同轮修 `tests/conftest.py` 的 `set_seed` teardown（`Modal_Decomposition.set_seed` 已不存在，
  导致每个用例 teardown 时 `AttributeError`）。
