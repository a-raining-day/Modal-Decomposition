# 实验与报告索引（tests/ ↔ docs/）

本索引把仓库 `tests/` 下的实验脚本、原始产物与 `docs/` 下的报告一一对应，
便于复现与追责。所有计时口径为**同进程/同脚本中位计时**（预热后测量），
跨日绝对耗时仅作参考。

## 1. EMD 相关实验

| 主题 | 实验脚本（tests/…） | 原始产物 | docs 报告 |
|---|---|---|---|
| `faster` 两档四向对比（quality/fast × PyEMD/PySDKit） | `comparison/bench_emd_faster.py` | `comparison/results/emd_faster_raw.json` | `docs/EMD_faster_Branch_Comparison_Report.md` |
| 停止判据变体 V0–V8（sd 收紧 / svar / 窄带门） | `comparison/bench_emd_stopping_variants.py` | `comparison/results/emd_stopping_variants_raw.json` | `docs/EMD_Quality_Gap_and_Optimization.md` |
| EMD × PyEMD 十二问逐点对照（E-A…E-H） | `comparison/bench_emd_vs_pyemd_alignment.py` | `comparison/results/emd_vs_pyemd_alignment_raw.json` | `docs/EMD_vs_PyEMD_Detailed_Comparison.md` |
| 三方验证网格（MD/PyEMD/PySDKit × A/B/C × 256–65536） | `comparison/bench_emd_validation.py` + `comparison/refresh_emd_validation_md.py` | `comparison/results/emd_validation_raw.json` | `docs/EMD_Validation_and_Comparison_Report.md` |
| 原生 vs `EMD_new` 性能 + 参数扫描 | `comparison/bench_emd_new.py` | 直接产出 docs 两份 | `docs/EMD_vs_EMD_new_Performance_Report.md`、`docs/EMD_new_Parameter_Sweep_Report.md` |
| 计时·质量三方（原生引擎，现行默认） | `comparison/bench_timing.py --method EMD` | `comparison/results/emd/{timing_raw.csv,timing_metrics.json}` | `docs/EMD_Timing_Memory_Quality_Report.md` |
| 大数据内存轴（三方 RSS，1MB–1GB × increasing/random） | `comparison/bench_memory.py`（**2026-09-10 原生引擎重跑**） | `comparison/results/memory/*.{json,csv}`（旧数据归档于 `_legacy_pyemd_wrapper/memory/`） | `docs/EMD_Large_Signal_Memory_Report.md`（重跑版） |
| 大数据内存矩阵（dtype × 长度 × pattern × chunk 策略） | `test_memory/run_matrix.py`、`test_memory/test_emd_memory.py`（**2026-09-10 原生引擎重跑**） | `test_memory/result_for_each_decomposition/EMD.{json,csv}`（旧数据归档为 `EMD_legacy_pyemd_wrapper.*`） | 同上 |

## 2. 其它方法实验

| 主题 | 实验脚本 | 原始产物 | docs 报告 |
|---|---|---|---|
| VMD（MD-VMD / vmdpy / PySDKit 三列，parity 参数） | `comparison/bench_timing.py --method VMD`、`comparison/summarize_vmd.py` | `comparison/results/vmd/*` | `docs/Decomposition_Methods_Timing_Report.md` |
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
| 组件统计 | `comparison/bench_timing.py`（metrics） | `docs/ComponentStatisticsReport.md` |

## 4. 未成报告 / 推迟的项

| 项 | 状态 |
|---|---|
| `comparison/verify_envelope_modes.py`、`comparison/verify_lmd_recon.py` | 控制台验证脚本（无独立结果文件）；结论并入 `docs/Decomposition_Methods_Timing_Report.md` |
| 3GB 与 ≥1GB 随机噪声内存格 | 推迟记录：`comparison/POSTPONED_3GB.md`；状态汇总见 `docs/EMD_Large_Signal_Memory_Report.md` §2 |
| 历史（PyEMD 包装时代）计时/内存结论 | 数据归档于 `comparison/results/_legacy_pyemd_wrapper/`；结论文档 `comparison/REPORT_EMD_MD_vs_PySDKit.md`（头部已加归档路径说明） |
| `ssa/results/*`（SSA 结果快照） | 已报告：`docs/SSA_Stride_and_Signal_Validation_Report.md` |
| `test_memory` 的 `_runs/*.report.json`（逐格原始报告） | 作为 `docs/EMD_Large_Signal_Memory_Report.md` 的底层证据保留，不单独出报告 |

## 5. tests/ 清理记录

本轮清理（无效/过时文件）:

* 删除 `tests/test_path.py` —— 引用不存在的 `PathDefine.ROOT`，pytest 收集即报错；
* 删除 `tests/test_optimize_decomposition/` —— 早期 EMD 原型草稿
  （`EMD_new.py`：`decompose` 无返回值、依赖已删除接口），无测试价值；
* 删除全部 `__pycache__/` 与 `.pytest_cache/`；
* 删除 `comparison/results/lmd_test{1..5}.log`（临时日志）；
* 删除 `comparison/results/efd/`、`comparison/results/lmd/` —— 被修正版
  `efd_fixed/`、`lmd_fixed/` 取代的旧数据；
* 归档（非删除）`comparison/results/{timing_*.csv,json,md, summary_timing.md,
  summary_memory.md, memory/, memory_after_opt/, vmd/, fmd/, efd_fixed/,
  lmd_fixed/}` → `comparison/results/_legacy_pyemd_wrapper/`，计时类由重跑
  生成新数据；**内存类按用户要求沿用归档数据不重跑**（报告写明适用边界）；
* 旧 `test_memory/result_for_each_decomposition/EMD.{json,csv}` 归档为
  `EMD_legacy_pyemd_wrapper.{json,csv}`（同样沿用，不重跑）。
