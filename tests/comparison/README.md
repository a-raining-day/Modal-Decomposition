# 对比与基准套件（tests/comparison）

本目录是库与外部实现（PyEMD / PySDKit / vmdpy 等）的**基准与验证套件**。
所有结论报告位于 **`docs/`**（见下表"docs 报告"列）；本目录只保留脚本与原始产物。

## 1. 脚本清单

| 文件 | 内容 | docs 报告 |
|---|---|---|
| `bench_emd_faster.py` | EMD `faster` 两档 × PyEMD/PySDKit 四向对比（A/B/C 全网格） | `docs/EMD_faster_Branch_Comparison_Report.md` |
| `bench_emd_stopping_variants.py` | 停止判据变体 V0–V6（sd 收紧 / svar / 窄带门） | `docs/EMD_Quality_Gap_and_Optimization.md` |
| `bench_emd_vs_pyemd_alignment.py` | EMD × PyEMD 十二问逐点对照（E-A…E-H） | `docs/EMD_vs_PyEMD_Detailed_Comparison.md` |
| `bench_emd_validation.py` + `refresh_emd_validation_md.py` | 三方验证网格（256–65536） | `docs/EMD_Validation_and_Comparison_Report.md` |
| `bench_emd_new.py` | 原生 vs `EMD_new` 性能 + 参数扫描 | `docs/EMD_vs_EMD_new_Performance_Report.md`、`docs/EMD_new_Parameter_Sweep_Report.md` |
| `bench_timing.py` | 通用计时轴：`--method EMD\|VMD\|LMD\|FMD\|EFD`，输出到 `results/<method>/` | `docs/EMD_Timing_Memory_Quality_Report.md`、`docs/Decomposition_Methods_Timing_Report.md` |
| `bench_memory.py` + `summarize.py` | 内存轴（1MB–1GB，子进程 + RSS 心跳 + 预算） | `docs/EMD_Large_Signal_Memory_Report.md`（归档数据） |
| `summarize_vmd.py` / `summarize_lmd.py` | VMD/LMD 目录的单次口径 summary（正式报告改用 3 次中位） | 同上 |
| `bench_plot.py` | 出图到 `figs/*.png` | — |
| `bench_cache.py` | 缓存层基准 | `docs/CacheSpeedReport.md` |
| `signals.py` / `quality.py` / `workers.py` / `case_worker.py` | 信号定义、质量指标、实现适配、内存单格 worker | —（身份与参数集中在此） |
| `verify_envelope_modes.py` / `verify_lmd_recon.py` | 包络模式 / LMD 重构校验（控制台） | 结论并入方法报告 |

## 2. 快速开始

```powershell
$env:PYTHONPATH='src'
# 计时 + 质量（EMD）
python tests\comparison\bench_timing.py --method EMD            # → results/emd/
# 其它方法
python tests\comparison\bench_timing.py --method VMD --out tests\comparison\results\vmd
# 内存轴（默认 1MB..1GB × increasing/random，20 s/格预算）
python tests\comparison\bench_memory.py
python tests\comparison\summarize.py
# 出图
python tests\comparison\bench_plot.py
```

## 3. 公平性约定

* 计时: 同进程、轮转顺序、每格多次重复取**中位**；import/预热不计时；每次
  运行前 `gc`，并校验输入未被实现改写（`bench_timing.py` 内置断言）。
* 内存: 每格独立子进程；输入完全建成后才计分解预算；RSS 以 50 ms 心跳采样
  （被 kill 的超时格也有增长轨迹，见 `results/memory/_runs/*.hb.log`）。
* 质量: 所有实现的重构/正交性/模式恢复指标在同一代码路径计算（`quality.py`）。
* 限制: 单机 16 GB RAM，系统负载波动 ±5–10%；亚毫秒格（n=256）的比值不作结论。

## 4. 数据布局

```
results/
  emd/                     # bench_timing.py --method EMD（现行引擎）
  vmd/ lmd_fixed/ fmd/ efd_fixed/   # 各方法计时
  emd_*_raw.json           # EMD 专项实验原始数据
  _legacy_pyemd_wrapper/   # 归档: PyEMD 包装时代的计时/内存数据 + 优化前后对照
  figs/*.png               # 出图
```

* 旧结论报告 `REPORT_EMD_MD_vs_PySDKit.md` 描述 **PyEMD 包装时代**；其引用的
  `results/summary_timing.md`、`results/memory/*` 等已归档到
  `results/_legacy_pyemd_wrapper/`（路径按该前缀解读）。
* 3 GB 与随机 ≥512 MB 的格子按 `POSTPONED_3GB.md` 推迟（只记录不执行）。
* 实验 ↔ 报告总索引见 `docs/EXPERIMENTS_INDEX.md`。
