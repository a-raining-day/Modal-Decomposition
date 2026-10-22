# 对比与基准套件（tests/comparison）

本目录是库与外部实现（PyEMD / PySDKit / vmdpy 等）的**基准与验证套件**，
只保留**脚本**与**原始产物（JSON）**。三类材料的落位：

* 叙述性结论 → `docs/test/`（总索引 `docs/test/EXPERIMENTS_INDEX.md`）；
* 汇总表格（csv/xlsx）→ `table/`（逐 run 的 `timing_raw.csv` 现在写进
  `table/methods/<method>/`，本目录只留 `timing_metrics.json` / `env.json`）；
* 图片 → `fig/comparison/`（`bench_plot.py`、`summarize_*.py` 的输出）。

## 1. 脚本清单

| 文件 | 内容 | 产出 / 报告 |
|---|---|---|
| `bench_timing.py` | 通用计时轴：`--method EMD\|VMD\|LMD\|FMD\|EFD`；逐 run 行写 `table/methods/<method>/timing_raw.csv`，指标写 `results/<method>/timing_metrics.json` | `docs/test/methods/{VMD,LMD,FMD,EFD}_Native_Engine_Report.md` |
| `bench_memory.py` | 内存轴（1 MB–1 GB，子进程 + RSS 心跳 + 预算）；JSON 留 `results/memory/`，csv 写 `table/memory/` | `docs/test/methods/Large_Array_Memory_Report.md` |
| `summarize.py` / `summarize_vmd.py` / `summarize_lmd.py` / `summarize_vmd_pysdkit.py` | 从 `table/methods/<m>/timing_raw.csv` + `results/<m>/timing_metrics.json` 生成 md 摘要与图 | 同上 |
| `bench_plot.py` | 出图到 `fig/comparison/*.png` | — |
| `bench_hilbert.py` / `bench_fft.py` | Hilbert / FFT 后端基准；结果 JSON 写 `results/utils/{hilbert,fft}/` | `docs/test/utils/{Hilbert,FFT}_Backend_Report.md` |
| `bench_slepian.py` | Slepian 三后端基准 | `docs/test/utils/Slepian_Backend_Report.md` |
| `bench_cache.py` | 缓存层基准（现行数值见深度测试） | `docs/test/base/Cache_Engine_Report.md` §5.2 |
| `signals.py` / `quality.py` / `workers.py` / `case_worker.py` | 信号定义、质量指标、实现适配、内存单格 worker | —（身份与参数集中在此） |
| `verify_envelope_modes.py` / `verify_lmd_recon.py` | 包络模式 / LMD 重构校验（控制台） | 结论并入方法报告 |
| `bench_vmd_native.py` / `bench_vmd_engine.py` / `bench_ewt_vs_ewtpy.py` | 三个方法的原生 vs 外部实现对照 | 并入对应方法报告 |

> **已被取代的旧脚本**：`bench_emd_faster.py`、`bench_emd_stopping_variants.py`、
> `bench_emd_vs_pyemd_alignment.py`、`bench_emd_validation.py`、
> `refresh_emd_validation_md.py`、`bench_emd_new.py` 及其 `docs/EMD_*.md` 报告已删除，
> 现行结论见 `docs/test/methods/EMD_Native_Engine_Report.md`。

## 2. 快速开始

```powershell
$env:PYTHONPATH='src'
# 计时 + 质量（EMD）→ 原始 JSON 到 results/emd/，逐 run csv 到 table/methods/emd/
python tests\comparison\bench_timing.py --method EMD
# 其它方法
python tests\comparison\bench_timing.py --method VMD --out tests\comparison\results\vmd
# 内存轴（默认 1MB..1GB × increasing/random，20 s/格预算）
python tests\comparison\bench_memory.py
python tests\comparison\summarize.py
# 出图 → fig/comparison/
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
  env.json                        # 解释器/依赖版本快照
  emd_new/ eemd_new/ ceemd_new/ ceemdan_new/ iceemdan_new/ rpsemd_new/   # 六个引擎的原始 JSON
  pareto/<METHOD>.json            # 15 个方法的 Pareto 扫描
  vmd/ vmd_pysdkit/ lmd_fixed/ fmd/ efd_fixed/   # 计时指标 JSON
  utils/{hilbert,fft}/            # 后端基准 JSON
  memory/                         # 内存轴 JSON + _runs/ 原始报告
  _legacy_pyemd_wrapper/          # 归档: PyEMD 包装时代的计时/内存数据
```

* 对应的**汇总表格**在 `table/`（`table/methods/<m>/timing_raw.csv`、
  `table/memory/`、`table/size_axis/`、`table/index.xlsx`）。
* 旧结论报告 `REPORT_EMD_MD_vs_PySDKit.md` 描述 **PyEMD 包装时代**；其引用的
  `results/summary_timing.md`、`results/memory/*` 等已归档到
  `results/_legacy_pyemd_wrapper/`（路径按该前缀解读）。
* 3 GB 与随机 ≥512 MB 的格子按 `POSTPONED_3GB.md` 推迟（只记录不执行）。
* 实验 ↔ 报告总索引见 `docs/test/EXPERIMENTS_INDEX.md`。
