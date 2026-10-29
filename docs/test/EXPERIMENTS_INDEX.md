# 实验与报告索引（tests/ ↔ table/ ↔ fig/ ↔ docs/）

本索引把**脚本、原始产物、汇总表格、图片、报告**一一对应，便于复现与追责。
四类材料的落位规则（本轮重构后强制）：

| 目录 | 放什么 | 例子 |
|---|---|---|
| `tests/` | 可执行脚本 + **原始逐格 JSON** | `tests/utils/test_Check_deep.py`、`tests/comparison/results/emd_new/pareto_sweep.json` |
| `table/` | **汇总表格**（csv / xlsx） | `table/size_axis/monotonicity_monotonic_full_scan.csv` |
| `fig/` | **图片**（png），按 `methods/`、`utils/`、`base/`、`comparison/`、`size_axis/` 分级 | `fig/utils/check/check_matrix.png` |
| `docs/` | **叙述性报告**（md） | `docs/test/utils/Check_Engine_Report.md` |

自检：`python tests/tools/check_links.py`（校验全部 md 链接与代码里点名的资源路径）。

计时口径：**同进程、warm-up 后重复取 best/median**；跨日绝对耗时仅作参考。
规模轴里的 **MB 一律指 MiB（2²⁰ B）**。

---

## 1. EMD 家族（EMD / EEMD / CEEMD / CEEMDAN / ICEEMDAN / RPSEMD）原生引擎验证

| 引擎 | 验证内容 | 实验脚本 | 原始产物 | docs 报告 |
|---|---|---|---|---|
| **EMD** | 三方（EMD_OLD / PyEMD）质量与计时、架构与返回契约、`faster` 门、`nbsym` 口径、`mirror_mod`、内存轴 | `.scratch/emd_new/t00`–`t28`（清单见数据目录 `README.md`） | `tests/comparison/results/emd_new/*.json` | `docs/test/methods/EMD_Native_Engine_Report.md` |
| **EEMD** | 四方对照、噪声口径对齐、残差恒等式、内层 EMD 档位消融、Pareto 调参、并行实现 | `.scratch/eemd_new/{bench,legacy_eemd,figs,pareto,fig_pareto,defaults_cmp,verify_fixes,test_worker_crash}.py` | `tests/comparison/results/eemd_new/{compare,inner_tiers,pareto_sweep,defaults}.json` | `docs/test/methods/EEMD_Native_Engine_Report.md` |
| **CEEMD** | 旧版可运行性重建与算法差异辨析、并行补齐、退化输入与 dtype 修复、互补性收益、Pareto | `.scratch/eemd_new/{ceemd_bench,legacy_ceemd,fig_ceemd,pareto,test_ceemd}.py` | `tests/comparison/results/ceemd_new/{compare,pareto_sweep}.json` | `docs/test/methods/CEEMD_Native_Engine_Report.md` |
| **CEEMDAN** | 10 项修复、每阶 spawn 并行分支、四方对比、`valid` 偏低成因、Pareto（主导旋钮 `M`） | `.scratch/eemd_new/{ceemdan_bench,legacy_ceemdan,fig_ceemdan,test_ceemdan,test_ceemdan_parallel}.py` | `tests/comparison/results/ceemdan_new/{compare,pareto_sweep}.json` | `docs/test/methods/CEEMDAN_Native_Engine_Report.md` |
| **ICEEMDAN** | 局部均值算子探针、10 项修复、每阶 spawn 并行与竞态修复、四方对比、Pareto（`ε₀=0.2`、`M=100`） | `.scratch/eemd_new/{iceemdan_bench,legacy_iceemdan,fig_iceemdan,test_iceemdan,test_iceemdan_parallel,probe_local_mean,probe_mirror}.py` | `tests/comparison/results/iceemdan_new/{compare,pareto_sweep,local_mean_probe}.json` | `docs/test/methods/ICEEMDAN_Native_Engine_Report.md` |
| **RPSEMD** | 13 项修复（含 `amp`/`freq` 传反根因）+ 两处规范修正、O(N²) 消失、瞬时频率评估、加权选簇评估、近邻频率分离用例、Pareto | `.scratch/eemd_new/{rpsemd_bench,legacy_rpsemd,fig_rpsemd,test_rpsemd,probe_rpsemd_*}.py` | `tests/comparison/results/rpsemd_new/{compare,pareto_sweep}.json` | `docs/test/methods/RPSEMD_Native_Engine_Report.md` |

> 旧原生基线取源：EEMD / CEEMD / CEEMDAN 用提交 `dd547f2`；ICEEMDAN 用 `d3eec7e`；
> RPSEMD 用 `833c1f4`（脚本按"工作区源码里还含不含 `_Registry`"自动回落）。

## 2. 其它方法实验

| 主题 | 实验脚本 | 原始产物 | docs 报告 |
|---|---|---|---|
| VMD（MD-VMD / vmdpy，parity 参数） | `comparison/bench_timing.py --method VMD`、`summarize_vmd.py` | `tests/comparison/results/vmd/*` → 计时表 `table/methods/vmd/` | `docs/test/methods/VMD_Native_Engine_Report.md` |
| VMD 三方重测 | `comparison/bench_timing.py --method VMD`、`summarize_vmd_pysdkit.py` | `tests/comparison/results/vmd_pysdkit/*` → `table/methods/vmd_pysdkit/` | 同上 |
| LMD（Hilbert-scipy / midpoint / Hilbert-FHT / PySDKit） | `comparison/bench_timing.py --method LMD`、`summarize_lmd.py` | `tests/comparison/results/lmd_fixed/*` → `table/methods/lmd_fixed/` | `docs/test/methods/LMD_Native_Engine_Report.md` |
| FMD（MD-FMD / PySDKit-FMD） | `comparison/bench_timing.py --method FMD` | `tests/comparison/results/fmd/*` → `table/methods/fmd/` | `docs/test/methods/FMD_Native_Engine_Report.md` |
| EFD（MD-EFD / PySDKit-EFD） | `comparison/bench_timing.py --method EFD` | `tests/comparison/results/efd_fixed/*` → `table/methods/efd_fixed/` | `docs/test/methods/EFD_Native_Engine_Report.md` |
| 其余 6 个方法（CEEFD/MEMD/SSA/EWT/SVMD + Pareto 扫描） | `.scratch/eemd_new/pareto_all.py`、`pareto_report.py` | `tests/comparison/results/pareto/<METHOD>.json` | `docs/test/methods/<METHOD>_Native_Engine_Report.md` |
| EFD / FMD 对 PySDKit 同口径基准（本轮） | `comparison/bench_efd_fmd_native.py`（`--mode compare` / `grid`） | `tests/comparison/results/efd_fmd_native/*.json` → `table/methods/efd_fmd_native/{compare,fmd_grid}.csv` | `docs/test/methods/EFD_FMD_vs_PySDKit_Report.md` |
| 大数组内存（100 MB / 500 MB / 1 GB） | `.scratch/eemd_new/{bench_tier,probe_one,bench_large}.py` | `tests/comparison/results/memory*` | `docs/test/methods/Large_Array_Memory_Report.md` |

## 3. 工具层深度测试（本轮，13 个模块）

每个模块一份 `pytest` 文件 + 一张汇总表 + 一张图 + 报告里的「深度测试」小节。
用例数 = `pytest -q` 的实际收集数。

| 模块 | 测试文件 | 用例 | 数据（`table/`） | 图（`fig/`） | 报告 |
|---|---|---|---|---|---|
| `Utils.Check` | `tests/utils/test_Check_deep.py` | 89 | `utils/check/{check_dtype_matrix,check_cases,check_ctas_matrix}.csv` | `utils/check/check_matrix.png` | `Check_Engine_Report.md` §5 |
| `Utils.Monotonicity` | `tests/utils/test_Monotonicity_deep.py` | 29 | `utils/monotonicity/{semigroup_table,monotonic_truth_table,give_monotony_exhaustive,chunk_invariance}.csv` | `utils/monotonicity/monotonicity_matrix.png` | `Monotonicity_Engine_Report.md` §6 |
| `Utils.Poly` | `tests/utils/test_Poly_deep.py` | 26 | `utils/poly/poly_vs_polyfit.csv` | `utils/poly/poly_vs_polyfit.png` | `Poly_Engine_Report.md` §5 |
| `Utils.Peaks` | `tests/utils/test_Peaks_deep.py` | 39 | `utils/peaks/{backend_agreement,peak_counts,backend_timing}.csv` | `utils/peaks/peaks_backends.png` | `Peaks_Engine_Report.md` §6 |
| `Utils.Envelope` | `tests/utils/test_Envelope_deep.py` | 20 | `utils/envelope/envelope_modes.csv` | `utils/envelope/envelope_modes.png` | `Envelope_Engine_Report.md` §6 |
| `Utils.Hilbert` | `tests/utils/test_Hilbert_deep.py` | 27 | `utils/hilbert/{backend_agreement,contract_matrix,backend_timing,zeropad_artefact}.csv` | `utils/hilbert/hilbert_backends.png` | `Hilbert_Engine_Report.md` §6 |
| `Utils.Mirror` | `tests/utils/test_Mirror_deep.py` | 51 | `utils/mirror/mirror_modes.csv` | `utils/mirror/mirror_modes.png` | `Mirror_Engine_Report.md` §6 |
| `Utils.Slepian` | `tests/utils/test_Slepian_deep.py` | 35 | `utils/slepian/{backend_agreement,n_sweep}.csv` | `utils/slepian/slepian_backends.png` | `Slepian_Engine_Report.md` §5 |
| `Base.Chunk` | `tests/base/test_Chunk_deep.py` | 50 | `base/chunk/{chunk_size_table,chunk_cases}.csv` | `base/chunk/chunk_size.png` | `Chunk_Engine_Report.md` §7 |
| `Base.Cache` | `tests/base/test_Cache_deep.py` | 29 | `base/cache/bench.csv` | `base/cache/cache_bench.png` | `Cache_Engine_Report.md` §5 |
| `Base.CPU` | `tests/base/test_CPU_deep.py` | 20 | `base/cpu/allocable_table.csv` | — | `CPU_Engine_Report.md` §7 |
| `Base.Memory` | `tests/base/test_Memory_deep.py` | 20 | `base/memory/budget_table.csv` | — | `Memory_Engine_Report.md` §5 |
| `Base.Seed` | `tests/base/test_Seed_deep.py` | 18 | — | — | `Seed_Engine_Report.md` §5 |

合计 **453 条**（`pytest tests/utils/test_*_deep.py tests/base/test_*_deep.py`），
全部只跑 ≤ 1 MiB 的数组，**单进程 ~12 s**。

发现汇总（每条都能在对应报告的「发现」表里查到现象与期望）：

| 级别 | 例子 |
|---|---|
| **真 bug** | `Monotonicity._give_monotony` 分块时丢边界增量（`chunk_size=1` 退化成 `Equal`）；`Envelope` 的 `hilbert_mod`/`peak_mod` 因 kwargs 泄漏而**不可用**；`Chunk.chunk_size` 单位混用导致**永不分块**；`Check._load` 超长 list 分支必抛 `TypeError` |
| **危险行为** | `Chunk` 的 exo-memory 块在迭代结束后 dereference → **access violation 崩溃**；`Chunk.chunked_fill` 长度不一致时**静默截断**；`cleanup_memmaps` 错误分支 `TypeError: unhashable type: 'list'` |
| **契约不一致** | `Hilbert` 三后端对 2-D/空输入/未知 kwarg/`matlab_phase` 的处理不同；`Peaks` 未知过滤键在 scipy 与 numpy/numba 上语义不同；`Cache` 的 dunder 只能在实例上用；`CPU` 的 `Base.allocable_cpu_num` 是陈旧快照 |
| **文档过期** | `Slepian` 默认后端写 "C"（实际 "scipy"）；`Mirror` 说非浮点 dtype 报错（实际静默提升）；`Hilbert` 说 FHT 只支持 2 的幂（实际自适应） |

### 3.1 本轮按指示修掉的 5 处（每个都有回归用例）

| # | 位置 | 改动 | 回归 |
|---|---|---|---|
| 1 | `LMD.__init__` / `f_LMD` | `hilbert_mod` 默认 `"Scipy"` → **`"scipy"`**（不做大小写归一）；`f_LMD` 不再透传已废弃的 `hilbert_backend` | `tests/methods/test_LMD.py` 的构造类用例 |
| 2 | `Chunk._exo_chunks` | 每块改由 `Memory.temp_memmap` 拥有，迭代结束**不关映射**（原先在 `finally` 里全关 → 之后 dereference 直接 access violation）；临时文件由 `cleanup_memmaps()`/退出回收 | `test_exo_memory_chunks_stay_valid_after_iteration` 等 4 条 |
| 3 | `Monotonicity._give_monotony` | ① 边界增量折进累加器；② 单元素块（`Equal` 是空信息）不再参与合并 | `test_give_monotony_chunked_matches_whole` + 旧 `test_chunked_give_monotony_matches_whole` |
| 4 | `Envelope.envelope` | `peak_mod` / `hilbert_mod` 由 `kwargs.get` 改成 **`kwargs.pop`**，不再透传给 `hilbert()` / `CubicSpline()` | `test_hilbert_backend_switch_works`、`test_peak_mod_kwarg_no_longer_leaks` |
| 5 | `Chunk.chunk_size` | 单位修正：入参字节、返回**元素数**（小输入 8 Mi 元素；大输入 = `min(nbytes, 0.75×budget)` 字节 ÷ `itemsize`），新增 `itemsize=` 并由 `chunk()`/`fill_memmap()` 传真实值 | `test_chunk_size_units_are_bytes_in_elements_out`、`_above_budget_is_capped`、`_itemsize_validation` |
| 附 | `Memory.cleanup_memmaps` | 错误分支 `_ERROR_FILES.add(list)` → `update()`（否则"清理失败"变成 `TypeError`）；并把 `FileNotFoundError` 视作"已删除"（POSIX 上外部删文件是常态） | `test_cleanup_memmaps_with_live_reference_is_platform_aware`、`test_cleanup_memmaps_tolerates_externally_removed_file` |
| 附 | `LMD` 在途重构 | 恢复 `d01fa2a` 里被删的 4 个模块级辅助函数（`_safe_interpolate`、`_safe_interpolate_pchip`、`_check_convergence`、`mirror_extrema`，150 行） | `tests/methods/test_LMD.py` + `tests/test_memmap_output.py`（148 passed） |

> 这 5 处修完后 `Chunk` 在 1–600 MiB **仍是 1 块** —— 这是**设计意图**（块长按预算取，
> 这几个档位都装得下），不是缺陷；单位修正只影响"超出 0.75×budget"的输入（原本会
> 拿到 8 倍过大的块）。

## 4. 规模轴：1 / 50 / 100 / 200 / 500 / 600 MiB

**分解法不做这条轴**（单模态 500 MB 就要 5.75–9.76 GB，见 §2 最后一行报告），
只覆盖"读一遍数据"量级的 8 个工具、共 198 格，**全部成功**。

| 内容 | 位置 |
|---|---|
| 结论与逐工具表格 | `docs/test/Scale_Axis_Report.md` |
| 驱动脚本 | `tests/size_axis/run_scale.py`（`--tool <name> --export`、`--rebuild-summary`） |
| 报告渲染 | `tests/size_axis/report.py`（从 `summary.csv` 生成 md） |
| 数据 | `table/size_axis/summary.csv` + `<tool>_<op>.csv` |
| 图 | `fig/size_axis/<tool>.png` |

覆盖：`Monotonicity`（5 op）、`Chunk`（6 op）、`Check`（6 op）、`Peaks`（4 op）、
`Mirror`（3 op）、`Poly`（2 op）、`Hilbert`（3 后端）、`Envelope`（4 模式）。

## 5. 未成报告 / 推迟的项

| 项 | 状态 |
|---|---|
| `comparison/verify_envelope_modes.py`、`comparison/verify_lmd_recon.py` | 控制台验证脚本（无独立结果文件）；结论并入对应方法报告 |
| 3 GB 与 ≥1 GB 随机噪声内存格 | 推迟记录：`comparison/POSTPONED_3GB.md`；现行内存轴见 `docs/test/methods/Large_Array_Memory_Report.md` |
| 历史（PyEMD 包装时代）计时/内存结论 | 数据归档于 `tests/comparison/results/_legacy_pyemd_wrapper/`、`table/methods/_legacy_pyemd_wrapper/` |
| `test_memory` 的 `_runs/*.report.json` | 作为内存轴的底层证据保留，不单独出报告 |
| `tests/import_time.py` | 三种惰性导入写法的耗时对比（A: try/except、B: flag+try/except、C: Cache.import_module），结论见 `Base/Cache_Engine_Report.md` §5.2 |

## 5.1 CI 修复轮（跨平台，全部已修）

| 现象 | 平台 | 根因 | 修法 |
|---|---|---|---|
| `LMD.decompose` 抛 `NameError: mirror_extrema`（+3 个同批被删的辅助函数） | 全部 | 在途重构删掉了模块级 `_safe_interpolate` / `_safe_interpolate_pchip` / `_check_convergence` / `mirror_extrema`，调用点仍在 | 从 `d01fa2a` 逐字恢复这 4 个函数（150 行），签名与调用点一致 |
| `test_base_reexport_is_a_stale_snapshot` → `num of value must in (0, 3)]` | CI runner 只有 3 核 | 测试硬编码 `set_allocable(4)` | 改成相对 `cpu_num` 取值；单核时 `skip` |
| `test_exo_memory_chunks_stay_valid_after_iteration` / `test_cleanup_memmaps_with_live_reference_is_platform_aware` → `DID NOT WARN` | Linux / macOS | POSIX **允许删除已映射的文件**，`cleanup_memmaps()` 返回 `True` 且不告警；Windows 才是 `False` + 告警 | 测试按 `os.name` 分支断言；两种平台都要求"块仍可读" |
| `test_fill_memmap_rejects_list` / `tests/base/test_Memory.py` → `FileNotFoundError` | Linux / macOS | 注册表里的文件被前一个用例手删后，`cleanup_memmaps()` 的 `os.remove` 只捕 `PermissionError` | `Memory.cleanup_memmaps` 把 `FileNotFoundError` 视作"已删除"（重试循环里同样处理），并新增回归用例 |
| `test_numba_compiled_once` | 全部（CI 不装 numba） | `_get_compiled` 从未被调用，`cache_info().currsize == 0` | 用例内先探测 numba，缺失则 `skip` |
| `test_linear_time_in_x_new` → `ratio 47.8 < 50` | macOS / Linux | 计时比值区间过窄（10k 档被固定开销主导） | 放宽到 `15 < ratio < 400`（O(n²) 会是 ~10000×） |

`pytest tests -q` → **1510 passed, 2 skipped, 1 xfailed**（Windows/py3.10 本地实测；
CI 矩阵为 ubuntu/windows/macos × py3.10/3.12）。

## 6. tests/ 结构（本轮后）

```
tests/
  conftest.py                 # sys.path 指向 src/、种子复位
  _cases.py                   # 15 个方法的快速参数集
  _bench.py                   # 计时/峰值分配/csv/绘图 公共设施（测试侧，不进包）
  tools/check_links.py        # md 链接与资源路径自检
  methods/test_<X>.py         # 15 个方法的统一契约测试
  utils/test_<Tool>.py        # 工具层契约烟测（原有）
  utils/test_<Tool>_deep.py   # 工具层深度测试（本轮，8 个）
  base/test_<Tool>.py         # Base 层契约烟测（原有）
  base/test_<Tool>_deep.py    # Base 层深度测试（本轮，5 个）
  size_axis/run_scale.py      # 1–600 MiB 规模轴驱动
  size_axis/report.py         # 规模轴报告渲染
  comparison/                 # 实验脚本与原始 JSON（不含 pytest 用例）
  test_memory/                # 内存轴驱动
```

## 7. 本轮目录重构记录

* **迁移脚本**（一次性，可重跑校验）：`.scratch/reorg/migrate.py` → `fixlinks.py`
  → `patch_scripts.py` → `patch_scratch_figs.py`；共 `git mv` **75** 个文件。
* `docs/test/` 由**平铺 39 个文件**改为 `methods/`（16）、`utils/`（11）、`base/`（6）、
  `_template/`（3）四个子目录；所有 md 内联链接由脚本重算相对路径，当前失效链接 **0**。
* `fig/` 由平铺改为 `methods/<m>/`、`utils/<t>/`、`base/<t>/`、`comparison/`、`size_axis/`、
  `explore/denoise/`；原有 `fig/EMD/`、`fig/tests/` 两个特例目录已并入。
* `docs/explore/denoise/figures/*.png`（**19 张**）也迁进 `fig/explore/denoise/`，
  8 份 explore 报告里的 19 个图片链接与 11 处"产物路径"全部改写；
  `explore/denoise/make_*.py`（9 个生成脚本）的输出目录同步改为 `fig/explore/denoise/`。
  现在**仓库内没有任何 png 落在 `fig/` 之外**（`tests/tools/verify_reorg.py` 会检查）。
* 新增 `table/`：逐 run 计时 csv 从 `tests/comparison/results/*/timing_raw.csv`
  迁入 `table/methods/*/`；`bench_timing.py`/`bench_memory.py`/`run_matrix.py`/
  `summarize_*.py`/`bench_plot.py` 的读写路径同步更新（JSON 仍留 `results/`）。
* 新增 `table/index.xlsx`：把 `table/` 下所有 csv 汇总成一个工作簿（每 csv 一个 sheet）。
* 历史删除记录保留在 §8；被删除的旧报告不再作为依据。

## 8. 已删除/归档的旧材料（历史记录，不再作为依据）

* 删除 `tests/test_path.py`（引用不存在的 `PathDefine.ROOT`）、`tests/test_optimize_decomposition/`
  （早期 EMD 原型草稿）、`tests/test_fht.py`（被 `tests/utils/test_Hilbert.py` + 深度测试取代）；
* 删除 EMD 对照实验脚本与 `docs/EMD_*.md` 全部旧报告，由 `.scratch/emd_new/` 与
  `docs/test/methods/EMD_Native_Engine_Report.md` 取代；
* 归档 `tests/comparison/results/_legacy_pyemd_wrapper/`（PyEMD 包装时代的计时/内存）；
* 删除旧工具文档（接口随重构消失）：`CacheConventions.md`、`CacheSpeedReport.md`、
  `ChunkReport.md`、`PeaksReport.md`、`Slepian_Cache_Report.md`、`ComponentStatisticsReport.md`、
  `Decomposition_Methods_Timing_Report.md`；
* 删除 `tests/` 下测试已移除 API 的旧用例：`test_registry`、`test_ewtpy_adapter`、
  `test_seed`、`test_spline`、`test_utils`、`test_slepian`、`test_contract`、
  `test_reconstruction`、`test_facade`、`test_fmd_lmd_refactor`、`test_peaks`、
  `test_mirror_new`、`test_envelope`、`test_monotonicity`、`test_montonic`、`test_cache`、`ssa/`。
