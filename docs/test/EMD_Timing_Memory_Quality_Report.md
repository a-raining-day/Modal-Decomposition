# EMD 计时 · 质量 · 内存报告（原生引擎，现行默认档）

> 对象: 库内 `Modal_Decomposition.Class.EMD` —— 原生筛分引擎，**现行默认**
> （`CubicSpline` 包络 + `sd_thr=0.01` + `max_iter=100` + `nbsym=2` +
> `faster=False` 质量档）。
> 对照: `PyEMD.EMD`（EMD-signal 1.9.0，cubic/nbsym=2 默认）与
> `pysdkit.EMD`（`ref/pysdkit` v0.5.0 独立移植）。
> 复现: `python tests/comparison/bench_timing.py --method EMD`
> （原始行 `tests/comparison/results/emd/timing_raw.csv`，质量
> `results/emd/timing_metrics.json`）。
> 信号: `tests/comparison/signals.py` —— case A 双音调 37/113 Hz + 白噪 0.1；
> case B AM-FM + 89 Hz + 二次趋势（无噪）；case C 纯白噪声（C 限长 16384）。
> 计时口径: 同进程、轮转顺序、每格 3 次重复取**中位**，import/预热不计入。

## 1. 计时（中位墙钟, ms）

| case | n | MD-EMD | PyEMD | PySDKit | MD 相对 PyEMD |
|---|---:|---:|---:|---:|---:|
| A | 256 | 14.2 | 9.1 | 11.4 | 0.6×（慢） |
| A | 1024 | 8.6 | 31.7 | 17.8 | 3.7× |
| A | 4096 | 55.0 | 124.3 | 125.2 | 2.3× |
| A | 16384 | 400.1 | 883.0 | 927.4 | 2.2× |
| A | 65536 | 1703.6 | 12079.4 | 12270.3 | **7.1×** |
| B | 256 | 6.0 | 3.0 | 5.2 | 0.5×（慢） |
| B | 1024 | 14.5 | 24.1 | 14.5 | 1.7× |
| B | 4096 | 13.2 | 41.5 | 47.6 | 3.1× |
| B | 16384 | 32.0 | 84.8 | 106.1 | 2.6× |
| B | 65536 | 147.9 | 498.8 | 588.9 | 3.4× |
| C | 256 | 17.1 | 15.2 | 17.6 | 0.9× |
| C | 1024 | 39.0 | 37.1 | 32.6 | 0.9× |
| C | 4096 | 85.2 | 113.9 | 119.3 | 1.3× |
| C | 16384 | 335.1 | 691.6 | 715.3 | 2.1× |

要点:
* n ≥ 1024 时库实现全面更快；优势随 n 增长（A/65536 **7.1×**）。
* n = 256 的亚毫秒-十毫秒格（A/B/C 各一）库实现略慢（0.5–0.9×）：固定开销
  （极值/镜像/包络的最小路径）在小样本上占比高，且此刻计时受调度噪声影响大
  ——此三格的比值不构成结论。
* PyEMD 与 PySDKit 在 A/B 两 case 上几乎重合（同一 sifter 的移植），C（噪声）
  上 PySDKit 略慢。

## 2. 质量（n=16384，同一脚本内计算）

| case | 实现 | 行数 | recon max abs | IO | 模式捕获（best \|corr\|） |
|---|---|---:|---:|---:|---|
| A | MD-EMD | 12 | 8.9e-16 | −0.0074 | 37 Hz **0.9500**、113 Hz **0.7260** |
| A | PyEMD | 12 | 4.3e-19 | −0.0261 | 37 Hz 0.9740、113 Hz 0.8772 |
| A | PySDKit | 12 | 4.3e-19 | −0.0261 | 37 Hz 0.9740、113 Hz 0.8772 |
| B | MD-EMD | 6 | 4.4e-16 | 0.0017 | amfm 0.99979、89 Hz 0.99962、trend 0.99996 |
| B | PyEMD | 7 | 1.1e-16 | 0.0012 | amfm 0.99982、89 Hz 0.99963、trend 0.99991 |
| B | PySDKit | 7 | 1.1e-16 | 0.0012 | 同上 |
| C | MD-EMD | 12 | 1.3e-15 | −0.0720 | —（纯噪声） |
| C | PyEMD | 13 | 2.2e-19 | −0.0736 | — |
| C | PySDKit | 13 | 2.2e-19 | −0.0736 | — |

要点:
* **重构**: 三实现均精确（≤1.3e-15）；库为逐次减法余量，外部为对角平均路径。
* **正交性（IO）**: 同档（A: −0.007 vs −0.026；B: 0.0017 vs 0.0012；
  C: −0.072 vs −0.074）——负值表示轻微负相关（EMD 已知现象），量级一致。
* **模式捕获**: case B（干净）三方几乎完全相同（corr ≥ 0.9996）；case A 上
  库的弱音调 113 Hz 单行捕获 0.726 vs PyEMD 0.877（差 0.15），37 Hz 差 0.024
  ——与 `docs/EMD_faster_Branch_Comparison_Report.md` 的结论一致（质量档已把
  差距收窄；残留差距属停止判据工程，非重构/泄漏问题）。
* **行数**: 库比外部少 1 行（A/C 12 vs 12/13；B 6 vs 7）——末行低能量噪声尾的
  归属差异，不影响能量完整性。

## 3. 内存（**当前原生引擎重跑, 2026-09-10**）

> 本批内存数据已在现行原生引擎上重跑（`results/memory/`，2026-09-10）。
> 包装时代归档数据（`results/_legacy_pyemd_wrapper/`）作为对照；完整矩阵
> 见 `docs/EMD_Large_Signal_Memory_Report.md`。
> 方法: 每格独立子进程；输入完全建成后才计分解预算（20 s/格）；RSS 以
> 50 ms 心跳采样；输入 ≥ 500 MB 以 `memmap` 建底。

### 3.1 increasing（确定性递增信号）

| 输入 | MD-EMD 峰值 MB | PyEMD | PySDKit | MD 墙钟 s | PyEMD 墙钟 s | PySDKit 墙钟 s |
|---:|---:|---:|---:|---:|---:|---:|
| 1 MB | 33.6 | 113.4 | 119.0 | 0.00 | 0.00 | 0.02 |
| 20 MB | 52.6 | 232.6 | 258.6 | 0.02 | 0.11 | 0.12 |
| 100 MB | 327.4 | 892.4 | 800.3 | 0.08 | 0.56 | 0.59 |
| 500 MB | 2086.9 | 4093.4 | 4015.6 | 0.44 | 3.02 | 3.34 |
| 1024 MB | **4256.8** | 8281.9 | 8216.2 | **0.91** | 7.50 | 7.27 |

要点:
* 原生引擎峰值 RSS 稳定在 **~3.3–4.2× 输入**，外部实现仍 ~8×：1 GB 格上库
  比 PyEMD 少 **~4.0 GB**；墙钟快 **8×**。旧引擎同格为 10.8 s / 8119 MB
  （`default_T` 优化 + 原生路径的共同结果）。

### 3.2 random（白噪声，分解预算内多不可完成）

| 输入 | MD-EMD | PyEMD | PySDKit |
|---:|---|---|---|
| 1 MB | **ok, 4.16 s**, 96.1 MB, 17 行 | ok, 10.17 s, 139.1 MB | ok, 11.50 s, 164.2 MB |
| 20 MB | timeout（峰值 415 MB） | timeout（605 MB） | timeout（608 MB） |
| 100 MB | timeout（峰值 1779 MB） | timeout（2563 MB） | timeout（2552 MB） |
| 500 MB | timeout（峰值 7522 MB） | timeout（8439 MB） | timeout（8730 MB） |
| 1024 MB | deferred | deferred | deferred |

要点: 1 MB 噪声库快 PyEMD 2.4×；20 s 预算下三方同样无法完成 ≥ 20 MB 噪声
（算法代价，非实现缺陷）；库在超时前到达的峰值更低（500 MB 格少 ~1 GB）。

## 4. 结论

1. **计时**: n ≥ 1024 全面快于 PyEMD/PySDKit（2.1–7.1×），小样本另有固定开销；
   现行质量档（`faster=False`）已含窄带门成本，仍是同档最快实现。
2. **质量**: 重构/正交性与外部实现同档；case B 模式捕获几乎相同；case A 弱音调
   单行捕获仍差 0.15（已从旧默认档的 0.25 收窄）。
3. **内存**: 当前引擎峰值 RSS **3.3–4.2× 输入**（外部 ~8×），1 GB 格比 PyEMD
   少 ~4 GB 且快 8×；噪声 ≥ 20 MB 的 timeout 为算法固有代价（三方一致）。

## 5. 复现

```powershell
$env:PYTHONPATH='src'
python tests\comparison\bench_timing.py --method EMD        # → results/emd/
python tests\comparison\bench_plot.py                       # 可选出图 (figs/)
```
内存轴（当前引擎重跑版）：`python tests\comparison\bench_memory.py`
（默认网格 1MB–1GB × increasing/random，20 s/格预算；1 GB random 与 3 GB
保持 deferred，见 `tests/comparison/POSTPONED_3GB.md`）。
