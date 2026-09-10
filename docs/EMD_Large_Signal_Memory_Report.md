# EMD 大数据内存报告（1 MB – 3 GB 轴，归档数据）

> ⚠️ **数据边界（重要）**: 本报告的两套内存实验均**沿用归档数据、未重跑**
> （按用户指示）。采集时间为 2026-09-05/06，当时库 EMD 仍是 **PyEMD 包装版**
> （原生引擎于 2026-09-09 转正）。因此：
> * 结论中"库 EMD 的峰值 RSS / 长度上限"反映**包装时代**的行为；
> * 现行原生引擎的内存特性尚未重测，但方向明确：原生实现去掉了包装层的
>   结果缓存与额外拷贝，且 `Check_Time_and_Signal(default_T=...)` 优化已消除
>   引擎不用的时间轴分配（§3 的"优化后"列即为该优化的实测收益）；
> * 数据文件完整保留，随时可在现行引擎上按本文命令重跑替换。
>
> 数据源:
> * `tests/comparison/results/_legacy_pyemd_wrapper/{summary_memory.md,
>   memory_flat.csv, memory/, memory_after_opt/}` —— 三方对照轴；
> * `tests/test_memory/result_for_each_decomposition/EMD_legacy_pyemd_wrapper.{json,csv}`
>   —— 库内 EMD 的 dtype × 长度 × pattern × safe 分级矩阵。

---

## 1. 三方对照轴（1 MB – 1 GB）

方法: 每格独立子进程；**输入完全建成后**才开始计分解预算（默认 20 s/格）；
RSS 以 50 ms 心跳采样；被 kill 的超时格也保留增长轨迹
（`memory/_runs/*.hb.log`）。输入 ≥ 500 MB 时以 `memmap` 建底。

### 1.1 increasing（确定性递增信号）

| 输入 | 实现 | 状态 | 墙钟 s | ΔRSS MB | 峰值 RSS MB | 峰值/输入 |
|---:|---|---|---:|---:|---:|---:|
| 1 MB | MD-EMD | ok | 0.00 | 0.1 | 113.0 | 113× |
| 1 MB | PyEMD | ok | 0.016 | 0.1 | 113.3 | 113× |
| 1 MB | PySDKit | ok | 0.00 | 0.1 | 119.0 | 119× |
| 20 MB | MD-EMD | ok | 0.125 | 141.6 | 273.8 | 13.7× |
| 20 MB | PyEMD | ok | 0.125 | 102.8 | 234.7 | 11.7× |
| 20 MB | PySDKit | ok | 0.109 | 113.4 | 250.7 | 12.5× |
| 100 MB | MD-EMD | ok | 0.734 | 780.1 | 992.2 | 9.9× |
| 100 MB | PyEMD | ok | 0.578 | 604.9 | 816.9 | 8.2× |
| 100 MB | PySDKit | ok | 0.593 | 600.3 | 818.2 | 8.2× |
| 500 MB | MD-EMD | ok | 4.313 | 3963.4 | 4576.0 | 9.2× |
| 500 MB | PyEMD | ok | 3.391 | 3488.5 | 4100.3 | 8.2× |
| 500 MB | PySDKit | ok | 3.844 | 3440.4 | 4058.2 | 8.1× |
| 1024 MB | MD-EMD | ok | 10.797 | 6983.1 | 8119.4 | 7.9× |
| 1024 MB | PyEMD | ok | 9.016 | 7153.8 | 8289.6 | 8.1× |
| 1024 MB | PySDKit | ok | 9.812 | 7018.8 | 8160.5 | 8.0× |

要点: 三方峰值 RSS 同量级（**8–9× 输入**；1 MB 格的倍数无意义——基线进程
RSS ~113 MB 占主导）；墙钟同档（库实现小格偏慢、大格持平，与计时轴一致）。

### 1.2 random（白噪声）

| 输入 | MD-EMD | PyEMD | PySDKit | 备注 |
|---:|---|---|---|---|
| 1 MB | ok, 10.66 s, 峰值 141.6 MB, 17 行 | ok, 10.75 s, 138.9 MB | ok, 11.42 s, 158.9 MB | 重构 8.7e-19 |
| 20 MB | timeout | timeout | timeout | 峰值 608–628 MB |
| 100 MB | timeout | timeout | timeout | 峰值 2567–2670 MB |
| 500 MB | timeout | timeout | timeout | 峰值 7789–8576 MB |
| ≥512 MB | deferred | deferred | deferred | 16 GB 机器推迟，见 `POSTPONED_3GB.md` |

要点: 白噪声的 sift 迭代远多于确定性信号，三方在 20 s 预算下**同样**无法完成
——是算法代价而非实现缺陷。库实现对噪声的实测速度优势（计时轴 case C 为
2.1×）意味着在同等预算下库可处理的噪声长度更长，但 1 GB 级噪声在当前预算
下对任何实现都不可行。

---

## 2. 库内 EMD 矩阵（dtype × 长度 × pattern × safe 分级）

源: `EMD_legacy_pyemd_wrapper.json`（108 条 = 36 格 × 3 个 safe 记录）。
长度轴 1 MB/20 MB/100 MB/500 MB/1024 MB/3072 MB；pattern: increasing / random；
dtype: float16 / float32 / float64；预算 20 s。

### 2.1 状态与峰值（每格取首条记录）

| dtype | 长度 | pattern | 状态 | n_imfs | 墙钟 s | 峰值 RSS MB | 峰值/输入 |
|---|---:|---|---|---:|---:|---:|---:|
| float16 | 1 MB | increasing | ok | 0 | 0.03 | 150 | 149.6× |
| float16 | 20 MB | increasing | ok | 0 | 0.65 | 797 | 39.8× |
| float16 | 100 MB | increasing | ok | 0 | 5.41 | 3953 | 39.5× |
| float16 | 500 MB+ | increasing | timeout | — | — | — | — |
| float32 | 1 MB | increasing | ok | 0 | 0.02 | 125 | 125.1× |
| float32 | 20 MB | increasing | ok | 0 | 0.28 | 454 | 22.7× |
| float32 | 100 MB | increasing | ok | 0 | 1.50 | 1962 | 19.6× |
| float32 | 500 MB+ | increasing | timeout | — | — | — | — |
| float64 | 1 MB | increasing | ok | 0 | 0.01 | 120 | 120.5× |
| float64 | 20 MB | increasing | ok | 0 | 0.18 | 277 | 13.8× |
| float64 | 100 MB | increasing | ok | 0 | 0.78 | 970 | 9.7× |
| float64 | 500 MB | increasing | ok | 0 | 4.09 | 4597 | 9.2× |
| float64 | 1024 MB | increasing | ok | 0 | 9.32 | 9317 | 9.1× |
| float64 | 3072 MB | increasing | timeout | — | — | — | — |
| float64 | 1 MB | random | ok | 3 | 6.66 | 139 | 138.7× |
| float16/32/64 | ≥20 MB | random | timeout | — | — | — | — |
| 全部 dtype | 3072 MB | 两种 | timeout | — | — | — | — |

### 2.2 要点

1. **精度提升是内存放大器**: `float16` 输入在库内被提升为 `float64` 工作精度，
   峰值达输入的 **39.5×**（20 MB 与 100 MB 格），`float32` 因内部多份 f64
   临时量也达 **19.6–22.7×**，而 `float64` 仅 **9.1–9.7×**。因此"能处理多大
   信号"的决定因素是输入 dtype：同为 100 MB 输入，float64 0.78 s 完成、
   float16 5.41 s、float32 1.50 s；500 MB 以上只有 float64 完成。
2. **increasing 格是"平凡完成"**: 单调信号没有任何 IMF（`n_imfs=0`），分解在
   极值预检处立即退出——这些格测的是**输入构建 + 扫描/分块**的内存与时间，
   不是分解能力；真正压分解的是 random 格（≥20 MB 即超时）。
3. **3 GB 及随机 ≥512 MB 全部 timeout/deferred**：与 `POSTPONED_3GB.md`
   的推迟记录一致（16 GB 机器的内存上限）。
4. **残差单调性扫描（scan）**: `ok` 格在三个 safe 分级下均完成（33 条 True /
   3 条 False，无缺失）；峰值增量在 safe=0 时最大（100 MB float16 格
   47.8 MB；float32 5.7 MB；float64 8.8 MB），safe=1/2 时 < 1.1 MB —— 即
   分级越高扫描越省内存（具体语义见 `tests/test_memory/run_matrix.py` 参数）。

---

## 3. 内存优化记录（`default_T`）

旧报告中针对 EMD 的优化：`Check_Time_and_Signal` 新增 `default_T` 开关，
EMD 不再分配引擎用不到的时间轴。归档的"优化后"数据
（`memory_after_opt/MD-EMD.csv`）：

| 输入 | 优化前峰值 MB | 优化后峰值 MB | 降幅 | 优化后墙钟 s |
|---:|---:|---:|---:|---:|
| 100 MB | 992.2 | 852 | −140 MB | 0.593 |
| 500 MB | 4576.0 | 4259 | −317 MB | 6.015 |
| 1024 MB | 8119.4 | 6950 | **−1169 MB** | 13.641 |

优化后库 EMD 在 1 GB 上的峰值 RSS 比 PyEMD（8289.6 MB）低约 **1.3 GB**。

---

## 4. 复现

```powershell
# 三方对照轴（默认 1MB..1GB × increasing/random，20s/格预算）
python tests\comparison\bench_memory.py
python tests\comparison\summarize.py            # → summary_memory.md / memory_flat.csv

# 库内 dtype × 长度矩阵
python tests\test_memory\run_matrix.py --methods EMD --sizes 1MB 20MB 100MB --dtypes float64
```

重跑提示: 现行原生引擎的内存行为**尚未重测**（本报告用归档数据）；若要更新
本报告，先删除或改名 `results/memory/*` 等旧产物再执行上面命令，并把新数字
替换本文表格（数据文件路径与字段不变）。
