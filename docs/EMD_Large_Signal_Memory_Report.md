# EMD 大数据内存报告（1 MB – 1 GB，**原生引擎重跑版**）

> **2026-09-10 重跑**: 本报告内存实验已在**当前原生引擎**上重做
> （此前版本沿用 PyEMD 包装时代归档数据）。旧数据保留于
> `tests/comparison/results/_legacy_pyemd_wrapper/` 作为对照。
> 复现: `python tests/comparison/bench_memory.py` 与
> `python tests/test_memory/run_matrix.py --methods EMD --sizes 1MB 20MB 100MB 500MB 1024MB
> --patterns increasing random --dtypes float16 float32 float64`。
> 数据: `tests/comparison/results/memory/*.json` 与
> `tests/test_memory/result_for_each_decomposition/EMD.{json,csv}`。
> 方法: 每格独立子进程；输入完全建成后才计分解预算（默认 20 s/格）；RSS 以
> 50 ms 心跳采样；输入 ≥ 500 MB 以 `memmap` 建底。

## 0. 结论速览（相对包装时代的变化）

1. **原生引擎把"可完成长度上限"整体上移**: 库内矩阵中 float16 的 500 MB、
   float32 的 500 MB / 1024 MB increasing 在旧引擎全部 timeout，现在分别
   2.71 s / 0.57 s / 2.11 s 完成。
2. **峰值内存约减半**: 1 GB increasing 峰值 RSS 从旧引擎的 8119 MB 降到
   **4257 MB**（输入 4.2×）；同格 PyEMD 8282 MB、PySDKit 8216 MB。
3. **墙钟快 6–12×**: increasing 轴 100 MB 0.08 s（旧 0.73 s）、500 MB 0.44 s
   （旧 4.31 s）、1024 MB 0.91 s（旧 10.80 s）；1 MB random 4.16 s（旧 10.66 s，
   PyEMD 10.17 s、PySDKit 11.50 s）。
4. **仍属算法代价的边界**: 白噪声 ≥ 20 MB 与 float16 的 1024 MB 在 20 s 预算
   内仍 timeout（三方一致）；1 GB random 与 3 GB 保持 deferred（16 GB 机器，
   见 `POSTPONED_3GB.md`）。

---

## 1. 三方对照轴（当前引擎, 1 MB – 1 GB）

### 1.1 increasing（确定性递增信号）

| 输入 | 实现 | 状态 | 墙钟 s | ΔRSS MB | 峰值 RSS MB | 峰值/输入 | 旧引擎峰值 MB |
|---:|---|---|---:|---:|---:|---:|---:|
| 1 MB | MD-EMD | ok | 0.00 | ~0 | 33.6 | 33.6× | 113.0 |
| 1 MB | PyEMD | ok | 0.00 | 0.1 | 113.4 | 113× | 113.3 |
| 1 MB | PySDKit | ok | 0.02 | 0.1 | 119.0 | 119× | 119.0 |
| 20 MB | MD-EMD | ok | 0.02 | ~18 | 52.6 | 2.6× | 273.8 |
| 20 MB | PyEMD | ok | 0.11 | 102.8 | 232.6 | 11.6× | 234.7 |
| 20 MB | PySDKit | ok | 0.12 | 113.4 | 258.6 | 12.9× | 250.7 |
| 100 MB | MD-EMD | ok | 0.08 | ~195 | 327.4 | 3.3× | 992.2 |
| 100 MB | PyEMD | ok | 0.56 | 604.9 | 892.4 | 8.9× | 816.9 |
| 100 MB | PySDKit | ok | 0.59 | 600.3 | 800.3 | 8.0× | 818.2 |
| 500 MB | MD-EMD | ok | 0.44 | ~1600 | 2086.9 | 4.2× | 4576.0 |
| 500 MB | PyEMD | ok | 3.02 | 3488.5 | 4093.4 | 8.2× | 4100.3 |
| 500 MB | PySDKit | ok | 3.34 | 3440.4 | 4015.6 | 8.0× | 4058.2 |
| 1024 MB | MD-EMD | ok | **0.91** | ~3000 | **4256.8** | **4.2×** | 8119.4 |
| 1024 MB | PyEMD | ok | 7.50 | 7153.8 | 8281.9 | 8.1× | 8289.6 |
| 1024 MB | PySDKit | ok | 7.27 | 7018.8 | 8216.2 | 8.0× | 8160.5 |

（ΔRSS 为"峰值 − 基线"；旧引擎峰值列为 2026-09-06 归档值。）

要点:
* 原生引擎的峰值 RSS 稳定在 **~3.3–4.2× 输入**（memmap 大格），外部实现仍为
  **~8×**：1 GB 格上库比 PyEMD 少 **~4.0 GB**。
* 墙钟上库实现快 8×（1024 MB 格）——与计时轴结论一致（`default_T` 优化 +
  原生筛分路径无包装拷贝）。

### 1.2 random（白噪声）

| 输入 | MD-EMD | PyEMD | PySDKit |
|---:|---|---|---|
| 1 MB | **ok, 4.16 s**, 96.1 MB, 17 行 | ok, 10.17 s, 139.1 MB, 17 行 | ok, 11.50 s, 164.2 MB, 17 行 |
| 20 MB | timeout（峰值 415 MB） | timeout（605 MB） | timeout（608 MB） |
| 100 MB | timeout（峰值 1779 MB） | timeout（2563 MB） | timeout（2552 MB） |
| 500 MB | timeout（峰值 7522 MB） | timeout（8439 MB） | timeout（8730 MB） |
| 1024 MB | deferred | deferred | deferred |

要点:
* 1 MB random 库快 PyEMD **2.4×**、快 PySDKit 2.8×（原生筛分对噪声的加速）；
* timeout 格的峰值：库在相同预算内到达的 RSS 更低（500 MB 格少 0.9–1.2 GB）
  ——同一预算下库"走得比外部实现远"的另一种体现。
* 1 GB random 保持 deferred（16 GB 机器；见 `POSTPONED_3GB.md`）。

---

## 2. 库内矩阵（dtype × 长度 × pattern，当前引擎）

源: `tests/test_memory/result_for_each_decomposition/EMD.{json,csv}`
（30 格 × 3 个 safe 记录 = 90 条）。预算 20 s。

| dtype | 长度 | pattern | 状态 | 墙钟 s | 峰值增量 MB | 输入形态 | n_imfs |
|---|---:|---|---|---:|---:|---|---:|
| float16 | 1 MB | increasing | ok | 0.00 | 7.3 | ndarray | 0 |
| float16 | 20 MB | increasing | ok | 0.08 | 327.6 | ndarray | 0 |
| float16 | 100 MB | increasing | ok | 0.36 | 1643.9 | ndarray | 0 |
| float16 | 500 MB | increasing | **ok（旧 timeout）** | 2.71 | 8249.4 | memmap | 0 |
| float16 | 1024 MB | increasing | timeout | — | — | memmap | — |
| float32 | 1 MB | increasing | ok | 0.00 | 1.3 | ndarray | 0 |
| float32 | 20 MB | increasing | ok | 0.02 | 69.5 | ndarray | 0 |
| float32 | 100 MB | increasing | ok | 0.11 | 500.3 | ndarray | 0 |
| float32 | 500 MB | increasing | **ok（旧 timeout）** | 0.57 | 2624.5 | memmap | 0 |
| float32 | 1024 MB | increasing | **ok（旧 timeout）** | 2.11 | 5376.1 | memmap | 0 |
| float64 | 1 MB | increasing | ok | 0.00 | 1.0 | ndarray | 0 |
| float64 | 20 MB | increasing | ok | 0.01 | 29.0 | ndarray | 0 |
| float64 | 100 MB | increasing | ok | 0.07 | 301.1 | ndarray | 0 |
| float64 | 500 MB | increasing | ok | 0.38 | 1558.6 | memmap | 0 |
| float64 | 1024 MB | increasing | ok | 0.97 | 3200.1 | memmap | 0 |
| float16 | 1 MB | random | ok | 13.51 | 106.4 | ndarray | 3 |
| float32 | 1 MB | random | ok | 5.47 | 65.6 | ndarray | 3 |
| float64 | 1 MB | random | ok | 2.33 | 54.8 | ndarray | 3 |
| 全部 dtype | 20 MB–1024 MB | random | timeout | — | — | — | — |

要点:
1. **精度提升仍是内存放大器，但幅度下降**: 100 MB 输入峰值增量 float16
   1644 MB（16.4×）> float32 500 MB（5.0×）> float64 301 MB（3.0×）——
   float16→float64 工作精度仍是主因；但 500 MB/1024 MB 的 float16/float32
   从旧引擎的全 timeout 变为可完成（原生路径没有包装层的额外副本）。
2. **random 1 MB 三种精度都完成**且比旧引擎快 ~3×（float64 6.66 s → 2.33 s）；
   噪声 ≥ 20 MB 在 20 s 预算内三方同样无法完成（算法代价）。
3. **increasing 格仍是"平凡完成"**（单调信号 n_imfs=0），测的是输入构建 +
   扫描/分块的内存与时间；分解压力由 random 格承担。
4. 残差单调性扫描（scan）三个 safe 分级在全部 ok 格完成，结果与此前口径一致
   （详见 `EMD.json` 的 scan 字段）。

---

## 3. 与包装时代的对照

| 量 | 旧引擎（2026-09-06, 归档） | 当前引擎（2026-09-10） |
|---|---|---|
| 1024 MB increasing 墙钟 | 10.797 s | **0.91 s（11.9×）** |
| 1024 MB increasing 峰值 RSS | 8119 MB（优化后 6950 MB） | **4257 MB（≈半）** |
| 1 MB random 墙钟 | 10.66 s | **4.16 s（2.6×）** |
| float16/float32 500 MB+ increasing | timeout | **完成（2.7 s / 0.6 s / 2.1 s）** |
| 500 MB random 超时前峰值 | 7789 MB | 7522 MB（略低） |

对照来源: `results/_legacy_pyemd_wrapper/{memory/, memory_after_opt/}`。

---

## 4. 边界与推迟

* **1 GB random 与 3 GB 全部 deferred**（16 GB 机器）：原生引擎峰值 ~4.2×
  输入，1 GB random 的峰值仍预计 ≥ 10 GB（噪声行逐条堆积），3 GB 需要
  ≥ 27 GB 机器；恢复步骤见 `tests/comparison/POSTPONED_3GB.md`（`--no-defer`）。
* 本报告 500 MB random 为**重跑实测**（此前 deferred 记录已更新）；timeout
  格的心跳轨迹在 `results/memory/_runs/*.hb.log`。

---

## 5. 外存映射（memmap）临时文件的清理语义

用户关注点: 采用映射到外存的策略后，外存文件是否会被清理、是否会一直占用。

库内两处外存策略的清理语义:

| 策略 | 位置 | 临时文件位置 | 清理时机 |
|---|---|---|---|
| 输入层临时 memmap（大 ndarray/list 的流式填充、f16→f64 转换） | `Utils/Check.py::_temp_memmap` | 系统临时目录（`md_memmap_*.dat`） | **进程退出时**：登记表 `_MEM_FILES` 由 `atexit` 统一先关 mmap 句柄再删除（Windows 要求先 close 才能 unlink）；进程被强杀时由 OS 临时清理器兜底 |
| 外存分块迭代器（`exo_chunks`） | `Utils/Chunk.py::exo_chunks` | 系统临时目录（`md_chunk_*.dat`） | **每次迭代结束**（含异常路径，`finally` 中关闭句柄 + `os.remove`） |

行为验证（测试已入库）: `tests/test_utils.py::test_temp_memmap_backing_file_registered_and_cleaned`
（创建 → 登记 → 清理后文件不存在、幂等）；端到端冒烟（触发真实 memmap 输入 →
进程退出后文件消失）。

注意:
* **不会一直占用**: 正常退出即删；长驻进程（Jupyter/服务）中，临时文件存活
  到对应输入被分解完毕后的下一次进程退出，可用文件名前缀 `md_memmap_` 手工
  识别清理（若进程被强杀）。
* 若担心长驻进程累积，可在业务侧调用完分解后执行
  `Modal_Decomposition.Utils.Check._cleanup_memmaps()`（幂等，删除全部本库
  输入层临时文件；已作为调试接口保留）。
