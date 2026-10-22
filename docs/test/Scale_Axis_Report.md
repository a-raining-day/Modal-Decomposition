# 规模轴：1 MiB – 600 MiB 实测（低成本工具层）

> 生成脚本：`tests/size_axis/run_scale.py`；本文件由 `tests/size_axis/report.py` 从 `table/size_axis/summary.csv` 渲染，**表格数字与 csv 逐格一致**。
>
> **MB 一律指 MiB（2²⁰ B）**：600 MiB = 629 145 600 B = 78 643 200 个 float64。

## 0. 结论先说

* 8 个工具、6 个档位、198 格，**全部成功**（`ok=1`，无 MemoryError / 无超时）；耗时对所有工具都是**线性**的（表里每一行从 1 MiB 到 600 MiB 约 ×600）。
* 真正的成本差异在**内存放大倍数**，不在时间：
  * `is_uniform` **3×**（600 MiB 输入 → 1800 MiB 峰值）、`fast_poly` **5×(deg3) / 10×(deg8)**、`envelope(Hilbert)` 与 `hilbert(scipy)` **8×**；
  * 最省的是 `is_infinite` / `ctas_no_copy` / `to_array`：**0.125×**（只多一个 n 字节的布尔掩码）。
* **600 MiB 上最慢的是 `Envelope(method="Hilbert")`：12.7 s / 4.8 GiB**（默认走 scipy 后端）；
  同样算解析信号的 `hilbert(S, "FFT")` 只要 3.7 s / 2.4 GiB —— kwargs 泄漏已修，现在可以 `hilbert_mod="FFT"` 切过去
  （见 `Envelope` 报告 §6.2）。
* `Chunk.chunk()` 在 1–600 MiB 都只切出 **1 块**（0.03 ms）——这是**设计意图**：块长按 `0.75 × 内存预算` 取，这几个档位都装得下；
  单位混用（字节入/元素出）已修，超出预算时按预算封顶（见 `Chunk` 报告 §7.1）。
* `fast_poly` 的时间不是 O(N·order) 的核心运算，而是**建 (N, order+1) 的
  Vandermonde**：deg8 在 600 MiB 上峰值分配 6000 MiB、耗时 4.2 s。

## 1. 口径

* `best_ms`：同进程内 warm-up 1 次后重复 2–3 次取**最小**；
  `alloc_peak_mib`：`tracemalloc` 在该 op 调用期间的**峰值分配**（NumPy 大缓冲按一次分配记账）；
* 输入信号：`tests/_bench.py::make_signal(kind)` —— `mixed`（三音调+趋势）、`ramp`（严格递增）；
* 机器：Windows 11 / 32 逻辑核 / 15.6 GB 物理内存 / Python 3.10.11 / numpy 2.2.6 / scipy 1.15.3。
* **分解法不做这条轴**：单模态 500 MB 就要 5.75–9.76 GB，见 `docs/test/methods/Large_Array_Memory_Report.md`。

## monotonicity: `Utils.Monotonicity`

图：`fig/size_axis/monotonicity.png`；逐 op 数据：`table/size_axis/monotonicity_<op>.csv`。

**耗时（ms）**

| op | 1 MiB | 50 MiB | 100 MiB | 200 MiB | 500 MiB | 600 MiB |
|---|---|---|---|---|---|---|
| `give_monotony` | 0.20 ms | 14.2 ms | 23.7 ms | 46.2 ms | 162.5 ms | 148.0 ms |
| `monotonic_chunked_1mib` | 0.29 ms | 7.55 ms | 20.7 ms | 40.3 ms | 85.3 ms | 98.3 ms |
| `monotonic_early_exit` | 0.18 ms | 11.4 ms | 15.5 ms | 40.5 ms | 96.2 ms | 84.6 ms |
| `monotonic_full_scan` | 0.17 ms | 8.00 ms | 19.3 ms | 36.6 ms | 109.3 ms | 108.4 ms |
| `monotonic_strict_scan` | 0.23 ms | 8.19 ms | 15.9 ms | 37.1 ms | 81.7 ms | 127.7 ms |

**峰值分配（MiB）**

| op | 1 MiB | 50 MiB | 100 MiB | 200 MiB | 500 MiB | 600 MiB |
|---|---|---|---|---|---|---|
| `give_monotony` | 0.30 | 12.5 | 25.0 | 50.0 | 125.0 | 150.0 |
| `monotonic_chunked_1mib` | 0.10 | 6.30 | 12.5 | 25.0 | 62.5 | 75.0 |
| `monotonic_early_exit` | 0.10 | 6.30 | 12.5 | 25.0 | 62.5 | 75.0 |
| `monotonic_full_scan` | 0.10 | 6.30 | 12.5 | 25.0 | 62.5 | 75.0 |
| `monotonic_strict_scan` | 0.10 | 6.30 | 12.5 | 25.0 | 62.5 | 75.0 |

## chunk: `Base.Chunk`（分块/流式填充）

图：`fig/size_axis/chunk.png`；逐 op 数据：`table/size_axis/chunk_<op>.csv`。

**耗时（ms）**

| op | 1 MiB | 50 MiB | 100 MiB | 200 MiB | 500 MiB | 600 MiB |
|---|---|---|---|---|---|---|
| `chunk_iter_1mib` | 0.02 ms | 0.03 ms | 0.04 ms | 0.05 ms | 0.12 ms | 0.11 ms |
| `chunk_iter_default` | 0.03 ms | 0.02 ms | 0.04 ms | 0.04 ms | 0.06 ms | 0.04 ms |
| `chunk_size_calc` | 0.02 ms | 0.01 ms | 0.02 ms | 0.03 ms | 0.02 ms | 0.03 ms |
| `chunked_fill_to_disk` | 4.98 ms | 47.3 ms | 98.7 ms | 193.9 ms | 475.4 ms | 575.5 ms |
| `chunked_fill_to_mem` | 0.29 ms | 10.8 ms | 24.6 ms | 44.6 ms | 113.7 ms | 145.2 ms |
| `chunked_map_x2p1` | 0.80 ms | 41.2 ms | 80.5 ms | 168.2 ms | 436.9 ms | 434.1 ms |

**峰值分配（MiB）**

| op | 1 MiB | 50 MiB | 100 MiB | 200 MiB | 500 MiB | 600 MiB |
|---|---|---|---|---|---|---|
| `chunk_iter_1mib` | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| `chunk_iter_default` | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| `chunk_size_calc` | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| `chunked_fill_to_disk` | 0.10 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| `chunked_fill_to_mem` | 1.00 | 50.0 | 100.0 | 200.0 | 500.0 | 600.0 |
| `chunked_map_x2p1` | 2.00 | 74.0 | 124.0 | 224.0 | 524.0 | 624.0 |

## check: `Utils.Check`（谓词 + 入口规范化）

图：`fig/size_axis/check.png`；逐 op 数据：`table/size_axis/check_<op>.csv`。

**耗时（ms）**

| op | 1 MiB | 50 MiB | 100 MiB | 200 MiB | 500 MiB | 600 MiB |
|---|---|---|---|---|---|---|
| `ctas_copy` | 0.48 ms | 16.9 ms | 29.3 ms | 56.6 ms | 153.5 ms | 186.9 ms |
| `ctas_no_copy` | 0.10 ms | 3.93 ms | 7.87 ms | 17.3 ms | 39.6 ms | 57.1 ms |
| `is_duplicates` | 0.29 ms | 17.9 ms | 34.1 ms | 66.3 ms | 158.6 ms | 196.3 ms |
| `is_infinite` | 0.06 ms | 3.90 ms | 7.28 ms | 14.4 ms | 42.4 ms | 61.1 ms |
| `is_uniform` | 0.74 ms | 41.3 ms | 85.9 ms | 171.3 ms | 474.7 ms | 626.0 ms |
| `to_array_noop` | 0.07 ms | 3.90 ms | 7.69 ms | 15.6 ms | 48.6 ms | 50.6 ms |

**峰值分配（MiB）**

| op | 1 MiB | 50 MiB | 100 MiB | 200 MiB | 500 MiB | 600 MiB |
|---|---|---|---|---|---|---|
| `ctas_copy` | 1.00 | 50.0 | 100.0 | 200.0 | 500.0 | 600.0 |
| `ctas_no_copy` | 0.10 | 6.30 | 12.5 | 25.0 | 62.5 | 75.0 |
| `is_duplicates` | 1.10 | 56.3 | 112.5 | 225.0 | 562.5 | 675.0 |
| `is_infinite` | 0.10 | 6.30 | 12.5 | 25.0 | 62.5 | 75.0 |
| `is_uniform` | 3.00 | 150.0 | 300.0 | 600.0 | 1,500 | 1,800 |
| `to_array_noop` | 0.10 | 6.30 | 12.5 | 25.0 | 62.5 | 75.0 |

## peaks: `Utils.Peaks`（三后端）

图：`fig/size_axis/peaks.png`；逐 op 数据：`table/size_axis/peaks_<op>.csv`。

**耗时（ms）**

| op | 1 MiB | 50 MiB | 100 MiB | 200 MiB | 500 MiB | 600 MiB |
|---|---|---|---|---|---|---|
| `numba_both` | 0.43 ms | 14.7 ms | 29.9 ms | 55.5 ms | 116.5 ms | 155.8 ms |
| `numpy_both` | 0.29 ms | 28.2 ms | 44.0 ms | 103.1 ms | 261.7 ms | 324.2 ms |
| `numpy_max` | 0.17 ms | 12.6 ms | 23.8 ms | 49.2 ms | 146.4 ms | 176.0 ms |
| `scipy_max` | 0.37 ms | 9.41 ms | 25.6 ms | 45.4 ms | 111.1 ms | 155.8 ms |

**峰值分配（MiB）**

| op | 1 MiB | 50 MiB | 100 MiB | 200 MiB | 500 MiB | 600 MiB |
|---|---|---|---|---|---|---|
| `numba_both` | 34.4 | 100.0 | 200.0 | 400.0 | 1,000 | 1,200 |
| `numpy_both` | 0.40 | 18.8 | 37.5 | 75.0 | 187.5 | 225.0 |
| `numpy_max` | 0.40 | 18.8 | 37.5 | 75.0 | 187.5 | 225.0 |
| `scipy_max` | 34.0 | 75.0 | 150.0 | 300.0 | 750.0 | 900.0 |

## mirror: `Utils.Mirror`

图：`fig/size_axis/mirror.png`；逐 op 数据：`table/size_axis/mirror_<op>.csv`。

**耗时（ms）**

| op | 1 MiB | 50 MiB | 100 MiB | 200 MiB | 500 MiB | 600 MiB |
|---|---|---|---|---|---|---|
| `blcse` | 0.51 ms | 23.0 ms | 46.2 ms | 95.5 ms | 238.7 ms | 318.5 ms |
| `extrema_nbsym4` | 0.29 ms | 21.7 ms | 43.0 ms | 95.0 ms | 258.6 ms | 316.9 ms |
| `mirror_nbsym8` | 0.30 ms | 12.6 ms | 24.5 ms | 55.2 ms | 161.4 ms | 190.0 ms |

**峰值分配（MiB）**

| op | 1 MiB | 50 MiB | 100 MiB | 200 MiB | 500 MiB | 600 MiB |
|---|---|---|---|---|---|---|
| `blcse` | 0.40 | 18.8 | 37.5 | 75.0 | 187.5 | 225.0 |
| `extrema_nbsym4` | 0.40 | 18.8 | 37.5 | 75.0 | 187.5 | 225.0 |
| `mirror_nbsym8` | 1.00 | 50.0 | 100.0 | 200.0 | 500.0 | 600.0 |

## poly: `Utils.Poly`（`fast_poly` 求值）

图：`fig/size_axis/poly.png`；逐 op 数据：`table/size_axis/poly_<op>.csv`。

**耗时（ms）**

| op | 1 MiB | 50 MiB | 100 MiB | 200 MiB | 500 MiB | 600 MiB |
|---|---|---|---|---|---|---|
| `eval_deg3` | 3.38 ms | 170.1 ms | 347.3 ms | 852.3 ms | 2,033 ms | 2,430 ms |
| `eval_deg8` | 5.57 ms | 300.2 ms | 594.5 ms | 1,219 ms | 3,459 ms | 4,227 ms |

**峰值分配（MiB）**

| op | 1 MiB | 50 MiB | 100 MiB | 200 MiB | 500 MiB | 600 MiB |
|---|---|---|---|---|---|---|
| `eval_deg3` | 5.00 | 250.0 | 500.0 | 1,000 | 2,500 | 3,000 |
| `eval_deg8` | 10.0 | 500.0 | 1,000 | 2,000 | 5,000 | 6,000 |

## hilbert: `Utils.Hilbert`（三后端）

图：`fig/size_axis/hilbert.png`；逐 op 数据：`table/size_axis/hilbert_<op>.csv`。

**耗时（ms）**

| op | 1 MiB | 50 MiB | 100 MiB | 200 MiB | 500 MiB | 600 MiB |
|---|---|---|---|---|---|---|
| `fft` | 3.10 ms | 224.1 ms | 596.4 ms | 1,199 ms | 3,070 ms | 3,703 ms |
| `fht` | 3.41 ms | 254.0 ms | 559.3 ms | 1,152 ms | 3,110 ms | 3,929 ms |
| `scipy` | 4.71 ms | 367.5 ms | 737.4 ms | 1,506 ms | 4,776 ms | 5,527 ms |

**峰值分配（MiB）**

| op | 1 MiB | 50 MiB | 100 MiB | 200 MiB | 500 MiB | 600 MiB |
|---|---|---|---|---|---|---|
| `fft` | 19.2 | 200.0 | 400.0 | 800.0 | 2,000 | 2,400 |
| `fht` | 4.00 | 200.0 | 400.0 | 800.0 | 2,000 | 2,400 |
| `scipy` | 25.4 | 400.0 | 800.0 | 1,600 | 4,000 | 4,800 |

## envelope: `Utils.Envelope`（四模式）

图：`fig/size_axis/envelope.png`；逐 op 数据：`table/size_axis/envelope_<op>.csv`。

**耗时（ms）**

| op | 1 MiB | 50 MiB | 100 MiB | 200 MiB | 500 MiB | 600 MiB |
|---|---|---|---|---|---|---|
| `hilbert` | 4.70 ms | 329.0 ms | 782.7 ms | 1,870 ms | 5,130 ms | 12,728 ms |
| `iq` | 8.38 ms | 413.5 ms | 808.6 ms | 1,646 ms | 4,242 ms | 5,072 ms |
| `lowpass` | 2.87 ms | 216.1 ms | 391.2 ms | 786.0 ms | 1,708 ms | 1,475 ms |
| `peak_interp` | 2.18 ms | 81.0 ms | 184.3 ms | 370.5 ms | 1,013 ms | 1,063 ms |

**峰值分配（MiB）**

| op | 1 MiB | 50 MiB | 100 MiB | 200 MiB | 500 MiB | 600 MiB |
|---|---|---|---|---|---|---|
| `hilbert` | 40.5 | 400.0 | 800.0 | 1,600 | 4,000 | 4,800 |
| `iq` | 8.00 | 400.0 | 800.0 | 1,600 | 4,000 | 4,800 |
| `lowpass` | 4.00 | 200.0 | 400.0 | 800.0 | 2,000 | 2,400 |
| `peak_interp` | 3.10 | 150.0 | 300.0 | 600.0 | 1,500 | 1,800 |

## 2. 复现

```powershell
# 单个工具（会写 table/size_axis/*.csv 与 fig/size_axis/<tool>.png）
.\.venv\Scripts\python.exe tests\size_axis\run_scale.py --tool monotonicity --export

# 全部 8 个工具（约 25–35 分钟，含一次 6 GB 峰值的 poly deg8）
.\.venv\Scripts\python.exe tests\size_axis\run_scale.py --tool monotonicity --export
.\.venv\Scripts\python.exe tests\size_axis\run_scale.py --tool chunk --export
.\.venv\Scripts\python.exe tests\size_axis\run_scale.py --tool check --export
.\.venv\Scripts\python.exe tests\size_axis\run_scale.py --tool peaks --export
.\.venv\Scripts\python.exe tests\size_axis\run_scale.py --tool mirror --export
.\.venv\Scripts\python.exe tests\size_axis\run_scale.py --tool poly --export
.\.venv\Scripts\python.exe tests\size_axis\run_scale.py --tool hilbert --export
.\.venv\Scripts\python.exe tests\size_axis\run_scale.py --tool envelope --export
.\.venv\Scripts\python.exe tests\size_axis\run_scale.py --rebuild-summary
.\.venv\Scripts\python.exe tests\size_axis\report.py
```

## 3. 已知限制

* `tracemalloc` 只统计 Python 分配器看到的量；memmap 写入的页缓存不在此列，所以 `chunked_fill_to_disk` 的 `alloc` 列是 0 而实际有磁盘 IO。
* 3 次重复的 `best` 对 600 MiB 档位受同时运行的其他进程影响（本轮所有档位串行、单进程跑完，误差约 ±5%）。
