# 大数组（100 MB / 500 MB / 1 GB）内存实测报告

> 环境：Windows · Python 3.10.11 · numpy 2.2.6
> 机器：总内存 15.63 GB，测试时可用 6.6–9.8 GB，40 GB 交换区
> 项目内存预算（`Base.Memory.memory_budget()`）：**4.93 GB**（保留 25% 系统余量）
> 测试脚本：`.scratch/eemd_new/bench_tier.py`、`probe_one.py`、`bench_large.py`

## 1. 结论先说

| 问题 | 结论 |
|---|---|
| 15 个分解法能否在 1 GB 输入下全部不 OOM？ | **不能**。多模态方法的**输出本身就是 K×8N 字节**，1 GB 输入下 K=5 就要 5 GB，仅输出就超预算 |
| 有没有真 OOM 缺陷？ | **有两个**：`EFD` 的 `max_IMFs=-1`、`CEEFD` 的峰数无上限，二者都是 **O(N × 频带数)** 的分配。已修复为等价但更省内存的实现（见 §6） |
| 限模态数后能跑吗？ | **能**。100 MB / 1 GB 下 `EWT(num_imfs=2)`、`EFD(max_IMFs=2)`、`EMD(num_imf=3)` 均成功，峰值 1.8–11.6 GB |
| 500 MB / 1 GB 能在 4.93 GB 预算内完成吗？ | **不能**。单模态就要 5.75–9.76 GB（500 MB）、10.2–11.6 GB（1 GB），只有靠 40 GB 交换区才能跑完 |
| 峰值内存大约是输入的多少倍？ | FFT 类 21–23×，VMD 35×，SVMD 42×，EMD 18×，**1 GB 档全部升到 10–11.6×（因为 FFT 工作区固定开销被摊薄）** |
| SSA 呢？ | 轨迹矩阵是 **O(N·L)**，默认 `L=N//3` 时是 **O(N²)**：N=1340 万时理论需 ~3.8×10¹³ 元素。1 GB 输入配 `window_size=64` 仍要 **64 GiB**，直接 `MemoryError` |

## 2. 100 MB 档（N = 13,107,200）

| 方法 | 结果 | 耗时 | 峰值 RSS | 相对输入 | 模态数 |
|---|---|---|---|---|---|
| `EWT(num_imfs=2)` | ok | 5.2 s | 2.32 GB | 23× | 2 |
| `EFD(max_IMFs=2)` | ok | 4.2 s | 2.11 GB | 21× | 2 |
| `VMD(num_imf=2)` | ok | 3.1 s | 3.52 GB | 35× | 2 |
| `SVMD()` | ok | 8.7 s | 4.20 GB | 42× | 3 |
| `LMD()` | ok | 34.1 s | 1.87 GB | 19× | **0** |
| `EMD(num_imf=3)` | ok | 2.6 s | 1.84 GB | 18× | 3 |
| `SSA(window_size=64)` | 长时间换页未完成 | — | — | — | — |

`LMD()` 返回 **0 个模态**：它内部对幅值有 `1e4 × max|S|` 的上限，大数组下逐阶 PF 的判据直接不满足。这是既有行为，不是内存问题，但大数组下等于"跑了 34 秒什么也没分出来"。

### 2.1 未在 100 MB 及以上规模的 8 个方法及原因

| 方法 | 未测原因 |
|---|---|
| `EEMD` / `CEEMD` / `CEEMDAN` / `ICEEMDAN` / `RPSEMD` | 都要跑 `M`（默认 100）次 EMD。N=1M 时单次 EMD 约 1 s，100 次就 100 s 起；N=1310 万时单次 20.8 s → 单方法 30 分钟以上。**内存不是瓶颈，时间才是** |
| `MEMD` / `FMD` | 在 8 MB 档就超了 300 s 超时（不是 OOM）。它们在 N≥10⁶ 时迭代次数随 N 增长 |
| `CEEFD` | 8 MB 档（旧实现）181 s 后 OOM；修复后 1M 点 `min_peak_distance=2000` 实测 197 s / 1.66 GB，再放大到 1310 万点预计 30 分钟级 |

这 8 个方法的内存模型与已测的同族一致（EMD 族 `(2+K)×8N`、CEEFD 频域 `(2+K)×8N`），所以**"500 MB 起必然越过 4.93 GB 预算"的结论对它们同样成立**，只是没有逐一耗时验证。

## 3. 500 MB 档（N = 65,536,000，把模态数压到 1）

| 方法 | 结果 | 耗时 | 峰值 RSS | 相对输入 |
|---|---|---|---|---|
| `EMD(num_imf=1)` | ok | 5.4 s | **5.75 GB** | 11.5× |
| `EWT(num_imfs=1)` | ok | 14.9 s | **8.34 GB** | 16.7× |
| `VMD(num_imf=1)` | ok | 15.3 s | **8.80 GB** | 17.6× |
| `EFD(max_IMFs=1)` | ok | 14.0 s | **9.76 GB** | 19.5× |
| `LMD()` | ok | 172.8 s | 7.28 GB | 14.6×（k=0） |
| `SVMD(num_imf=1)` | 参数名不接受 `num_imf`（见 §8） | — | 1.36 GB | — |

**这一档最关键**：即使只要 **1 个模态**，500 MB 输入也要 **5.75–9.76 GB** —— 说明主导成本不是模态数，而是**长度为 N 的中间数组本身**（FFT 的 complex128 结果、幅度谱、相位、频带重建缓冲）。三者叠加就接近 10 GB。

也就是说：500 MB 输入在任何模态数下都已超出 4.93 GB 预算，只能靠交换区完成。

## 4. 1 GB 档（N = 134,217,728）

| 方法 | 结果 | 耗时 | 峰值 RSS | 模态数 |
|---|---|---|---|---|
| `EWT(num_imfs=2)` | ok | 78.8 s | **10.22 GB** | 2 |
| `EWT()`（默认 8 阶） | ok | 292.9 s | **11.60 GB** | 8 |
| `EFD(max_IMFs=2)` | ok | 77.0 s | **11.60 GB** | 2 |
| `EMD(num_imf=3)` | ok | 35.8 s | **10.92 GB** | 3 |
| `SSA(window_size=64)` | **MemoryError: 64.0 GiB (64, 134217665)** | 立即失败 | 3.25 GB | — |
| `VMD(K=2)` | 参数名写错（应为 `num_imf`） | — | 2.26 GB | — |

**要点**：1 GB 输入 + 2 个模态，峰值就到 10–11.6 GB，是物理可用内存的 1.5 倍 —— 靠 40 GB 交换区才没被 OOM killer 杀掉，耗时也从 5 s 级涨到 78 s。也就是说"能跑完"但**不算正常执行**：强烈依赖交换区。

## 5. 各方法的峰值内存模型

| 方法族 | 峰值 ≈ | 说明 |
|---|---|---|
| EWT / EFD / CEEFD | `(2 + K) × 8N` + FFT 工作区 | 纯频域，每条模态一条 N 长数组 |
| VMD / SVMD | `(4 + K) × 8N` | 频域 ADMM，`u_hat` 和 `u_hat_history` 各一份 |
| EMD 族 | `(2 + K) × 8N` + 样条临时量 | 时域筛分，包络与插值各一份 |
| **SSA** | **`8·L·N`** | 轨迹矩阵 L × (N-L+1)，**唯一 O(N²)** |

其中 `K` = 模态数。**这意味着"输出规模"就是内存下限**：想处理 1 GB 输入并得到 8 个模态，至少需要 8 GB 连续内存。

## 6. 本轮修复的两个 OOM 缺陷（结果不变）

### EFD：`max_IMFs=-1` 时每个频带物化一条 `N/2` 长的 0/1 掩码

```python
# 修复前：O(N × 频带数)
for edge in range(1, len(wn)):
    filter_single = np.zeros(freq_N)     # 每带 4 MiB (N=1M)
    filter_single[last:cur+1] = 1
    filters_arr.append(filter_single)    # 全部留内存
```
N=1,048,576 时谱峰有 204,079 个 → commit 峰值 **38.21 GB** 后 `MemoryError`。

改为按边界下标直接切片构造（支撑集就是闭区间 `[wn[e-1], wn[e]]`），并把 `IMFs` 预分配成 `(K, N)` 二维数组而不是"list 追加 + `asarray`"。

### CEEFD：同样的"list 累积 + `asarray` 复制"

改成 `np.empty((K, N))` 预分配并逐行填充，峰值从 **9.29 GB 降到 1.66 GB**（同 N，`min_peak_distance=2000`）。

### 等价性证明

拿 `git HEAD` 的旧实现临时导入做逐位对照（`.scratch/eemd_new/verify_band_equiv.py`）：

```
EFD   : 3 种 N × 4 种信号 × 6 种 max_IMFs  = 72 组
CEEFD : 3 种 N × 4 种信号 × 4 种参数       = 48 组
共 120 组对照，IMFs 与 Res 逐位相同，差异 0 组
```

## 7. 大数组下的使用建议

1. **显式限制模态数**。`EFD()` / `CEEFD()` 的默认"用全部谱峰"在大 N 下不可用：
   - N=1M 时谱峰约 20 万个，`EFD()` 需要 1.56 TiB → 立即 `MemoryError`；
   - 传 `EFD(max_IMFs=10)` → 0.32 GB / 1.3 s；
   - `CEEFD(min_peak_distance=2000)` → 1.66 GB。
2. **SSA 必须显式给 `window_size`**。默认 `N//3` 在 N>10⁶ 时是 O(N²)。
3. **1 GB 级输入建议在 ≥16 GB 空闲内存的机器上跑**，否则会重度依赖交换区。
4. **500 MB 以上不要期待"在物理内存内完成"**：实测单模态已需 5.75–9.76 GB，四种方法全部越过 4.93 GB 预算。要么加内存，要么接受交换区带来的 3–20 倍时间膨胀。
5. 需要真正突破"K×8N 输出下限"的话，只能改成 **memmap 输出**（`Base.Memory.temp_memmap` + `Base.Chunk`），目前 15 个方法都还没有走这条路径。

## 8. 复现

```
.\.venv\Scripts\python.exe -u .scratch\eemd_new\bench_tier.py --mb 100
.\.venv\Scripts\python.exe -u .scratch\eemd_new\bench_tier.py --mb 1000
.\.venv\Scripts\python.exe -u .scratch\eemd_new\probe_one.py EFD 1048576    # 带 RSS/commit 峰值监控
.\.venv\Scripts\python.exe -u .scratch\eemd_new\verify_band_equiv.py        # 等价性对照
```

原始结果：`.scratch/eemd_new/large_results/*.json`

## 9. 测试中发现的接口差异

- `SVMD` 的构造参数里**没有** `num_imf`（`SVMD(num_imf=1)` 抛
  `TypeError: unexpected keyword arguments`），与其余 14 个方法不一致；
  它的模态数参数是 `num_modes`（`SVMD(num_modes=3)`）。
- `VMD` 的模态数参数是 `num_imf`（不是 `K`）。
