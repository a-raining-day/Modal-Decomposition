# VMD 三方对比报告：MD-VMD / vmdpy / PySDKit

本报告补齐 VMD 的**第三方独立实现**对比。此前仓库里只有
`docs/VMD_Native_vs_vmdpy_Report.md`（原生实现 vs `vmdpy`，即"换版"对比），
以及 `docs/Decomposition_Methods_Timing_Report.md` §1 里一列 PySDKit 计时。
本报告把三方放在**同一进程、同一脚本、同一 parity 配置**下重新测一遍，
并首次给出 PySDKit 列的**质量指标**。

- 实验脚本：`tests/comparison/bench_timing.py --method VMD`
- 汇总脚本：`tests/comparison/summarize_vmd_pysdkit.py`
- 原始数据：`tests/comparison/results/vmd_pysdkit/{timing_raw.csv,timing_metrics.json,summary.md}`
- 机器可读结果：`docs/VMD_Native_vs_vmdpy_vs_PySDKit_Results.json`

---

## 1. 口径与 parity 配置

| 项 | 取值 |
|---|---|
| 实现 | `MD-VMD` = `Class.VMD`（原生 ADMM，2026-09 起不再依赖 vmdpy）；`vmdpy` = `vmdpy.VMD` 直调；`PySDKit` = `pysdkit.VMD.fit_transform` |
| `alpha` | 2000 |
| `tau` | 0.0 |
| `init` | uniform（MD 侧为 `init_mod="uniform"`） |
| `DC` | False |
| `tol` / `epsilon` | 1e-6 |
| `max_iter` / `n` | 500（三方一致） |
| K | case A=3, B=3, C=4 |
| 计时口径 | 同进程、同脚本、预热后 **3 次重复取中位数** |
| PySDKit 版本 | 0.5.0（`.venv` 内安装） |

### 1.1 一个必须先说明的修复

本报告运行前，`tests/comparison/workers.py` 的 `run_vmd_stack` 里 MD 列**是坏的**：
它仍在用 VMD 重构前的废弃参数名

```python
_MD.Class.VMD(alpha=2000, tau=0.0, K=int(K), DC=0, init=1, tol=tol)
```

而现行 `VMD` 只认 `num_imf` / `init_mod` / `epsilon`，未知名直接抛 `TypeError`。
也就是说，**在此之前 `--method VMD` 的整个 MD 列都无法运行**，
`docs/Decomposition_Methods_Timing_Report.md` §1 的 VMD 数字无法用现行脚本复现。

本次已修正为现行参数名，并让三列都**补上残差行**（`S − Σmodes`），
否则三列的 `n_rows` 与正交性指标不可比。修正后三方均通过自检（见 §4）。

---

## 2. 计时（中位数，秒；越小越好）

### case A（双音 37 Hz + 113 Hz）

| n | MD-VMD | vmdpy | PySDKit | vmdpy/MD | PySDKit/MD | PySDKit/vmdpy |
|---:|---:|---:|---:|---:|---:|---:|
| 256 | 0.0015 | 0.0019 | 0.0330 | 1.3× | **22.3×** | 17.4× |
| 1024 | 0.0024 | 0.0066 | 0.0782 | 2.7× | **32.0×** | 11.9× |
| 4096 | 0.0114 | 0.0529 | 0.2523 | 4.7× | **22.2×** | 4.8× |
| 16384 | 0.7295 | 3.1071 | 4.3707 | 4.3× | 6.0× | 1.4× |
| 65536 | 2.5232 | 19.5730 | 34.6092 | 7.8× | **13.7×** | 1.8× |

### case B（AM-FM + 趋势）

| n | MD-VMD | vmdpy | PySDKit | vmdpy/MD | PySDKit/MD | PySDKit/vmdpy |
|---:|---:|---:|---:|---:|---:|---:|
| 256 | 0.0754 | 0.1092 | 0.1069 | 1.4× | 1.4× | 1.0× |
| 1024 | 0.0048 | 0.0099 | 0.2275 | 2.0× | **47.1×** | 23.0× |
| 4096 | 0.0095 | 0.0239 | 0.6145 | 2.5× | **64.7×** | 25.7× |
| 16384 | 0.0663 | 0.1847 | 4.1380 | 2.8× | **62.4×** | 22.4× |
| 65536 | 0.1691 | 1.1177 | 25.9028 | 6.6× | **153.2×** | 23.2× |

### case C（白噪声，sifting 压力）

| n | MD-VMD | vmdpy | PySDKit | vmdpy/MD | PySDKit/MD | PySDKit/vmdpy |
|---:|---:|---:|---:|---:|---:|---:|
| 256 | 0.0061 | 0.0131 | 0.1730 | 2.1× | **28.2×** | 13.2× |
| 1024 | 0.0368 | 0.0475 | 0.2870 | 1.3× | 7.8× | 6.0× |
| 4096 | 0.1535 | 0.6446 | 0.8755 | 4.2× | 5.7× | 1.4× |
| 16384 | 1.8711 | 5.8079 | 6.5198 | 3.1× | 3.5× | 1.1× |

> case C 未跑 65536：白噪声下三方都几乎不提前收敛，单格 >150 s，超出本次预算。

### 2.1 结论

- **MD-VMD 在全部 14 个可比格子上都快于两个第三方实现**，对 vmdpy 快
  **1.3–7.8×**，对 PySDKit 快 **1.4–153×**。
- 对 PySDKit 的加速比**随信号长度增长**（case B：n=1024 的 47× → n=65536 的 153×），
  因为 PySDKit 的每次 ADMM 迭代开销更大，长度放大后差距累积。
- **case C（噪声）是差距最小的场景**：任何引擎在噪声上都几乎跑满迭代预算，
  共有的固定迭代成本把差距压到 1.1–3.5×。

---

## 3. 质量（数值精度）

### 3.1 质量指标（各 case 的最大可比尺寸）

| case | impl | n | rows | recon max abs err | orthogonality idx | 模态恢复（best abs corr） |
|---|---|---:|---:|---:|---:|---|
| A | MD-VMD | 65536 | 4 | 2.78e-17 | 4.5209e-03 | tone_37hz=0.9993, tone_113hz=0.9973 |
| A | vmdpy | 65536 | 4 | 2.78e-17 | 4.5209e-03 | 同上 |
| A | PySDKit | 65536 | 4 | 2.78e-17 | 4.5205e-03 | 同上 |
| B | MD-VMD | 65536 | 4 | 1.73e-18 | 4.4523e-02 | amfm=0.9996, tone_89hz=0.9999, trend=0.9948 |
| B | vmdpy | 65536 | 4 | 1.73e-18 | 4.4521e-02 | 同上 |
| B | PySDKit | 65536 | 4 | 1.73e-18 | 4.4525e-02 | 同上 |
| C | MD-VMD | 16384 | 5 | 4.44e-16 | 5.9595e-01 | （纯噪声，无真值） |
| C | vmdpy | 16384 | 5 | 4.44e-16 | 5.9595e-01 | 同上 |
| C | PySDKit | 16384 | 5 | 4.44e-16 | 5.9594e-01 | 同上 |

### 3.2 三方差异有多大（精确量化）

> 更正一处易犯的表述：三方**不是"逐位完全相同"**。三者是同一个 ADMM 求解器的
> 三个独立实现，浮点求和顺序不同，因此结果在**浮点级**上有差异。下表是全网格实测
> 的最坏相对差（以 MD-VMD 为基准）：

| case | 正交性指数最大相对差 | 模态 corr 最大绝对差 | 重构误差是否同值 |
|---|---:|---:|---|
| A（双音，K=3） | 1.38e-04 | 9.45e-08 | 14 格中 13 格同值 |
| B（AM-FM+趋势，K=3） | **1.50e-03** | **7.04e-04** | 14 格中 11 格同值 |
| C（白噪声，K=4） | 4.46e-04 | 0.00e+00 | 14 格中 12 格同值 |

- **量级**：正交性指数（自身量级 4.5e-03 ~ 6.0e-01）的相对差在 **1e-5 ~ 1.5e-03**；
  模态相关系数的绝对差 ≤ **7.0e-04**。即三方是"同一个解"意义上的等价，
  但**不是** bit-level 一致。
- **最大差异出现在 case B / n=256**（正交性 1.5e-03、corr 7.0e-04），
  正是 §4.2 中"三方都跑满 500 步未收敛"的那一格 —— 未收敛时停机点对
  浮点扰动更敏感，差异被放大。
- 对实践的含义：**选 MD-VMD 的理由是速度与内存，不是精度**。三方精度等价到这个
  量级，任何质量对比都不会改变结论。

这与 `docs/VMD_Native_vs_vmdpy_Report.md` 的结论方向一致（原生实现与 vmdpy
同质量：ω 差 ≤4e-9、残差比与迭代数逐项相同），本报告把它扩展到了 PySDKit，
并给出了三方差异的量化上界。

---

## 4. 自检与异常

### 4.1 三方自检

修正 `run_vmd_stack` 后，三列在 case A / n=1024 上的自检结果：

```
MD-VMD   -> stack (4, 1024) | recon err 2.78e-17
vmdpy    -> stack (4, 1024) | recon err 2.78e-17
PySDKit  -> stack (4, 1024) | recon err 2.78e-17
```

### 4.2 已解释的反直觉格子：case B / n=256 三方都慢

case B 在 n=256 时耗时（0.075 s / 0.109 s / 0.107 s）**高于** n=1024
（0.0048 s / 0.0099 s / 0.2275 s）。根因已定位，不是测量噪声：

| n | n_iter | wall(s) | converged |
|---:|---:|---:|---|
| 256 | **500** | 0.0119 | **False** |
| 1024 | 20 | 0.0011 | True |
| 4096 | 19 | 0.0028 | True |
| 16384 | 20 | 0.0241 | True |

n=256 时 VMD **跑满 500 次迭代仍未收敛**；n≥1024 时约 20 次即收敛。
原因是 case B 的 AM-FM 分量中心频率 89 Hz 带宽很窄，而 n=256 的频谱分辨率
约 3.9 Hz/格（相对带宽 Q≈2.5），VMD 在这么短的样本上无法把该窄带模态
从相邻成分中分辨出来 —— 这是 VMD 本身对"短信号 + 窄带"的已知限制，
**三方同时受限**，不是任何一个实现的缺陷。

### 4.3 计时离群值

case A / n=65536 的三次原始 wall 为：

```
MD-VMD   2.2224 / 3.2678 / 2.5232
vmdpy   25.1787 / 19.5730 / 17.4041
PySDKit 34.6092 / 28.0561 / 36.8505
```

单次波动可达 ±30%，故报告一律用**中位数**，且加速比只做同格比较。

### 4.4 复现并修正 PySDKit `store_history` 的归因

`docs/Decomposition_Methods_Timing_Report.md` §1 把 PySDKit 的慢归因于
`store_history=True`（默认）。本次**直接实测验证**了该归因，并修正了旧报告
一处不准确的表述：

| n | `store_history` | 中位 wall (s) | 实际迭代数 | 与 `False` 的模态差 |
|---:|---|---:|---:|---:|
| 1024 | True | 0.07417 | — | 1.205e-04 |
| 1024 | **False** | **0.00283** | — | — |
| 4096 | True | 0.27542 | **498** | 8.556e-05 |
| 4096 | **False** | **0.00892** | **19** | — |
| 16384 | True | 4.43551 | — | 3.748e-05 |
| 16384 | **False** | **0.13372** | — | — |

- **归因成立**：PySDKit 的 `omega` 历史长度直接暴露了迭代数 —— 同一条 case B /
  n=4096，`store_history=True` 跑了 **498** 次 ADMM 迭代，`store_history=False`
  只跑 **19** 次。默认路径的收敛累加器不重置，于是**永不满足 `tol` 判据**，
  每次都跑满 `max_iter`。实测加速 **26–33×**，远大于与 vmdpy 的差距。
- **修正旧报告**：旧报告称两种设置下"模式数值不变（corr ~ 1）"。实测并**非**
  数值不变，模态最大差 **3.7e-05 ~ 1.2e-04**。原因是两者在**不同迭代步停机**：
  默认路径跑到 498 步（未收敛），关闭后 19 步就收敛退出，落点是不同的解
  （§2 的三方一致也由此而来：三方都跑满 500 步时落在同一个未收敛解上）。
- **可行动项**：只要用 PySDKit 的 VMD，就应显式传 `store_history=False`；
  否则既慢 26–33×，拿到的还是未收敛解。本报告 §2 的 PySDKit 列**沿用其默认**，
  因此那些数字应理解为"其默认（非收敛）配置下的耗时"，而非其最优耗时。

---

## 5. 局限与未验证项

1. **未做内存对比。** `bench_timing.py` 只采 RSS delta；`docs/VMD_Refactor_and_Validation_Report.md`
   给出的内存结论（原生实现峰值低于 vmdpy、受限预算下落盘再降 25–38%）
   未在 PySDKit 上复测。
2. **未测 `engine="chunked"` / `out_of_core`。** 三方对比只覆盖 `engine="auto"`
   的默认路径；MD-VMD 的分块与外存引擎是它独有的能力，第三方无对应物。
3. **未测 `init_mod="peak"`。** 该初值是 MD-VMD 原生实现独有的，vmdpy/PySDKit
   没有对应项，无法 parity 对比。
4. **PySDKit 的 `store_history` 未纳入控制。** 本次沿用其默认
   `store_history=True`；§4.4 实测表明关掉它可快 **26–33×**（4096 点：0.275 s → 0.0089 s），
   且会**改变结果**（跑到收敛而非跑满 500 步）。因此 §2 中 PySDKit 的绝对耗时
   应理解为"其默认（未收敛）配置下的耗时"，而非"其最优耗时"。
5. **case C / n=65536 缺失**（预算原因），因此噪声场景的长时间行为未覆盖。
6. **K 固定为 3/3/4。** 未扫描 K 对三方差异的影响。

---

## 6. 复现

```powershell
$env:PYTHONPATH = "$PWD\src"
.\.venv\Scripts\python.exe tests\comparison\bench_timing.py --method VMD --repeats 3 --out tests\comparison\results\vmd_pysdkit
.\.venv\Scripts\python.exe tests\comparison\summarize_vmd_pysdkit.py
```

第二行会重写 `tests/comparison/results/vmd_pysdkit/summary.md` 与
`docs/VMD_Native_vs_vmdpy_vs_PySDKit_Results.json`。

---

## 7. 相关文档

| 文档 | 覆盖范围 |
|---|---|
| `docs/VMD_Refactor_and_Validation_Report.md` | 原生实现 vs vmdpy 的**换版**验证（同质量、3–9× 更快、重构由 17–26% 误差变为精确） |
| `docs/VMD_Native_vs_vmdpy_Report.md`、`docs/VMD_Native_vs_vmdpy_Results.json` | 换版性能/内存明细 |
| `docs/VMD_Large_Array_Iteration_Report.md` | 并行与分块/外存的评估（**历史归档**：其"不引入分块"结论已被三态引擎取代） |
| **本报告** | 三方（MD / vmdpy / PySDKit）同口径计时 + 首次 PySDKit 质量指标 |
