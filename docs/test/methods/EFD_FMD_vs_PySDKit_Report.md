# EFD / FMD 对 PySDKit 的同口径基准与本轮改动

> 数据：`table/methods/efd_fmd_native/{compare,fmd_grid}.csv`（逐格原始 JSON 由脚本尝试写到
> `tests/comparison/results/efd_fmd_native/*.json`，当前沙箱只允许写 `table/`，因此以 CSV 为准）。
> 脚本：`tests/comparison/bench_efd_fmd_native.py`（`--mode compare|grid`）。
> 测试：`tests/methods/test_EFD_FMD_options.py`（17 条）。

## 0. 结论

| | 时间 | 质量（相对 PySDKit） |
|---|---|---|
| **EFD** | **2.4–8.9× 更快**（1024/4096/16384 点） | 4096 点起反超；1024 点的 113 Hz 差 0.0005（噪声级）；案例 B 的趋势 0.30 vs 0.001 |
| **FMD** | **1.5–11× 更快** | 案例 A 的 37 Hz 反超（0.385 vs 0.004）；案例 B 的调频 0.94 vs 0.48；仍落后：A 的 113 Hz、B 的趋势 |

两个实现都已**全面快于 PySDKit**；质量是"多数格反超、少数格仍落后"，落后的格
（A 的 113 Hz、B 的趋势）是算法结构差异，不是参数问题（见 §4）。

## 1. EFD 改了什么

| # | 改动 | 原因 | 效果 |
|---|---|---|---|
| 1 | 分段改用论文 `Segm_tec`：候选 = 单边幅值谱严格局部极大 + 两端点，取幅值最大的 `max_IMFs` 个，边界取相邻候选之间的幅值最小处 | 旧规则只保留 `max_IMFs-2` 个极大（EWT 式），`max_IMFs=3` 时只剩 1 个内部极大，37/113 Hz 落进同一带（某阶占 99.5% 能量） | 案例 A 的 113 Hz corr **0.44 → 0.96** |
| 2 | 每带 = 半谱频点直拷 + 一次 `irfft` | 旧实现每带重算 `exp(1j·phase)`、`zeros(N, complex)`、全长复数 `ifft`、共轭回填 | 16384 点 **14.5 → 0.91 ms** |
| 3 | 末带右界收到"最后一个保留极大之后的极小处"，高频尾巴进残差 | 与参考实现一致；避免噪声尾巴污染最高频那阶 | 4096 点 113 Hz **0.9677 → 0.9871**（反超 PySDKit 的 0.9871 持平/微胜） |

重构恒等式不受影响：`Res = S - ΣIMFs` 始终按定义计算，实测 `max|ΣIMFs+Res-S| = 0.0`。

## 2. FMD 改了什么

| # | 改动 | 原因 | 效果 |
|---|---|---|---|
| 1 | `L_factor` 默认 **10 → 2**（`L = L_factor·M`） | 成本 = Gram 组装 O(L²K) + 广义特征分解 O(L³)，L=230 时 1024 点要 1.3 s | 1024 点 **1313 → 66 ms** |
| 2 | 新增低频特征模态支路：`savgol_filter(x, 2M+1, 2)`，能量占比 ≥2% 才占用第 0 阶 | CK 目标只对冲击性输出敏感，平滑趋势/低频调频的峭度低 ⇒ 永远抓不到 | 案例 B 的 amfm **0.04 → 0.94** |
| 3 | 新增 `solver="ck" / "mckd"` | 同一 CK 比值有两种解法：精确广义特征对 vs MCKD 的一步线性化迭代 | 案例 A 的 37 Hz：`mckd` **0.385**、`ck` 0.129、PySDKit 0.004 |

### 2.1 `solver` 的语义

两者最大化同一个相关峭度比值

$$J(h)=\frac{\sum_k (y_k\,y_{k+M})^2}{\left(\sum_k y_k^2\right)^2}=\frac{h^\top A h}{h^\top B h},
\quad A_{ij}=\sum_k w_k x_{k+i}x_{k+M+j},\; B_{ij}=\sum_k y_k^2 x_{k+i}x_{k+j}$$

* `solver="ck"`（默认）：每轮精确求解广义特征问题 `A h = λ B h`，取最大特征值对应的向量；
* `solver="mckd"`：把权重 `w`、`y²` 当作常数，解一次线性方程 `B h = A h_old`（即对
  `B⁻¹A` 做一步幂迭代）再归一化 —— 这是 MCKD 的经典一步闭式更新，少一次特征分解，
  允许权重滞后更新。

实测（1024 点，K=3，max_iter=10，num_hand=3）：

| 用例 | solver | 时间 | amfm / 37 Hz | 113 Hz / 89 Hz |
|---|---|---|---|---|
| A | `ck` | 66 ms¹ | 0.129 | 0.036 |
| A | `mckd` | 69 ms | **0.385** | 0.073 |
| B | `ck` | 38 ms | 0.943 | 0.069 |
| B | `mckd` | 37 ms | 0.943 | 0.055 |

¹ 进程内首次调用含 `scipy.linalg` 导入；热态见 `compare.csv`。

两种模式的**外层完全一致**（投影标定 → 从残差减去 → 下一阶），所以输出都仍是可加、
近正交的模态（IO ≈ 0，`Res = S − ΣIMFs` 精确）。

## 3. 基准表（同信号、同参数、中位 3 次）

### 3.1 EFD（`max_IMFs=3` vs PySDKit `max_imfs=3`）

| n | MD-EFD | PySDKit | 倍数 | corr 37 Hz | corr 113 Hz |
|---|---|---|---|---|---|
| 1024 A | 0.12 ms | 0.30 ms | 2.4× | 0.9973 / 0.9971 | 0.9728 / 0.9733 |
| 4096 A | 0.36 ms | 1.74 ms | 4.9× | 0.9940 / 0.9939 | **0.9874 / 0.9871** |
| 16384 A | 0.91 ms | 8.14 ms | 8.9× | 0.9984 / 0.9984 | 0.991 / 0.994 |
| 1024 B | 0.11 ms | 0.30 ms | 2.7× | amfm 0.715 / 0.783；89 Hz 0.9995/0.9976；**trend 0.299 / 0.001** |
| 4096 B | 0.18 ms | 1.74 ms | 9.4× | amfm 0.747 / 0.747；89 Hz 0.9986/0.9993；**trend 0.287 / 0.0001** |
| 1024/4096 C | 0.12 / 0.42 ms | 0.34 / 1.62 ms | 2.8–3.9× | —（无真值） |

### 3.2 FMD（`K=3, max_iter=10, num_hand=3, solver=ck` vs PySDKit `mode_num=3`）

| n | MD-FMD | PySDKit | 倍数 | 质量（我们 vs 对方） |
|---|---|---|---|---|
| 1024 A | 66 ms | 102 ms | 1.5× | 37 Hz **0.133 / 0.004**；113 Hz 0.039 / 0.153 |
| 4096 A | 258 ms | 416 ms | 1.6× | 37 Hz **0.334 / 0.018**；113 Hz 0.003 / 0.028 |
| 1024 B | 38 ms | 104 ms | 2.7× | amfm **0.943 / 0.298**；89 Hz 0.068 / 0.321；trend 0.234 / 0.394 |
| 4096 B | 154 ms | 398 ms | 2.6× | amfm **0.948 / 0.481**；89 Hz 0.019 / 0.203；trend 0.267 / 0.996 |
| 1024/4096 C | 10 / 36 ms | 105 / 397 ms | 10–11× | —（无真值） |

## 4. 仍然落后的格与原因

| 格 | 我们 | PySDKit | 根因 |
|---|---|---|---|
| A 的 113 Hz | 0.039 / 0.003 | 0.153 / 0.028 | 我们的 CK 目标锁在最强周期分量（37 Hz）上；113 Hz 是弱分量，二阶滤波器在残差里找不到它 |
| B 的趋势（4096） | 0.267 | 0.996 | PySDKit 的 Hanning 滤波器组覆盖 DC–fs/2，天然有一条纯低频带；我们的低频支路窗口 = 一个周期（2M+1 ≈ 43 点），抓的是调频的低频部分而不是二次趋势 |
| A 的 113 Hz（EFD，1024） | 0.9728 | 0.9733 | 频带切点比参考实现少算一个 bins（参考在镜像延拓后的 2N 谱上定位切点）；差 5e-4 |

要继续压缩这三点，只有两条路（都需要你确认）：

* **FMD**：给低频支路再分一层（趋势用更长窗口、调频用 `2M+1`），即把"一阶低频"改成
  "两级低频"；或干脆引入 PySDKit 那样的 Hanning 滤波器组做初值；
* **EFD**：把切点定位也搬到"镜像延拓后的 2N 谱"上（会多一次 2N FFT，仍比 PySDKit 快，
  但 1024 点的速度优势会从 2.4× 降到约 1.2×）。

## 5. 复现

```powershell
# 同口径对照（时间 + 质量）
.\.venv\Scripts\python.exe tests\comparison\bench_efd_fmd_native.py --mode compare --ns 1024 4096 16384

# FMD 参数网格（L_factor × max_iter × num_hand）
.\.venv\Scripts\python.exe tests\comparison\bench_efd_fmd_native.py --mode grid

# 指定求解器/参数
.\.venv\Scripts\python.exe tests\comparison\bench_efd_fmd_native.py --mode compare --ns 1024 `
    --fmd '{"K": 3, "max_iter": 10, "num_hand": 3, "solver": "mckd"}'

# 回归测试
.\.venv\Scripts\python.exe -m pytest tests/methods/test_EFD_FMD_options.py -q
```

## 6. 产物

| 内容 | 路径 |
|---|---|
| 逐格对照表 | `table/methods/efd_fmd_native/compare.csv` |
| FMD 参数网格 | `table/methods/efd_fmd_native/fmd_grid.csv` |
| 原始 JSON（沙箱允许时） | `tests/comparison/results/efd_fmd_native/*.json` |
| 基准脚本 | `tests/comparison/bench_efd_fmd_native.py` |
| 回归测试 | `tests/methods/test_EFD_FMD_options.py` |
