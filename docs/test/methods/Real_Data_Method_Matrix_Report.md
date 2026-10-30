# 真实数据多方法分解对比（EMD 家族 / VMD / FMD / EFD / EWT …）

> 数据表：`table/methods/real_data/methods_matrix.csv`（39 格）
> 脚本：`tests/comparison/bench_real_data.py`（数值）、`tests/comparison/plot_real_data.py`（图）
> 图：`table/methods/real_data/*.png`（待沙箱放行后迁入 `fig/methods/real_data/*.png`）
> 测试：`tests/methods/test_real_data_smoke.py`（数据缺失自动 skip）

## 1. 数据与方法

| 数据 | 内容 | n |
|---|---|---|
| `cwru4k` / `cwru8k` | CWRU `IR007_105_0.mat` 驱动端 `X105_DE_time`（内圈故障，12 kHz） | 4096 / 8192 |
| `ecg4k` | PySDKit `ecg_055m.npy`（心电，含 QRS 尖峰） | 4096 |
| `demo8k` | PySDKit `fmd_demo.npy` 实测信号 | 8192 |

方法：EMD / EEMD / CEEMD / CEEMDAN / ICEEMDAN / RPSEMD、EFD、EWT、LMD、VMD、SVMD、SSA、
FMD（`solver=ck` / `mckd`），以及 PySDKit 的 EMD / EFD / FMD / VMD 作为外部基线。
统一参数：EMD 家族 `num_imf=5, M=10`；EFD `max_IMFs=5`；EWT `num_imfs=4`；LMD `max_pf=3`；
VMD `num_imf=3, n=100`；SVMD `num_modes=3`；SSA `window_size=128`；FMD `K=3, max_iter=10, num_hand=3`。

指标：`wall_ms`（warm-up 后 2 次最快）、`k` 阶数、`recon` 重构最大误差、`IO` Huang 正交指数、
`kurt_max` 各阶峰度最大值（冲击性，轴承故障的关键）、`kurt_top` 能量最大阶的峰度。

## 2. CWRU 内圈故障（n=4096）

| 方法 | 时间 ms | 冲击性 kurt_max | IO | 备注 |
|---|---|---|---|---|
| **MD-EFD** | **0.34** | 1.9 | −0.000 | PySDKit-EFD 2.9 ms → **8.5×** |
| MD-EWT | 1.6 | 1.9 | 0.046 | 频带直切，忠实分带 |
| MD-VMD | 2.9 | 1.7 | 0.036 | PySDKit-VMD **276 ms**（torch 实现）**96×** |
| MD-EMD | 3.6 | 2.1 | 0.012 | PySDKit-EMD 30.7 ms → **8.5×** |
| MD-SVMD | 8.6 | 0.6 | 0.031 | 模态最平滑 |
| MD-RPSEMD | 29 | 2.4 | 0.065 | — |
| MD-LMD | 31 | **1553** | −0.004 | PF 在尖峰数据上极端峰化，不代表更好 |
| MD-EEMD | 55 | 2.3 | 0.057 | 集合平均把冲击抹平 |
| MD-ICEEMDAN | 78 | 2.3 | 0.046 | — |
| MD-CEEMDAN | 79 | **12.0** | 0.092 | EMD 家族里冲击性最强 |
| MD-CEEMD | 106 | 2.3 | 0.059 | — |
| **MD-FMD (`ck`)** | 327 | **126** | −0.018 | PySDKit-FMD 492 ms / kurt 7.1 → 1.5× 快、**18× 冲击性** |
| MD-FMD (`mckd`) | 328 | 2.7 | −0.016 | 该数据上 `ck` 明显优于 `mckd` |
| MD-SSA | 450 | 22 | 3.29 | k=128，分量重叠，IO 无意义 |

n=8192 时结论不变：FMD `ck` **637 ms / kurt 290** vs PySDKit **755 ms / kurt 6.4**（1.2× 快、**45×**）；
EFD 0.67 vs 4.04 ms（6×）；VMD 5.6 vs 769 ms（137×）。

## 3. 交叉域结果：FMD 的边界

| 数据 | MD-FMD `ck` kurt_max | PySDKit-FMD kurt_max |
|---|---|---|
| CWRU（周期冲击） | **126 / 290** | 7.1 / 6.4 |
| `ecg4k`（QRS 尖峰，非周期） | 0.6 | **51.7** |
| `demo8k`（宽带实测） | 0.2（`mckd` 1.5） | **4.1** |

CK 目标为"周期冲击"设计，在轴承故障上碾压；冲击不按固定周期出现时，PySDKit 的
Hanning 滤波器组 + MCKD 覆盖更广。这是目标函数作用域差异，不是参数问题。

## 4. 图

* `cwru4k_decompositions.png` —— 6 种方法的"输入 + 各阶模态"叠置图；
* `cwru4k_envelope.png` —— 各方法最冲击模态的包络谱（轴承诊断视图）；
* `ecg4k_decompositions.png` / `ecg4k_envelope.png` —— 同一视图下的 ECG 对比。

图上可直接看出：FMD 在 CWRU 上抽出的是窄带强冲击的"共振冲激"，而 EMD/EFD/EWT/VMD 是
宽带带通分量；ECG 上反过来，EMD 的 IMF1 与 QRS 对齐最好。

## 5. 复现

```powershell
.\.venv\Scripts\python.exe tests\comparison\bench_real_data.py    # 数值表
.\.venv\Scripts\python.exe tests\comparison\plot_real_data.py     # 图
.\.venv\Scripts\python.exe -m pytest tests/methods/test_real_data_smoke.py -q
```
