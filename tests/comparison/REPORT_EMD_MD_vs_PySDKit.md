# EMD 对比报告: Modal_Decomposition.EMD vs PySDKit.EMD (+PyEMD 基线)

> 日期: 2026-09-06 · 目录: `tests/comparison` · 原始数据: `results/`(csv/json 可复现)
> 结论速览: **两者性能与内存几乎一致; pysdkit 的 EMD 是 PyEMD 同源移植,
> 常规长度下约慢 5~20%; 我库封装开销 ~0–4%; 分解结果逐位级一致**。
> 3GB/超大噪声实验已推迟并记录: 见 `POSTPONED_3GB.md`。
>
> 📦 **数据归档说明 (2026-09-10)**: 本次清理把本报告引用的原始数据移动到
> `results/_legacy_pyemd_wrapper/` —— 下文出现的 `results/summary_timing.md`、
> `results/timing_median.csv`、`results/summary_memory.md`、`results/memory/*`、
> `results/memory_after_opt/*` 等路径，请统一加该前缀解读（如
> `results/_legacy_pyemd_wrapper/summary_timing.md`）。计时与内存实验均已由
> 现行原生引擎重跑（报告见 `docs/EMD_Timing_Memory_Quality_Report.md` 与
> `docs/EMD_Large_Signal_Memory_Report.md`）；本报告历史结论仅适用于包装版时代。

---

## ⚠️ 2026-09 之后更新: 库 EMD 已转正为原生实现

本报告 `MD-EMD` 列及其结论写于 **PyEMD 包装版时代** (MD-EMD 内部即 PyEMD
sifter, "同引擎、隔离封装开销"的对照设计)。此后库内 `EMD` 已由自研原生
筛分实现转正 (原 `EMD_new`; 默认 linear 包络 + sd_thr=0.3, 参数扫描选定)。
三者如今是**真正互相独立的引擎**。以"当前优化后的最佳配置 (默认)"展开的
新三方对照 (PyEMD / PySDKit / 库 EMD 默认 + 同包络档) 见
`docs/EMD_vs_EMD_new_Performance_Report.md` (及参数扫描
`docs/EMD_new_Parameter_Sweep_Report.md`); 本报告历史结论仅适用于包装版时代。

---

## 1. 被测对象

| 列 | 是什么 | 引擎 |
|---|---|---|
| `MD-EMD` | 我的库 `Modal_Decomposition.Class.EMD.decompose` | **2026-09 前**: 薄封装 PyEMD sifter (`src/Modal_Decomposition/EMD.py`); **现**: 原生筛分实现 (原 `EMD_new`, 见上方更新块与 docs 报告) |
| `PyEMD` | `PyEMD.EMD.emd` 直用 | EMD-signal 1.9.0 随包提供的 `PyEMD` legacy 别名 —— **包装版时代与 MD-EMD 内部同一引擎**, 作为隔离"封装开销"的基线 |
| `PySDKit` | `pysdkit.EMD.fit_transform` (v0.5.0, `ref/pysdkit`) | 独立移植 PyEMD sifter: 极值查找/边界镜像/样条辅助全部自行重写 (`_find_extrema.py`、`_prepare_points.py`、`_splines.py`) |

> *(历史实验设计 —— 描述包装版时代的运行口径, 现引擎关系见上方更新块与
> `docs/EMD_vs_EMD_new_Performance_Report.md`)*

三者均以**默认参数**运行 (spline=cubic, nbsym=2, 停判阈值一致:
std=0.2, svar=1e-3, energy=0.2, range=1e-3, total_power=5e-3, 迭代上限 1000 ——
已在源码逐项核对, PyEMD/EMD.py 与 pysdkit/_emd/emd.py 两侧一致)。

### 引擎同源性 (为什么两边性能天然接近)

对照 `ref/pysdkit/_emd/emd.py` 与 `.venv/.../PyEMD/EMD.py`:

- 筛分主循环结构逐行同源 (每轮 sift: 极值扫描 → 上/下包络样条 → 去均值 →
  默认停判分支再扫一次极值 + Cauchy 类收敛判据; 每 sift 实际调 `find_extrema` 3 次);
- 样条均调用 scipy 插值器 (Akima1DInterpolator / CubicSpline / PchipInterpolator
  的同一包装, 3 点立方为同一份 `cubic_spline_3pts` 代码);
- 时间轴归一化算法相同 (`(t-t0)/min(dt)`), `extrema_detection='simple'` 时都忽略传入 T;
- 差异仅是组织方式与平台化细节 (pysdkit 把 plateau 处理写成 Python 循环、
  prepare_points 在时间坐标上做镜像等), 单次 sift 的浮点计算量基本不变。

→ 因此预期: 单次分解耗时、内存剖面几乎相同, 差异来自常数因子。

## 2. 方法

**常规长度轴 (计时/质量)**: 确定性合成信号 @1000 Hz, 三种输入
(见 `signals.py`):

- A: 37 Hz + 113 Hz 双纯音 + 弱噪声(σ=0.1);
- B: AM-FM 扫频 + 89 Hz 纯音 + 二次趋势(无噪声);
- C: 纯白噪声(筛分压力测试)。

长度 n ∈ {256, 1024, 4096, 16384, 65536}, 每格 3 次重复取中位数,
三实现轮转顺序以抵消漂移, 预热与 import 不计时 (`bench_timing.py`)。
质量指标: 重构误差 max|S−Σ|、正交性指数 IO、对已知模式的恢复相关/误差 (`quality.py`)。

**大数据内存轴**: 数据尺寸 {1MB 20MB 100MB 500MB 1GB} × 模式
{单调递增, 随机噪声}, float64; 每格独立子进程, 输入建成后才开始计
20 s 分解预算, 期间 50 ms 心跳采样 RSS (`bench_memory.py` + `case_worker.py`, 
机制沿用 `tests/test_memory`)。**3GB 全部格子与 ≥1GB 随机噪声按计划推迟/记录**
(本机 16GB RAM, 先例见 `tests/test_memory` EMD.csv: 3GB 格全部 timeout)。

**环境**: 24 核/32 线程, 15.6GB RAM, Windows 10 (26200), 本仓库 venv
Python 3.10.11 · numpy 2.2.6 · scipy 1.15.3 · EMD-signal(=PyEMD 别名) 1.9.0 ·
pysdkit 0.5.0 · Modal_Decomposition src@c20c694。机器负载会导致 ±5~10% 单次波动;
计时结论均基于中位数/区间表述。

## 3. 计时结果 (中位数, 秒)

完整表: `results/summary_timing.md`、`results/timing_median.csv`。
比值 >1 = 比 MD-EMD 慢。

| case | n | MD-EMD | PyEMD | PySDKit | PyEMD/MD | PySDKit/MD |
|------|---|-------:|------:|--------:|---------:|-----------:|
| A | 256 | 0.0047 | 0.0039 | 0.0043 | 0.82x | 0.91x |
| A | 1024 | 0.0126 | 0.0130 | 0.0152 | 1.03x | 1.21x |
| A | 4096 | 0.0567 | 0.0590 | 0.0642 | 1.04x | 1.13x |
| A | 16384 | 0.4275 | 0.4625 | 0.4206 | 1.08x | 0.98x |
| A | 65536 | 5.778 | 5.384 | 5.826 | 0.93x | 1.01x |
| B | 256 | 0.0016 | 0.0019 | 0.0021 | 1.19x | 1.30x |
| B | 1024 | 0.0089 | 0.0050 | 0.0049 | 0.56x | 0.55x |
| B | 4096 | 0.0112 | 0.0131 | 0.0126 | 1.18x | 1.13x |
| B | 16384 | 0.0298 | 0.0302 | 0.0373 | 1.01x | 1.25x |
| B | 65536 | 0.207 | 0.187 | 0.215 | 0.90x | 1.04x |
| C | 256 | 0.0048 | 0.0059 | 0.0085 | 1.23x | 1.76x |
| C | 1024 | 0.0097 | 0.0087 | 0.0120 | 0.89x | 1.23x |
| C | 4096 | 0.0448 | 0.0478 | 0.0552 | 1.07x | 1.23x |
| C | 16384 | 0.2233 | 0.2260 | 0.2515 | 1.01x | 1.13x |
| C | 65536 | 2.174 | 2.244 | 2.402 | 1.03x | 1.10x |

统计(仅 n ≥ 1024): PyEMD/MD 中位 ≈ **0.96–1.03x** (封装开销 ≈ 0–4%,
小 n 上 ms 级校验/构造费, 见 §5); PySDKit/MD 中位 ≈ **1.07x (A), 1.08x (B),
1.18x (C)**, 即 pysdkit 常规长度下**慢约 8–18%**、最重噪声压力(C)差距最大;
n=256 的极端比值是亚毫秒计时噪声, 无意义。
缩放: 噪声类输入约 n^1.5–n^1.8 超线性 (IMF 数与每阶筛分轮数随长度增长)。
图: `figs/timing_wall_case{A,B,C}.png`、`figs/timing_ratio_caseA.png`。

## 4. 分解质量

三种实现**数值上几乎完全一致**: 所有格子的重构误差 ~1e-19 (机器精度),
对已知模式的恢复相关系数三个实现完全相同 (逐位一致到打印精度), IO 相同。
唯一分歧: case A @ n=65536, PyEMD/MD 出 15 行、pysdkit 14 行 —— 低能量尾段
最后两个边界 IMF 的合并/拆分不同 (对应行相关 0.94~0.999), 高能 IMF 仍一致;
能量 ~1e-4 量级以下的尾部分裂对 EMD 边界处理的小数值差异敏感, 属正常范围,
不影响结论: **分解质量无实质差别**。
(注: 大噪声上 pysdkit 一阶 IMF 前与 PyEMD 逐行一致到 ~1e-5 相对误差, 见下。)

## 5. 内存/吞吐 (大数据轴)

完整表: `results/summary_memory.md`、`results/memory/<impl>.{json,csv}`。

**单调递增 (可完成格, peak RSS 为进程绝对峰值):**

| size | MD-EMD wall / peak | PyEMD wall / peak | PySDKit wall / peak |
|------|-------------------:|------------------:|--------------------:|
| 1MB | ~0 / 113 MB | ~0 / 113 MB | ~0 / 119 MB |
| 20MB | 0.13 s / 274 MB | 0.13 s / 235 MB | 0.11 s / 251 MB |
| 100MB | 0.73 s / 992 MB | 0.58 s / 817 MB | 0.59 s / 818 MB |
| 500MB | 4.31 s / 4576 MB | 3.39 s / 4100 MB | 3.84 s / 4058 MB |
| 1GB | 10.8 s / 8119 MB | 9.0 s / 8290 MB | 9.8 s / 8160 MB |

- 三实现驻留峰值基本持平 (~8–9× 输入: 引擎每轮持有多份全尺寸数组 + memmap 页入),
  500MB/1GB 的墙钟差 (~1–2 s) 处于单次测量 + 页面回收噪声范围;
- **我库的一个可优化点(已修复, 见 §8)**: `Utils/Check.py` 的 `Check_Time_and_Signal` 在
  T=None 时总是分配 `arange(N, float64)` (≈+1×输入的一次性瞬时内存,
  100–500MB 上观察到 MD-EMD 峰值比 PyEMD 高 ~0.1–0.5GB), 而 PyEMD sifter
  在默认 `extrema_detection='simple'` 下**忽略传入的 T** 自行重建时间轴 ——
  该默认 T 纯属浪费。

**随机噪声 (筛分压力):** 1MB (n=131072) 三家都 ~10.7–11.4 s 完成, 17 个 IMF;
≥20MB 在 20 s 预算内全部 timeout, 心跳轨迹显示三实现 RSS 增长速率几乎重合
(kill 时峰值: 20MB ≈ 0.61–0.63GB, 100MB ≈ 2.57–2.67GB, 500MB ≈ 7.8–8.6GB,
均远未完成第一/二阶 IMF) —— **内存增长剖面无差别, pysdkit 无更优的大数据路径**。
随机 500MB 已贴近 16GB 机器换页边界; 随机 ≥1GB 与全部 3GB 格子推迟,
理由/恢复步骤: `POSTPONED_3GB.md`。

## 6. 结论

1. **引擎层面**: 两边同源于 PyEMD sifter, 单次筛分计算量几乎相同;
   pysdkit 未做速度优化 (其卖点是"全家桶集成 + 可视化", 不是引擎性能)。
2. **速度**: 常规长度下 MD-EMD ≈ PyEMD (+0–4% 封装费), pysdkit 慢 ~5–20%
   (压力越大越明显); 无任何长度上 pysdkit 系统性更快。
3. **内存**: 三者几乎一致 (~8–9× 输入驻留); 我库原本唯一的额外内存是
   T=None 时默认 `arange(N)` 的 ~1× 输入瞬时分配 —— **已修复** (见 §8,
   复测 100MB/500MB/1GB 峰值分别降 ~0.1–0.2GB / ~0.5–1.5GB / ~1.5GB)。
4. **质量**: 逐位级一致, 尾部边界 IMF 偶尔 ±1 行差异, 无实质差别。
5. **3GB 实验**: 按指示推迟, 已记录计划与恢复命令 → `POSTPONED_3GB.md`。

## 7. 复现

```powershell
# 常规长度计时+质量 (完整网格约 5 分钟)
.venv\Scripts\python.exe tests\comparison\bench_timing.py
# 大数据内存网格 (1MB..1GB; 3GB/≥1GB随机默认记录为 deferred)
.venv\Scripts\python.exe tests\comparison\bench_memory.py
# 汇总表格与图
.venv\Scripts\python.exe tests\comparison\summarize.py
.venv\Scripts\python.exe tests\comparison\bench_plot.py
```

## 8. 基于本报告的优化 (已实施, 以 EMD 为代表)

按用户指示范围收敛: **只改 EMD 一个代表方法 + 公共校验工具**, 其余算法
(VMD/SVMD/EWT/LMD/SSA/CEEFD/EFD/FMD/EEMD/CEEMD/CEEMDAN/ICEEMDAN 等)
保持原样, 留待其算法自身的优化轮次。

改动 (共 2 个文件):

1. `src/Modal_Decomposition/Utils/Check.py`
   `Check_Time_and_Signal` 新增 `default_T: bool = True` 参数: 为 False 且
   T=None 时不再分配 `arange(N, float64)`, 返回 `T=None`。默认值不变,
   对所有现有调用者零影响 (`tests/test_utils.py` 全绿)。
2. `src/Modal_Decomposition/EMD.py`
   - 调用 `Check_Time_and_Signal(..., default_T=False)`: PyEMD sifter 会自行
     重建时间轴且忽略传入 T, 省掉 ≈1× 输入的一次性分配;
   - 修复潜在崩溃: 原 `Res = arr[-1, :]` 在 ndim 检查前执行, 1-D 后端输出会
     直接 IndexError; 现先判 `arr.ndim` 再切片, 并补齐 0-D 防御分支。

验证:

* 冒烟: `default_T=False` 返回 T=None; EMD 传/不传 T 分解结果逐位一致
  (重构误差 ~8.7e-19); 编译检查通过。
* 测试: `test_utils + test_reconstruction + test_contract` 159 passed;
  `test_facade` 18 passed; `test_registry/test_seed` 75 passed
  (注: `test_seed` 与 `test_contract` 同一会话连跑会因 conftest 双模块实例的
  全局种子泄漏而失败 —— 测试基础设施的既有问题, 与本次改动无关)。

大数据内存复测 (MD-EMD, increasing, 两次独立运行, 单机噪声 ±10%):

| size | 优化前 peak RSS | 优化后 peak RSS | 降幅 |
|------|----------------:|----------------:|-----:|
| 100MB | 992 MB | 813~870 MB | ≈ -0.1~0.2 GB |
| 500MB | 4576 MB | 3087~4062 MB | ≈ -0.5~1.5 GB |
| 1GB | 8119 MB | 6572~6628 MB | ≈ -1.5 GB |

原始数据: 优化前 `results/memory/MD-EMD.json`(去重后 10 条),
优化后 `results/memory_after_opt/MD-EMD.json`。墙钟方向为负或持平
(受机器页面回收噪声主导, 不单列为结论)。
