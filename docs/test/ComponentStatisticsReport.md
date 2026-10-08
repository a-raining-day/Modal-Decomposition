# 组件统计报告 —— Modal_Decomposition 分解方法内部组件复用分析

- 日期:2026-09-05
- 范围:`src/Modal_Decomposition/` 全部 **15 个注册方法**(CEEFD、CEEMD、CEEMDAN、EEMD、EFD、EMD、EWT、FMD、ICEEMDAN、LMD、MEMD、RPSEMD、SSA、SVMD、VMD)+ 内部模块 `_SVMD_numba.py`
- 统计方式:四批并行全文精读(每文件逐行,无跳过)+ 注册表/惰性导入/自定义 helper 的一手 grep 核对
- 数据口径:后端形态、行数、外部依赖(含函数体内惰性导入)、decompose 结构、受控组件类别使用、自定义辅助函数、收敛/停止判据、重复实现模式

---

## 1. 总览(数字)

| 指标 | 数值 |
|---|---|
| 注册方法 | 15 + 内部 `_SVMD_numba` |
| 方法文件总行数 | ~2 695 行(方法文件 2 678 + 占位 17) |
| 方法平均行数 | ~180(薄包装 ~113;构建于本库 EMD 上 ~183;纯原生 ~223) |
| 依赖外部分解库的方法 | 5 直接 + 3 传递 = **8/15**;**PyEMD 系影响 6/15**(EMD、EEMD、CEEMDAN + 链式 CEEMD、RPSEMD、ICEEMDAN) |
| 随机源 | 5/15 走 seed 处理(FMD 仅影响候选初始化;SVMD/SSA/谱类确定性) |

## 2. 实现形态分类

| 形态 | 方法(行数) | 说明 |
|---|---|---|
| 薄外部包装 | EMD(90)、EEMD(117)、CEEMDAN(150)、VMD(99)、EWT(110)|透传 + 末行切残差 + 装箱,自定义 helper 全为零|
| 构建于本库 EMD(传递依赖 PyEMD)|CEEMD(149)、RPSEMD(147)、ICEEMDAN(253)|原生编排(噪声/相位集成),底层筛分在 PyEMD|
| 纯原生(scipy/numpy)|LMD(228)、MEMD(285)、FMD(411)、SVMD-numpy(209)、SSA(133)、EFD(137)、CEEFD(160)|无外部分解库;SSA 零 scipy|
| 占位|`_SVMD_numba`(17)|无 `@njit`、无 numba 导入,恒 `raise RealizationError`;**库内实际 njit 使用量为 0**;SVMD `backend="numba"` 分派已接线但不可运行|

## 3. 组件 × 方法使用矩阵(受控类别,本地代码)

| 组件类别 | 使用的方法(本地实现) | 统计 / 备注 |
|---|---|---|
| 极值检测 | LMD、MEMD(筛分采样);CEEMD、FMD(仅停止);EFD、CEEFD(谱峰)|6/15;API 分裂:`argrelextrema`(LMD/CEEMD/FMD/EFD)vs `find_peaks`(MEMD/CEEFD)|
| 端点镜像 | 仅 LMD `_mirror_extend_real`(偶反射);MEMD 端点锚定;EMD 系经 PyEMD `nbsym`|本地 1;三种端点哲学|
| 包络插值(时域)|LMD(CubicSpline-natural)、MEMD(interp1d,默认 linear);均带 try 样条 → `np.interp` 回退|本地 2(回退样板 1+2 处重复);PyEMD 内另有 3|
| 包络均值/局部均值|LMD(相邻极值中点);MEMD(投影 × k 方向 `(max+min)/2`)|本地 2,思路不同|
| 筛分停止判据|LMD(包络平坦度)、MEMD(SD<0.2)、CEEMD(单调∨极值<3)、RPSEMD(四条件)、ICEEMDAN(能量+单调)|**本地 5 套各写各的**|
| 单调残差检查(本库 Utils)|`is_monotonic`:LMD、RPSEMD、ICEEMDAN;`monotonic`:MEMD(逐通道)、CEEMD|5/15;接口命名分裂,同语义两种入口|
| 噪声/相位集成|EEMD、CEEMDAN(PyEMD);CEEMD(±成对白噪)、RPSEMD(相移正弦,确定性)、ICEEMDAN(预分解噪声池 + β 标度)|5,本地 3 套|
| "EMD(max_imf=1) 取首 IMF" 模式|CEEMD、RPSEMD、ICEEMDAN|3 处同一模式|
| FFT/谱 | SVMD(ADMM 谱域)、RPSEMD(rfft 主频)、FMD(谱熵/周期)、EFD、CEEFD|本地 5;EFD/CEEFD 共享"频域布尔掩码分段重建"内核(谷界 vs 峰心包络)|
| Hilbert/瞬时属性|仅 FMD(SNR 与周期估计)|1/15|
| 滤波(savgol/平滑)|LMD(包络幅度)、FMD(SNR)、CEEFD(boxcar 谱包络)|3|
| 优化/分解|FMD(eigh 广义特征问题 + eig 回退)、SVMD(ADMM)、SSA(svd)|本地 3|
| seed 处理|EEMD、CEEMDAN(注入 PyEMD);CEEMD、ICEEMDAN(rng 自管);FMD(滤波候选)|5/15|
| 多通道|仅 MEMD((d, N) Hammersley 方向投影联合分解)|1/15|

## 4. 重复实现清单(去重候选)

| 重复模式 | 位置 | 去重目标 |
|---|---|---|
| 插值失败 → `np.interp` 回退样板 | LMD 1 处 + MEMD 2 处 | `_safe_interp(x, y, xi, kind)` |
| "单调 或 稀疏极值 停止" 残差判停 | LMD、MEMD、CEEMD(+ RPSEMD/ICEEMDAN 单调部分)|`residual_is_done()` 公共停止助手|
| 极值双 API 同语义 | 6 文件 | 统一 `count_extrema` |
| `log2(N)` max_imf 启发式系数不一致 | CEEMD `log2+2`、RPSEMD `log2`、ICEEMDAN `log2+5`、LMD/MEMD `log2` | 统一常量/规则 |
| 末行=残差切分 + reshape 兜底 | EMD/EEMD/CEEMDAN | 包装层公共代码 |
| Hankel 嵌入视图血缘 | FMD(sliding_window_view)/SSA(as_strided),互不复用 | 可选共享 |
| 变分频域族 | SVMD 维纳分母 `1+α(f−ω)²` 与 VMD/EWT 同族 | 谱族组件 |
| 残差语义三种约定 | VMD/SSA `Res=None`;EWT 末行趋势;EFD/CEEFD `S−Σ`(EFD 加回 MEAN)|全库规范;跨方法消费 `result.Res` 需区分|

## 5. 本库 Utils 复用统计(15 方法)

| Utils 组件 | 使用数 | 使用者 |
|---|---|---|
| `Check_Time_and_Signal` | **15/15**(唯一全复用件)|全部|
| `is_monotonic` / `monotonic` | 5/15 | LMD、MEMD、CEEMD、RPSEMD、ICEEMDAN(全集中在原生实现方法;包装方法由外部库内部判据替代)|
| `resolve_seed` | 5/15 | EEMD、CEEMDAN、CEEMD、ICEEMDAN、FMD |
| `is_uniform` | 1/15 | ICEEMDAN(强前置校验)|

## 6. 关键结论

1. **PyEMD 是真正的"根依赖"**:6/15 方法(含 3 个原生编排方法)传递依赖它;实测其单次导入 ~1.1–1.4s,筛分核心 ~75% 在包络样条 —— 先原生实现 EMD 并配套共享的"包络插值 + 筛分引擎"组件,一处加速辐射 6 个方法;
2. **7 个方法已是原生**,组件统计的现实对象明确:极值检测(6 文件)、停止判据(5 套)、单调检查(5 文件)、插值回退(3 处)、首 IMF 集成(3 处)五组最该先抽公共组件;
3. **numba 现状为零使用**(`_SVMD_numba` 是占位)——"可选 njit"不是加选项,而是补实现;从统计看,EEMD/CEEMDAN 的 ensemble 循环与 LMD 的纯调频迭代是真正适合 njit 的热点形态;
4. **谱类与筛分类组件重叠接近零**(除基础设施与单调判停;EFD/CEEFD 内核同构、VMD/SVMD 变分同族)——原生化与编译优化应分两族推进,而非统一内核;
5. **FMD(411 行)已自带完整信号处理工具链**(近似熵/SNR/周期估计/滤波器初始化),可作为"自研算法工程化"的参考模板。

## 附录 A:PyEMD-EMD 热点剖析证据(65536 随机点,max_imf=3)

- 总耗时 2.675s,其中 **~1.1–1.4s(≈40%)为导入开销**(PyEMD 拉入 matplotlib/pylab);
- 筛分核心 ~1.07s,**~75% 花在包络样条路径**(`spline_points`/`cubic` 0.81s,含每次构造 `CubicSpline` 对象);
- 外推:EMD 随机数据 131k 点 ≈ 8.4s、262k ≈ 31s(超线性);内存管道实测 1GB f64 递增输入 EMD 9.3s 完成、峰值 ~8.2GB;100MB f16 递增输入峰值 ~3.7GB(f16→f64 转换 + 工作数组)。

## 附录 B:逐文件速查表

| 文件 | 行数 | 形态 | 外部依赖(惰性) | 自定义 helper |
|---|---|---|---|---|
| EMD.py | 90 | 薄包装 | PyEMD | — |
| EEMD.py | 117 | 薄包装 | PyEMD(EEMD + 内嵌 EMD) | — |
| CEEMDAN.py | 150 | 薄包装 | PyEMD(CEEMDAN) | — |
| CEEMD.py | 149 | 建于本库 EMD | scipy argrelextrema | — |
| RPSEMD.py | 147 | 建于本库 EMD | numpy rfft | — |
| ICEEMDAN.py | 253 | 建于本库 EMD | logging;强校验(is_uniform) | — |
| LMD.py | 228 | 原生 | scipy argrelextrema/savgol/interpolate | `_mirror_extend_real`、`_safe_interpolate`、`_check_convergence` |
| MEMD.py | 285 | 原生(多通道) | scipy find_peaks/interp1d | `_generate_hammersley_points`、`_radical_inverse(_vdc)`、`_generate_primes`、`_compute_local_mean`、`_should_stop` |
| FMD.py | 411 | 原生 | scipy argrelextrema/eigh/eig/hilbert/savgol/find_peaks/svd | `_auto_estimate_num_candidates`、`_estimate_snr`、`_estimate_signal_complexity`、`_robust_period_estimation`、`_advanced_filter_init`、`_bounded_convolution` |
| SVMD.py | 209 | 原生 | scipy.fft;numba 分支分派 | —(实例缓存槽位 `_freqs` 等) |
| SSA.py | 133 | 原生(零 scipy) | — | `hankel`、`diagonal_average_fast` |
| EFD.py | 137 | 原生 | scipy argrelmax | —(全部内联) |
| CEEFD.py | 160 | 原生 | scipy find_peaks/windows | `_compute_spectral_envelope`、`_extract_imf` |
| VMD.py | 99 | 薄包装 | vmdpy | — |
| EWT.py | 110 | 薄包装 | ewtpy | — |
| _SVMD_numba.py | 17 | 占位 | — | `numba_svmd`(恒抛 RealizationError) |

## 附录 C:分析执行方法

- 四批并行分析员逐行精读:① EMD/EEMD/CEEMDAN/CEEMD/RPSEMD/ICEEMDAN;② LMD/MEMD;③ FMD/SVMD/_SVMD_numba/SSA;④ VMD/EWT/EFD/CEEFD;
- 一手核对:`git` 注册表(`Modal_Decomposition.Class` 15 项)、每方法构造参数数与 decompose 内惰性导入清单、helper 函数清单 grep、PyEMD cProfile 热点剖析;
- 本次统计仅记录,未改动任何源码。
