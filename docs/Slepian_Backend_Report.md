# Slepian (DPSS) 后端报告 —— numpy / scipy / C 三后端

- 日期: 2026-09-15
- 范围: `Utils/Slepian.py`(对外入口) + `Utils/_Slepian/`(numpy 后端 / C 后端 / ctypes 封装) + `setup.py` 构建接线
- 复现: `python tests/comparison/bench_slepian.py`(全量) / `--quick`(快速); 正确性测试 `pytest tests/test_slepian.py -q`(255 项)
- 数据口径: 稳态耗时取 best-of-3; 峰值内存用 `tracemalloc`; 冷启动为整进程 wall time; 参考实现为 `scipy.signal.windows.dpss`

---

## 0. 结论 (TL;DR)

1. **三后端数值一致**: 与 scipy 的偏差, 序列逐元素 ≤ 1.5e-12、集中比 ≤ 6.4e-15 (K ≤ N/2 的实用范围, 数百个组合); 大 N 下**相对**精度仍到机器精度。
2. **默认后端 = `scipy`**(见 §6 决策): 稳态最快(LAPACK `dstevr`)、峰值内存最低、零构建负担。
3. **C 后端**(自研, ctypes + 预分配缓冲) 的价值在别处: **冷启动最快**(省掉 `import scipy.signal`, N=1e5 时 401ms vs 630ms)、调用路径不依赖 scipy、内存/时间均为 O(N) 且可自我复现数值; 稳态比 scipy 慢约 2×。
4. **numpy 后端** 无需 scipy 但受稠密分解限制: 由 `SLEPIAN_NUMPY_MAX_BYTES=256MB` 守卫, 适用 `N ≤ 11584`; 相对"直接构造 N×N 稠密矩阵"的朴素实现快 1.5–3×、省内存 2.4×。

---

## 1. 文件与接口

| 位置 | 作用 |
|---|---|
| `Utils/Slepian.py` | **对外入口** `slepian(N, halfBW, nTapers=None, mod=None, return_ratios=False, norm=None, sym=True)`; 后端分发 + `auto` + 默认路径降级 |
| `Utils/_Slepian/numpy_slepian.py` | 纯 NumPy 后端 `generate_slepian_numpy(...)` |
| `Utils/_Slepian/C.py` | **ctypes 封装** `generate_slepian_c(...)` + `available()` / `library_path()` / `load_library()` |
| `Utils/_Slepian/_C/slepian_core.c` / `slepian_dpss.h` | **C 核心**: 三对角 + 对称折半 + 二分/反迭代 + 集中比 FFT |
| `setup.py` | 扩展 `Modal_Decomposition.Utils._Slepian._C._slepian_native` (MSVC `/O2 /utf-8`, GCC `-O3 -std=c99`); 未编译时自动降级 numpy |
| `Base/ConstDefine.py` | `SLEPIAN_BACKEND_LIST` / `SLEPIAN_BACKEND` / `SLEPIAN_NUMPY_MAX_BYTES` |
| `Utils.get_slepian()` | 惰性 import + 进程级 import 缓存注册(键 `Modal_Decomposition.Utils.Slepian`) |

统一的返回契约: 始终 `(nTapers, N) float64`(行 = 阶数, 集中比降序), `return_ratios=True` 时同时返回集中比; `nTapers=None` → `max(1, floor(2·NW−1))` 并截断到 `solve_n`(scipy 在 `Kmax=None` 时返回单条一维窗的行为已在本层规范化)。

---

## 2. 算法 (三后端同一套数学)

DPSS 是**对称三对角**矩阵的特征向量 (Slepian 1978; Percival & Walden 1993 eq. 380):

```
d_i = ((N-1-2i)/2)² · cos(2πW),   e_i = i(N-i)/2,   W = NW/N
```

- **中心对称折半**(本次新增的优化): 该矩阵满足 `d_i = d_{N-1-i}`、`e_i = e_{N-i}`, 故特征向量有确定奇偶性, 问题解耦为两个**半尺寸**问题。约化方程本身非对称(奇数 N 的对称支折回项落在下三角), 经 `u → G^{-1/2}u`(`G_j = 2`, 唯自映射点取 1) 相似变换后与对称三对角同谱:
  `B[j][j] = d_j + fold_j`, `B[j][j+1] = e_{j+1}·√(w_j/w_{j+1})`; N 偶时 `fold = ±e_{N/2}` 落在末主对角, N 奇时对称支的折回被权重完全吸收(该结构恒有 `e_m = e_{m+1}`)。
- **numpy 后端**: 折半后对两个半尺寸矩阵做稠密 `eigh`(时间 ≈ O(N³/4), 内存 ≈ O(N²/2))。
- **C 后端**: 不构造稠密矩阵 —— Gershgorin 区间上用 **Sturm 计数二分**定位前若干阶特征值, 再用 **反迭代**(Thomas 解三对角, 移位加扰动、主元相对下限、失败自动加大扰动重试、Rayleigh 商验收 + 商迭代精化)求向量; 时间 O(k·N·log N) 量级、内存 O(k·N)。
- **集中比**: 三后端同为**自相关法**(Percival & Walden pg 390): `l1 = Σ_j r_j·autocorr_j`, `r_j = 4W·sinc(2W·j)`, `r_0 = 2W`; C 端用自实现的 radix-2 FFT。
- **符号约定**: 同 scipy/P&W pg379 —— 偶阶(对称支)序列和为正, 奇阶(反对称支)首个显著瓣为正。

---

## 3. 正确性

`tests/comparison/bench_slepian.py --quick` (N ∈ {256,1024,4096}, NW ∈ {1,2,4}, K ∈ {1,3,5}, 27 组):

| 后端 | 组合数 | 序列 max\|Δ\| | 集中比 max\|Δ\| |
|---|---:|---:|---:|
| numpy | 27 | 1.51e-12 | 3.00e-15 |
| C | 27 | 8.23e-13 | 6.44e-15 |
| scipy | 27 | 0 | 0 (基准) |

更广的扫描(测试与开发期实测):

| 场景 | 结论 |
|---|---|
| C vs scipy, 465 组 (N ≤ 1024, K ≤ N/2, NW ∈ 0.2…8) | 序列 ≤ 6.0e-13, 集中比 ≤ 4.1e-15 |
| numpy vs scipy, 90 组 (N ≤ 1024) | 序列 ≤ 2.1e-13, 集中比 ≤ 2.7e-15 |
| `norm="approximate"` / `sym=False` | 三后端一致 ≤ 1.5e-14 |
| 大 N (C vs scipy, K=5, NW=4) | N=16384 → 1.9e-12; 65536 → 1.6e-11; 262144 → 2.2e-10; 1048576 → 1.0e-09 |
| 大 N 的**相对**精度 | 特征方程残差 ‖Tv−λv‖ / ‖T‖ ≈ 3e-16(N=1e6), 即机器精度(‖T‖ ~ (N/2)² ≈ 2.6e11) |

---

## 4. 性能与峰值内存 (NW=4.0, K=5, 不含集中比, best-of-3)

| N | scipy ms | numpy ms | C ms | scipy MB | numpy MB | C MB |
|---:|---:|---:|---:|---:|---:|---:|
| 1 024 | 1.33 | 127.4 | 1.64 | 0.1 | 10.0 | 0.1 |
| 2 048 | 2.17 | 459.7 | 3.37 | 0.2 | 40.1 | 0.3 |
| 4 096 | 4.16 | 2 105.7 | 7.28 | 0.5 | 160.2 | 0.6 |
| 8 192 | 8.49 | 12 970.8 | 14.98 | 1.0 | 640.3 | 1.1 |
| 16 384 | 22.7 | — | 34.0 | 1.9 | — | 2.3 |
| 65 536 | 78.8 | — | 127.5 | 7.8 | — | 9.0 |
| 262 144 | 290.0 | — | 510.6 | 31.0 | — | 36.0 |
| 1 048 576 | 1 113.0 | — | 2 144.6 | 124.0 | — | 144.0 |
| 4 194 304 | 4 566.1 | — | 9 276.6 | 496.0 | — | 576.0 |

要点:

- **scipy 稳态最快**(LAPACK `dstevr` 的分段二分/反迭代经过高度优化), C 约为其 1.4–2× 耗时;
- **numpy 后端随 N 急剧劣化**(稠密 `eigh`: N=8192 时 13 s / 640 MB), 因此设 `SLEPIAN_NUMPY_MAX_BYTES = 256MB` → **N ≤ 11584** 的硬守卫, 超限直接报错并提示改用 `"C"`/`"scipy"`(绝不静默 GB 级分配);
- numpy 折半优化的实测收益(对同文件内的朴素全长稠密实现): N=256 → 2.3× / 内存 2.5×; N=2048 → 1.5× / 内存 2.4×;
- C 峰值内存比 scipy 略高(+16% @N=1e6): 工作区含 `2·(k+2)·⌈N/2⌉` 的候选向量缓存(`k=5` 时 ≈ 7·N/2 个 double)与集中比 FFT 缓冲; 关闭 `return_ratios` 可省掉 FFT 部分。**这是可优化项**(见 §7)。

---

## 5. 冷启动 (N=100 000, 整进程 wall ms, 含解释器启动)

| 后端 | 首次 | 第二次 | 说明 |
|---|---:|---:|---|
| C | 401 | 387 | ctypes 装载共享库(~0 ms), 无 scipy import |
| scipy | 630 | 611 | 首次需 `import scipy.signal`(≈230 ms) |

N=8192 的快速版: C 168/157 ms vs scipy 491/483 ms。冷启动差值即为 `scipy.signal` 的导入成本。

---

## 6. 默认后端的选择

`Base/ConstDefine.SLEPIAN_BACKEND` 现为 **`"scipy"`**, 依据:

1. **稳态速度**: scipy 比自研 C 快约 2×, 比 numpy 快 2~3 个数量级(见表 §4);
2. **峰值内存**: scipy 124 MB < C 144 MB(N=1e6);
3. **零构建**: 不需要编译器/预编译产物, 三平台一致;
4. **冷启动优势在库内被抵消**: EWT 等调用路径本就会导入 scipy(`Utils.Peaks`/`Utils.Spline`), 因此 C 的"省掉 scipy 导入"收益在实际使用中通常不成立。

C 后端仍然保留且推荐用于: 不希望在调用路径上依赖 scipy、需要自研实现的可复现数值、或一次性调用的冷启动敏感场景。切换方式(任选其一):

```python
from Modal_Decomposition.Utils.Slepian import slepian
slepian(65536, 4.0, 5, mod="C")          # 单次显式指定
slepian(65536, 4.0, 5, mod="auto")       # 大 N(> 11584)且 C 可用时自动走 C
# 或改 Base/ConstDefine.SLEPIAN_BACKEND = "C" 作为全局默认
```

`mod=None`(默认路径)在后端不可用时**降级 numpy 并发一次 `UserWarning`**; 显式指定不可用的后端则报错(`RealizationError`/`ImportError`, 含编译/安装提示)—— 与 `Utils.FFT` 的后端约定一致。

---

## 7. 边界与限制(实测)

| 情形 | 行为 |
|---|---|
| `N=1, 2` | numpy/C 正常返回(`N=1` 序列为 `[1]`); **scipy 自身会失败** |
| **`NW ≥ N/2`(默认参数下 `N ≤ 6`)** | scipy 必然抛错 ⇒ **自动改用 `SLEPIAN_SMALL_N_ORDER` 的首个可用后端**(默认 "C")并发 `UserWarning`; 即使显式 `mod="scipy"` 也如此(scipy 必错, 换比报错有用)。实测小 N 单次开销 C 4.5–6.7 µs/2.3–3.2 KB vs numpy 38–67 µs/7.1–7.6 KB, 故 C 排首位。切换后数值与 numpy 后端一致 |
| `K` 接近 `N/2` 或 `= N` | 低阶特征值数值简并(集中比≈0), 各实现的**排序/符号**可不同; 实测相关矩阵为**完美置换**(最小匹配相关 = 1.000000)、集中比差 ≤ 4.4e-16 ⇒ 返回的序列**集合**完全一致(测试 `test_full_basis_matches_scipy_as_set` 固定该性质) |
| `norm="subsample"` | 仅 scipy 支持; numpy/C 明确报错并指向 `mod="scipy"` |
| numpy 后端 `N > 11584` | `ValueError`, 提示改用 `"C"`/`"scipy"`(内存守卫) |
| C 未编译 | 默认路径降级 numpy + `UserWarning`; `mod="C"` 报 `RealizationError` |

**已知可优化项**(未做, 留待需要时):

1. C 的二分定位目前对每个候选都从完整 Gershgorin 区间重开(≈45 次 Sturm 扫描/候选), 占其总耗时约 60%; 用"逐阶收紧区间 + 复用前一个特征值"可省 20–30% 但仍难超过 LAPACK;
2. C 的候选向量缓存按 `2(k+2)` 条分配, 但最终只需保留前 `k` 条 → 内存可再降约一半;
3. `EWT` 的 `Slepian-Optimize` 预处理分支目前仍直接调 `scipy.signal.windows.dpss`(带 `EWT_SLEPIAN_MAX_SAMPLES=32768` 上限); 接入本入口只需把该调用换成 `Utils.get_slepian().slepian(..., mod=...)`, 属独立小改动。

---

## 8. 测试覆盖

`tests/test_slepian.py`(286 项, 全绿):

- 正确性: 三后端 × 参数网格 vs scipy(序列逐元素含符号 + 集中比)、正交性、dtype、形状;
- 契约: 默认条数、集中比降序且有界(`(0,1]`, 大 NW 时前 `2NW−1` 阶 ≈ 1)、`norm` 三档、`sym=False`;
- 边界: N=1/2/3/5/7、`NW` 极小与极大、K=N 的集合一致性、**小 N 自动换后端(warning + 数值一致)**;
- 守卫: 非法 `N`/`halfBW`/`nTapers`/`norm`/`sym`/后端名、numpy 预算上限、C 不可用时的报错路径;
- 集成: `Utils.get_slepian()` 惰性注册与实例唯一性、`available_backends()`、`resolve_backend()` 的默认/auto 解析;
- **缓存**: Tier-1 缓存的命中/未命中/精确位键/LRU 淘汰/字节预算/只读主副本/副本可写/开关与统计(见 `docs/Slepian_Cache_Report.md`)。
