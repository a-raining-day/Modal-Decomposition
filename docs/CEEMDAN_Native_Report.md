# CEEMDAN 原生实现报告 (`CEEMDAN`)

- **实现**: `src/Modal_Decomposition/CEEMDAN.py` (注册键 `CEEMDAN`, 自 0.3.0 起为原生实现; `pyemd=True` 保留 PyEMD 过渡通道)
- **测试**: `tests/test_ceemdan.py` (**71 passed**) · `tests/test_eemd.py` (**61 passed**)
- **对照**: `EEMD` / `CEEMDAN` 于 0.3.0 完成换版 —— 见 §10
- **生成日期**: 2026-09-16 · Python 3.10 · numpy 2.2.6 · scipy 1.15.3 · PyEMD 1.9.0

---

## 1. 结论摘要

1. `CEEMDAN` 是 **Torres 2011 规范式** CEEMDAN 的原生实现, 不依赖 PyEMD。
2. 递归正确性有**独立参照证明**: `noise_width=0, trials=1` 时逐阶与
   "在归一化信号上反复 `EMD(max_imf=1)` 并减残差" 的轨迹**逐位相同**
   (`diff = 0.000e+00`, 见 §5.1)。
3. 重构精度 `6.9e-18 ~ 2.2e-16` (`S == IMFs.sum(axis=0) + Res` 硬性成立)。
4. 内层 EMD 的 **6 个引擎参数全部显式暴露并透传**, 无一写死 (§4)。
5. `CEEMDAN`(PyEMD 版) 与 `CEEMDAN`(原生) **并存**, 便于逐点对拍; PyEMD
   仅在 `pyemd=True` 类的过渡分支中保留 (本实现暂未提供, 见 §8)。
6. **性能**: 全规模上 `CEEMDAN` 快于 PyEMD 版 **2.0–4.9×**, 也快于同样实现
   Colominas 2014 的原生 `ICEEMDAN` **1.2–4.0×** (见 §5.7)。

## 2. 算法规范 (Torres 2011)

记 `E_j(·)` 为原生 `EMD` 引擎提取的第 `j` 阶 IMF, `w_i` 为第 `i` 条白噪声实现,
`M = trials`, `ε = noise_width`:

```
r_0 = S
c_1 = (1/M) Σ_i E_1( r_0 + ε·w_i )
r_1 = r_0 − c_1
c_k = (1/M) Σ_i E_1( r_{k−1} + ε·E_k(w_i) )        (k ≥ 2)
r_k = r_{k−1} − c_k
```

**与 EEMD 的唯一区别在第 2 式**: 每阶注入的是**噪声自身的第 `k` 阶 IMF**
`E_k(w_i)`, 而不是原始白噪声。EEMD 用同一份白噪声去打每一阶, 高频噪声会在
低频阶引入虚假模态 (模态混叠); CEEMDAN 改成 "本级频段的噪声", 于是每阶的辅助
扰动都落在该阶将要提取的频段上。这也是 CEEMDAN 所需 `trials` 远小于 EEMD 的原因。

实现要点:

- `E_k(w_i)` 由同一批噪声实现**一次性预分解**得到 (`info["noise_imfs"]` 记录
  每条实现的可用阶数), 跨阶复用;
- 各实现的可用阶数不同 (实测如 `[6,6,6,6,6,6,7,6,6,...]`), 本实现**按阶动态
  跳过**阶数不足的实现, 实际参与数记在 `info["trials_used"]`;
- 递归整体在**单位标准差**的信号上进行 (与 PyEMD 同做法), 故 `ε` 与两个阈值
  对任意幅度的信号含义一致; 模态与残差最后统一乘回原尺度。

### 停机判据

| `stop_reason` | 含义 |
|---|---|
| `constant_signal` | 输入标准差为 0 (常量) |
| `residual_monotonic` | 残差单调 (⇔ 无可提取极值) |
| `range_thr` | 残差极差 `max(r)−min(r) < range_thr` |
| `total_power_thr` | 残差总功率 `Σ\|r\| < total_power_thr` |
| `max_imf` | 达到 `max_imf` 上限 |
| `noise_exhausted` | 噪声池可用阶数耗尽 (阶数上限约束) |
| `residual_no_extrema` | 某阶全部扰动都取不到 IMF |

`residual_no_extrema` 与 `noise_exhausted` 是**本实现特有的**约束: PyEMD 版
没有"噪声池阶数上限"这一限制, 它的阶数只受 `max_imf` 与自身停机条件约束。

## 3. 顺序性: 为什么没有并行

CEEMDAN 是**残差链迭代**, 整条链串行, **没有并行空间**:

- 各阶之间 `r_k = r_{k−1} − c_k` 是链式依赖, 第 k 阶必须等第 k−1 阶算完;
- 同一阶内的 `M` 次试验共用**同一个** `r_{k−1}`, 看似独立, 但这一批只能在
  "第 k 阶残差就位"之后开始, 下一批又要等本批均值 `c_k` 定下来才算得出 `r_k`
  ——批与批之间仍被链卡住, 并行只发生在批内部;
- 噪声池预分解 `E_k(w_i)` 与主链无关, 但那是一次性 `(M, N)` 批量操作, 摊到每阶
  后只剩查表。

在阶内强上进程池, 省下的只是单次 EMD (几千点上毫秒级) 的耗时, 而每阶都要重开
一次池并同步一轮, 开销直接盖过收益。故 `CEEMDAN` **不提供 `parallel` 参数**
(对比 `EEMD` 每次试验都在原始信号上独立完成, 是 embarrassingly parallel,
那里的 `parallel=True` 有意义)。

> 环境注记: 本机沙箱禁止命名管道, `multiprocessing.Pool()` 抛
> `PermissionError: [WinError 5]`。这同时也是 PyEMD 版 `CEEMDAN(parallel=True)`
> 在本环境不可用的原因。

## 4. 参数表

### 4.1 算法参数

| 参数 | 默认 | 语义 |
|---|---|---|
| `trials` | `100` | 每阶集成实现数 `M` (≥1) |
| `noise_width` | `0.005` | 相对注噪幅度 `ε` (≥0); `0` = 关闭注噪 |
| `max_imf` | `-1` | 最大阶数; `-1` = 完全分解 |
| `seed` | `None` | 局部随机种子 (全局种子会覆盖并告警) |
| `range_thr` | `0.01` | 极差停机阈值 (归一化尺度) |
| `total_power_thr` | `0.05` | 总功率停机阈值 (归一化尺度) |
| `rich_info` | `False` | 额外输出 `info["ensemble_std"]` |

### 4.2 内层 EMD 引擎参数 (全部透传, 无写死)

| 参数 | 默认 | 透传至 |
|---|---|---|
| `spline_kind` | `"CubicSpline"` | `EMD(spline_kind=...)` |
| `nbsym` | `2` | `EMD(nbsym=...)` |
| `max_iter` | `100` | `EMD(max_iter=...)` |
| `sd_thr` | `0.01` | `EMD(sd_thr=...)` |
| `find_peaks_mod` | `"numpy"` | `EMD(find_peaks_mod=...)` |
| `faster` | `True` | `EMD(faster=...)` |

单一构造点 `_engine(max_imf)`, 同时供**噪声池预分解** (`max_imf=-1`) 与
**主循环** (`max_imf=1`) 使用 —— 两处共用同一套引擎参数, 不存在只影响其一的
"半透传"。

### 4.3 `faster` 默认值的选择

默认 **`True`**, 与裸 `EMD` 的默认 (`faster=False`, 质量档) **不同**。理由:
CEEMDAN 每阶要跑 `trials` 次引擎、每次只为读**一阶** IMF, 而 `trials` 次平均
基本上吸收了窄带门 `|zc − ext| ≤ 1` 能带来的行级纯度。实测 (5 次中位):

| 信号 | `faster=True` | `faster=False` | 比值 |
|---|---:|---:|---:|
| N=1024, trials=30 | 0.383 s | 1.281 s | 3.35× |
| N=2048, trials=20 | 1.103 s | 1.187 s | 1.08× |
| N=2048, trials=20 (noise 0.6) | 0.942 s | 1.289 s | 1.37× |
| N=4096, trials=20 (noise 0.6) | 1.280 s | 1.900 s | 1.49× |

即 `faster=False` 最坏慢约 3×, 在测过的信号上**从不更快**。需要与裸 `EMD`
完全一致时显式传 `faster=False`。

### 4.4 `find_peaks_mod` 为何在结果上看不出差异

`Utils.Peaks` 的契约就是 `numpy` / `scipy` 两后端在**无平台信号**上检测结果
逐位一致 (库文档明确记录该契约)。实测 `find_peaks_mod="scipy"` 既不改变噪声池
阶数分布也不改变输出 —— 这是**契约成立**的证据, 不是死参数。平台信号上两者
计数口径不同 (见 `Utils/Peaks` 与 `docs/EMD_vs_PyEMD_Detailed_Comparison.md`)。

## 5. 验收数据

### 5.1 递归正确性 (核心验收项)

`noise_width=0, trials=1` 时辅助项消失, 每阶 `M` 次试验完全同解, 均值即该解,
故整个递归退化为 "在归一化信号上反复 `EMD(max_imf=1)` 并减残差"。
用**同一个 EMD 实例**复刻该轨迹, 逐阶比较:

```
阶0 diff vs CEEMDAN = 0.000e+00
阶1 diff vs CEEMDAN = 0.000e+00
...
阶7 diff vs CEEMDAN = 0.000e+00        (linear 包络, 512 点双音)
```

**逐位相同**。该对照同时验证递归公式与残差链, 是 `tests/test_ceemdan.py::
test_zero_noise_degenerates_to_stepwise_emd` 的断言基础 (2 个 `faster` 档 ×
3 个 `spline_kind` = 6 组全部通过)。

> 踩坑记录: 最初的参照用 `EMD(max_imf=-1)` 逐阶取, 在 `linear` 包络下**不成立** ——
> 该包络的线性样条在数值噪声量级的残差上衰减极慢, 参照会一路拆到 **31 阶**,
> 而 CEEMDAN 在第 8 阶因"噪声池阶数耗尽"停下。**两者停机机制本就不同**, 参照
> 必须复刻同一口径 (残差单调即止) 才可比。

### 5.2 重构精度

| 输入 | 阶数 K | 重构误差 `max\|recon − S\|` |
|---|---:|---:|
| 双音 (N=64/256/1024) | 3–5 | ≤ 1e-16 |
| 双音 + 噪声 0.15 | 6 | 6.9e-18 |
| 双音 + 噪声 0.6 | 6 | ≤ 1e-16 |
| 纯噪声 (N=512) | 6 | 2.8e-17 |
| 单音 | 7 | 2.0e-20 |
| 常量 / 零信号 | 0 | 0.0 |
| 单调斜坡 | 1 | ≤ 1e-9 |

### 5.3 参数有效性 (无静默死参数)

每个引擎参数都必须真的改变**噪声池**或**输出** (实测):

| 参数改动 | 噪声池阶数分布变 | 输出变 | K |
|---|---|---|---|
| `spline_kind=PCHIP` | ✅ | ✅ | 8 |
| `spline_kind=linear` | ✅ | ✅ | 10 |
| `nbsym=0` | ✅ | ✅ | 4 |
| `max_iter=3` | ✅ | ✅ | 5 |
| `sd_thr=0.3` | ✅ | ✅ | 5 |
| `faster=False` | ✅ | ✅ | 6 |
| `find_peaks_mod=scipy` | — (契约一致) | — (契约一致) | 5 |

### 5.4 噪声幅度确实生效

`noise_width` 变化必须改变结果 (排除"被归一化约掉的死参数"嫌疑):

```
nw=0.0   -> K=6      nw=0.002 -> K=6
nw=0.005 -> K=6      nw=0.02  -> K=7      nw=0.1 -> K=7
```

### 5.5 引擎调用次数 (逐次计数核验)

用计数器包住 `EMD.decompose` (**只计顶层调用** —— 早期版本把 `__call__` 引起的
嵌套调用一并计入, 得到虚高的 625, 已修正) 实测:

| 路径 | 公式 | `trials=20, K=5` |
|---|---|---:|
| 噪声池 | `M` | 20 |
| 主循环 (各阶试验) | `K·M` | 100 |
| 残差极值探测 | `K + 1` | 6 |
| **合计 (含探测)** | `M + K·M + K + 1` | **125** |
| 对照: 不含探测 | `M + (K+1)·M − 跳过数` | 138 |

按 `max_imf` 分类的实测结果 `{-1: 20, 1: 105}` 与公式完全吻合
(`20` = 噪声池, `105 = 5×20 + 6`)。

**残差极值探测是省的不是费的**: 不探测时, 失败的 K+1 阶仍要跑满 M 次试验才发现
"残差已无内容" (实测该阶跑了 18 次); 探测只在每个**成功**阶多付 1 次。只要
`K < M` 就净赚, 而集成法恒满足 (trials 通常 50~100, K 是个位数) —— 故 K 越小
优势越大 (`trials=20, K=5` 时 138 → 125)。

墙钟 (5 次中位, 单进程, `spline_kind="CubicSpline"`, `faster=True`):

| N | trials | 耗时 |
|---:|---:|---:|
| 1024 | 30 | 0.34 s |
| 2048 | 20 | 0.82 s |
| 4096 | 20 | 1.04 s |

### 5.6 契约与注册

```
ast.parse + compile                OK
test_ceemdan.py                71 passed
tests/_cases.py 契约矩阵            test_contract / test_facade / test_reconstruction 全过 (18 方法)
DecompositionResult / Decomposer   契约成立
Class == Name == Reference == _ClassRegistry == Function    一致 (18 个方法)
Function.CEEMDAN == Class.CEEMDAN                   等价
CEEMDAN (PyEMD 版)                 仍可用, 零改动
```

### 5.7 与 PyEMD 版 CEEMDAN / 原生 ICEEMDAN 的性能对比

同一进程、同一信号 (双音 + 0.15 噪声)、同 `seed`, 3 次中位
(N=8192 因耗时较长取 1 次)。PyEMD 一律 `parallel=False` (其默认为 `True`,
而本环境禁止命名管道, 见 §3 环境注记)。

| N | trials | `CEEMDAN` (native) | `CEEMDAN` (PyEMD) | `ICEEMDAN` (native) | vs PyEMD | vs ICEEMDAN |
|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 50 | **0.568 s** | 2.764 s | 2.295 s | **4.87×** | 4.04× |
| 2048 | 100 | **5.315 s** | 10.578 s | 6.109 s | 1.99× | 1.15× |
| 4096 | 100 | **5.860 s** | 18.723 s | 10.301 s | 3.20× | 1.76× |
| 8192 | 100 | **9.998 s** | 44.795 s | 19.729 s | 4.48× | 1.97× |

结论:

- 全规模上原生实现都更快, 优势 **2.0–4.9×**;
- 相对 `ICEEMDAN` 的优势 (1.2–4.0×) 来自两点: `faster=True` 内部档位, 以及
  `ICEEMDAN` 的默认 `ensemble_size=300` (此处已对齐为 `ensemble_size=trials`,
  否则差距更大);
- N=2048 那组的优势偏小 (1.99×), 属该规模下的正常波动 —— 其余三组都在
  3.2–4.9×。

**注意可比性**: 三者中 `CEEMDAN` 与 `ICEEMDAN` 是**不同算法**
(Torres 2011 vs Colominas 2014), `CEEMDAN`(PyEMD) 与 `ICEEMDAN` 才是同一
算法族 (见 §6.1)。上表是**成本对比**, 不是"同算法换实现"的加速比 ——
后者只在 `CEEMDAN`(PyEMD) 与 `ICEEMDAN` 之间成立。

## 6. 与 PyEMD 版 `CEEMDAN` 的差异

### 6.1 算法版本 (最重要)

**PyEMD 的 `CEEMDAN` 不是 Torres 2011, 而是 Colominas 2014 改进版。**
依据: `PyEMD/CEEMDAN.py` 自身 docstring 说明 "contains proposed improvements
from paper [Colominas2014]", 且代码在每阶只算**一组**局部均值并递归展开
(`PyEMD/CEEMDAN.py` 的 `local_mean` 分支), 而非 Torres 2011 的
"第 k 阶对全部 trial 做 EMD 后取平均"。

后果: **该改进版在本库已有原生实现, 即 `ICEEMDAN`。** 所以:

- `CEEMDAN` 走规范式, 与 `ICEEMDAN` 构成**两种真正不同的算法**;
- `CEEMDAN`(PyEMD 版) 与 `ICEEMDAN` 在算法上高度重复;
- 因此 `CEEMDAN` 与 `CEEMDAN` 的输出**必然不同**, 对拍只能定性
  (本库测试只断言"阶数量级接近 + 首 IMF 频段一致")。

### 6.2 注噪幅度参数

| | PyEMD 版 `CEEMDAN` | `CEEMDAN` |
|---|---|---|
| 参数名 | `noise_scale` (默认 `1.0`) | `noise_width` (默认 `0.005`) |
| 是否生效 | **否** —— `beta_progress=True` (其默认) 会把每条噪声 IMF 除以自身首阶 IMF 的 std, EMD 对幅度齐次, 于是 `noise_scale` 被整体约掉 | **是** |
| 真实旋钮 | 未暴露的 PyEMD `epsilon` (默认 `0.005`) | 即 `noise_width` |

实测: PyEMD 版 `noise_scale` 由 `1.0` 改到 `100.0` (相差 100 倍), 输出最大差
仅 **4.440892098500626e-16** (1 ulp)。对拍关系为 `noise_width ≈ epsilon`。

### 6.3 其它

| 项 | PyEMD 版 | `CEEMDAN` |
|---|---|---|
| `spline_kind` 词汇 | PyEMD 名 (`cubic`/`pchip`/`akima`/...) | canonical `Utils.Spline` 名 (`CubicSpline`/`PCHIP`/`linear`) |
| `extrema_detection` | 有 | 无 (原生引擎无此概念) |
| `noise_kind` | 有 (`normal`/`uniform`) | 无 (固定高斯) |
| `parallel`/`processes` | 有 (本环境不可用) | 无 (无并行空间, 见 §3) |
| `info` | `{}` 空 | 5–6 个诊断键 + 可选 `ensemble_std` |
| 阶数上限 | 仅受 `max_imf` | 另受"噪声池可用阶数"约束 |

## 7. 已知限制

1. **阶数受噪声池约束**: 最大可提取阶数 = 各噪声实现可用阶数的最小上界
   (`info["noise_imfs"]` 的 max)。噪声自身分解不出更多阶时, 主循环以
   `noise_exhausted` 停止。PyEMD 版无此约束。
2. **无并行**: 见 §3, 这是算法的顺序性决定的, 不是未实现。
3. **无 `pyemd=True` 过渡分支**: 本实现尚未提供; 需要 PyEMD 语义时直接用
   `Class.CEEMDAN` (两者并存, 无需开关)。
4. **`noise_kind` 未提供**: PyEMD 支持 `uniform`; 本实现固定高斯白噪声, 需要时
   属于后续增补项。
5. **`linear` 包络下的尾部阶**: 线性样条在数值噪声量级的残差上衰减极慢, 尾阶
   数值不稳定 (§5.1 踩坑记录)。这是 `EMD` 引擎的既有特性, 非本实现引入。

## 8. 待定夺事项

| # | 事项 | 说明 |
|---|---|---|
| 1 | **算法版本** | PyEMD 的 CEEMDAN 实为 Colominas 2014, 而本库 `ICEEMDAN` 已是该版本的原生实现。是保持"`CEEMDAN` = Torres 2011 规范式" (与 `ICEEMDAN` 区分), 还是让 `CEEMDAN` 复刻 PyEMD 语义 (则与 `ICEEMDAN` 重复)? |
| 2 | **`noise_width` 与 PyEMD `epsilon` 的对拍关系** | 现按 `noise_width ≈ epsilon` 对齐 (默认 `0.005`)。是否需要在文档/参数名上进一步显式声明? |
| 3 | **`noise_kind`** | 是否增补 `uniform` 选项以对齐 PyEMD 参数面? |
| 4 | **转正时机** | `EEMD` 的先例是"验收通过后再切换到正式键"。`CEEMDAN` 是否同样待验收后再占用 `CEEMDAN` 键? |
| 5 | **`faster` 默认值** | 现为 `True` (§4.3)。是否接受与裸 `EMD` 默认不同? |

## 9. 复现

```bash
# 运行本报告的全部测试
python -m pytest tests/test_ceemdan.py -q

# 与 PyEMD 版并存使用
python -c "
import numpy as np
from Modal_Decomposition import Class
t = np.linspace(0, 1, 1024, endpoint=False)
S = np.sin(2*np.pi*5*t) + 0.5*np.sin(2*np.pi*37*t)
print('native:', Class.CEEMDAN(trials=50, seed=0).decompose(S).IMFs.shape)
print('pyemd :', Class.CEEMDAN(trials=50, seed=0).decompose(S).IMFs.shape)
"
```
