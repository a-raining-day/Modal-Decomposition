# FFT 后端实测报告 (Utils.FFT) + Memory 检测失效修复

- 日期: 2026-03-04
- 范围: 新增 `src/Modal_Decomposition/Utils/FFT.py`(后端分发)、`Base/ConstDefine.py` 的
  `BIG_ARRAY` / `FFT_BACKEND` / `FFT_BACKEND_SMALL` / `FFT_BACKEND_BIG`, 以及
  `Utils/Memory.py` 的检测修复
- 基准脚本: `tests/comparison/bench_fft.py`(可复现), 原始数据 `docs/FFT_Backend_Results.json`
- 结论一句话: **默认后端选 `numpy`**(冷态在每个尺寸都最快、提交内存最省、零依赖);
  `pyfftw` **暖态**最强(10 MB 快 5×、100 MB 快 3.1×, 需多线程)但冷态要付 ~0.15 s 规划、
  且 ≥500 MB 因内存压力反转变慢 ⇒ 列为**批量场景的一行切换**;
  `scipy` 因 `copy=True` 又慢又费内存; `tiled`(四步分块)在时间与内存两个维度**都**没有收益;
  `cupy` 传输主导, 全尺寸慢于 numpy, 只允许显式请求。

---

## 1. 环境与后端可用性

| 项 | 值 |
|---|---|
| Python / NumPy / SciPy | 3.10.11 / **2.2.6** / **1.15.3** |
| `np.fft.rfft(float32)` | **complex64**(NumPy 2.x 原生支持 float32, 不需 scipy) |
| `scipy.fft` | 支持 `workers`(多线程) |
| `pyfftw` | **0.15.0 已安装**(导入 0.25s; `interfaces.cache` 已启用) |
| `cupy-cuda12x` | 14.2.0 已安装; 首次 `import cupy` ≈2.9s(另有 >300s 无进展的观测) |
| 机器 | 32 逻辑核, 物理内存 15.6 GB(实测**可用 3.0–8 GB**, 另有 ~15 GB 空闲页面文件) |

`fft.available_backends()` 默认用 `importlib.util.find_spec` 判断**是否安装**
(不导入, 因此不会被 cupy 的慢导入卡住); `probe=True` 才真正导入。

**默认路径的缺库降级**: `mod=None`/`"auto"` 选中的后端若未安装, 会发一次 `UserWarning`
并降级为 `numpy`(默认路径不该因可选依赖缺失而抛错); **显式**指定后端时仍抛
`ImportError` 并附 `pip install` 命令。

---

## 2. 对外 API (唯一入口: `class fft`, 全部为静态方法)

```python
from Modal_Decomposition.Utils import fft        # 类, 不需要实例

fft.fft(x)                    # 默认变换 (复数 FFT)
fft.ifft(X); fft.rfft(x); fft.irfft(X, n=N)
fft.fftshift(X); fft.ifftshift(X)
fft.rfft(x, mod="pyfftw", workers=8)             # 单次指定后端
fft.resolve_backend(None, x.nbytes)              # 当前体积会选中哪个后端
fft.available_backends(); fft.describe()
```

设计上刻意保持朴素: **只有静态方法, 没有实例语义、没有元类/描述符**; 模块级除该
类之外不暴露任何公开函数。`mod` 取值即 `Base.ConstDefine.FFT_BACKEND_LIST`
(另接受 `FFT_BACKEND_ALIAS` 里的简写, 如 `"np"` / `"fftw"` / `"gpu"`); `"auto"` 额外允许
(默认)。相关常量全部集中在 `Base/ConstDefine.py`: `BIG_ARRAY`、`FFT_BACKEND`、
`FFT_BACKEND_SMALL`、`FFT_BACKEND_BIG`、`FFT_BACKEND_LIST`、`FFT_BACKEND_ALIAS`、
`FFT_PIP_PACKAGE`、`FFT_THREAD_MIN_ELEMS`、`FFT_TILED_MIN_ELEMS`, 以及
`CACHE_KEY["fft"]` (后端库的缓存键)。

- `workers` / `planner_effort` 是**工具层**参数: 只对 `scipy` / `pyfftw` 生效, 其它后端**忽略**
  (不是报错), 调用方无需按后端分支写代码。
- 默认路径 (`mod=None`/`"auto"`) 选中未安装的后端时**降级 numpy + `UserWarning`**;
  显式指定时抛 `ImportError` 并附 `pip install` 命令。

### 2.1 基准协议(以及本轮修掉的三个度量缺陷)

`bench_fft.py` 的每条 (后端 × 尺寸) 记录 = 独立子进程 + 数据生成 + 该后端的 `rfft`/`fft`。

安全闸门(吸取上一轮 VMD 实验把机器拖入换页的教训):

1. **跑前过预算闸门**: `Utils.Memory.memory_budget(预计峰值)`, 不通过就记 `over_budget`
   并且**不分配**;
2. **监控提交内存**(`vms`/private), 不再用工作集 RSS —— Windows 会主动裁剪工作集,
   只看 RSS 会让"提交 10 GB 而物理只剩 3 GB"的进程被判为安全;
3. 超预算/超时即杀, 逐条记录状态。

本轮修掉的三个**度量缺陷**(都会直接骗到结论):

| 缺陷 | 症状 | 修正 |
|---|---|---|
| 父进程 50 ms 轮询采样 | 小尺寸 FFT 仅 16 ms, 采样完全错过峰值 | worker 内部起 5 ms 采样线程 |
| 校验混入采样区 | 参考数组/差值数组额外占一份全量内存, 污染"FFT 本身峰值" | 校验移出采样区 |
| "输出前缀 vs 前缀的 FFT" | 数学上不相等(实测假误差 9.9e+03) | 大尺寸改为**同一段切片喂两个后端**比对 |

---

## 3. 实测结果

### 3.1 rfft(float64 → 半谱), best-of-N

| 尺寸 | 指标 | numpy | scipy-w1 | scipy-w4 | scipy-w8 | scipy-all | tiled | auto |
|---|---|---|---|---|---|---|---|---|
| 1 MB | 秒 | **0.001** | 0.001 | — | — | 0.001 | 0.001 | 0.001 |
| 10 MB | 秒 | 0.016 | 0.017 | — | — | **0.014** | 0.013 | 0.014 |
| 100 MB | 秒 | **0.211** | 0.404 | 0.383 | 0.364 | 0.400 | 0.559 | 0.214 |
| 100 MB | 提交峰值(MB) | **402** | 1260 | 1260 | 1261 | 1260 | 1260 | 402 |
| 500 MB | 秒 | **1.221** | 1.430 | 1.426 | 1.434 | 1.431 | 1.485 | 1.247 |
| 500 MB | 提交峰值(MB) | **2005** | 3264 | 3264 | 3264 | 3264 | 3264 | 2005 |
| 1024 MB | 秒 | **2.872** | 3.309 | 3.258 | 3.319 | 3.306 | 3.182 | 3.042 |
| 1024 MB | 提交峰值(MB) | **4106** | 5889 | 5890 | 5889 | 5890 | 5889 | 4106 |

`max|err|`: numpy 恒为 0(参考实现本身); 其余 ≤ 1.5e-12。`pyfftw` 两行均为
`unavailable`(未安装, 报错附安装命令)。`auto` 在 <300 MB 选 numpy, ≥300 MB 按
`FFT_BACKEND_BIG` 选 numpy(该常量已按本表结论由 `"scipy"` 改为 `"numpy"`)。

### 3.2 复数 fft 对照组(说明 tiled 的适用面)

| 尺寸 | numpy | scipy-all | tiled | pyfftw-all |
|---|---|---|---|---|
| 10 MB | 0.031 s | **0.027 s** | 0.074 s | — |
| 100 MB | 0.373 s / 803 MB | 0.467 s / 1760 MB | 0.924 s / 1561 MB | **0.115 s** / 1565 MB |
| 500 MB | 1.965 s / 4009 MB | 1.995 s / 5768 MB | 4.744 s / 4768 MB | 2.033 s / 5943 MB |
| 1024 MB | 6.144 s / 8210 MB | 6.930 s / 11020 MB | 10.546 s / 8968 MB | **3.068 s** / 11036 MB |

**注意 rfft 与 fft 在大尺寸上的赢家不同**: 1 GB 复数 fft 是 pyfftw 快 2×(3.07 vs 6.14),
而 1 GB `rfft` 是 numpy 快 2×(2.87 vs 5.71) —— 后者因为 pyfftw 处理实数输入要额外做一份复数
拷贝(内存 6796 vs 4106 MB)。库内主要用 `rfft`/`irfft`, 故默认仍取 numpy; 若某方法以复数
`fft` 为主, 应单独按本表选型。

### 3.3 pyfftw(0.15.0) —— **暖态最快, 冷态与 500MB+ 都不划算**

暖态 = 同形状 plan 已被 `pyfftw.interfaces.cache` 复用; 冷态 = 进程内首次(含 FFTW 初始化+规划)。
`reps=3`, 表内 `cold/warm`(秒), 括号为提交峰值增量:

| 尺寸 | numpy | pyfftw-w1 | pyfftw-w8 | pyfftw-all |
|---|---|---|---|---|
| 1 MB | 0.001 / 0.001 | 0.156 / 0.0004 | 0.149 / 0.0006 | 0.154 / 0.0005 |
| 10 MB | 0.016 / 0.016 | 0.192 / 0.0074 | 0.189 / **0.0031** | 0.207 / **0.0027** |
| 100 MB | 0.211 / 0.211 (402MB) | 0.649 / 0.457 | 0.514 / 0.088 (1212MB) | 0.514 / **0.069** (1215MB) |
| 500 MB | **1.221** / 1.221 (2005MB) | 3.429 / 3.429 | 2.743 / 2.743 (3719MB) | 2.664 / 2.664 (3747MB) |
| 1024 MB | **2.872** / 2.872 (4106MB) | — | 6.177 / 6.077 (6790MB) | 6.048 / **5.706** (6796MB) |

- **暖态**: 10 MB 快 5×、100 MB 快 3.1×(全核); 单线程反而比 numpy 慢(100 MB 0.457 vs 0.211)
  ⇒ **pyfftw 的收益依赖多线程**。
- **冷态**: numpy 在**每个尺寸**都更快(1 MB: 0.001 vs 0.149; 100 MB: 0.211 vs 0.514;
  1 GB: 2.872 vs 6.048) —— pyfftw 要付 ~0.15 s 的 FFTW 初始化与规划。
- **≥500 MB 反转**: pyfftw 慢 2.2×(500 MB)、2×(1 GB), 且提交峰值高 1.7–1.9×
  (1 GB: 6796 vs 4106 MB)。原因是它的多线程工作区更大, 在内存紧张的机器上先触发换页,
  把线程收益吃光 —— **换更大内存的机器需要复测**。
- **`planner_effort`**: `FFTW_MEASURE` 在 1M 点上规划 9.2 s(暖态不比 ESTIMATE 快),
  在 **500 MB 上规划超过 700 s(2757 s CPU)仍未完成, 被迫中止** ⇒ 库内默认
  `FFTW_ESTIMATE` 是唯一实用选择; MEASURE 只适合"一次规划、长期批量"的场景。
- **盈亏平衡(按冷/暖差估算)**: 1 MB 永不划算; 10 MB 约需 15 次同形状调用;
  100 MB 约 2–3 次。库内单次 `decompose` 通常只发 2–3 次变换(形状 2 种),
  因此**默认不用 pyfftw**; 批量处理(同形状反复调用)则强烈推荐。

### 3.4 cupy (GPU)

| 尺寸 | best(s) | 提交峰值(MB) | max\|err\| | 备注 |
|---|---|---|---|---|
| 1 MB | 2.896 | 727 | 6.6e-13 | 含首次 `import cupy` |
| 10 MB | 0.761 | 758 | 3.1e-12 | |
| 100 MB | **0.788** | 1346 | 2.4e-12 | numpy 同尺寸 **0.206 s / 402 MB** |

**结论: cupy 在本机全部尺寸上都比 numpy 慢**(100 MB 慢 3.8×)且更费内存。原因是库契约要求返回
主机数组 ⇒ 每次调用都有"主机→设备→主机"两次 PCIe 传输, 在这些尺寸下传输完全主导变换时间;
GPU 的 crossover 点远在 1 GB 以上(且要避免回传, 需要库外持有设备数组 —— 与当前契约冲突)。

关于导入: 本轮实测 `import cupy` ≈ **2.9 s**(含 CUDA 初始化); 但本会话早期有两次探测
**>137 s / >300 s 无进展**(当时另一个全局 Python 进程正占满一个核, 疑与 CUDA 冷启动/驱动争用有关)。
因此仍按"可能很慢"处理: **只在 `mod="cupy"` 显式请求时导入**, 不进 `auto`、不进 `find_spec` 之外的探测。

---

## 4. 结论与默认选择

默认策略落在两个常量上(`Base/ConstDefine.py`):

```python
FFT_BACKEND        = "auto"      # 按体积分流
FFT_BACKEND_SMALL  = "numpy"     # < BIG_ARRAY (300MB)
FFT_BACKEND_BIG    = "numpy"     # >= BIG_ARRAY
```

1. **默认 = `numpy`(两个区间都是)**, 依据:
   - **冷态**在每个尺寸都最快(见 §3.3), 而库内一次 `decompose` 只发 2–3 次变换、
     形状只有 2 种 ⇒ pyfftw 的规划成本摊销不掉;
   - 提交内存最省(1 GB: 4106 MB vs pyfftw 6796 MB / scipy 5889 MB), 与
     `BIG_ARRAY` 想表达的"大数组要克制"一致;
   - 零可选依赖, 行为可预测。
2. **批量/热循环请把 `FFT_BACKEND_SMALL` 改成 `"pyfftw"`**: 暖态 10 MB 快 5×、100 MB 快 3.1×
   (需多线程, 全核最佳); 盈亏平衡见 §3.3。这是"一次改常量"级别的切换。
3. **`scipy` 不做默认**: 它比 numpy 慢且更费内存(§3.1), `workers` 也几乎不影响结果;
   其开销来自 `copy=True` 默认多复制一份输入。若要用, 应显式 `copy=False`/`overwrite_x=True`
   (会破坏调用方输入, 库内默认不能这么干)。
4. **`tiled`(四步分块)两个维度都没有收益**: 复数 fft 慢 2–2.5×, 内存还更高;
   `rfft` 更是**不能**这样分块(半峰索引呈阶梯形, 中间量必须 N 个复数, 峰值 ≥12N,
   比直算 4N+8N 更大)。保留为显式选项, 不参与 `auto`。
5. **`cupy` 只允许显式请求**: 每次调用含两次主机↔设备传输, 实测全尺寸慢于 numpy
   (100 MB 慢 3.8×), 且导入开销 2.9 s 起。
6. **`pyfftw` 是唯一"分场景更优"的后端**, 所以它留在名单里、并写进了
   `FFT_BACKEND_SMALL/BIG` 的注释与本节 —— 而不是被删掉。

### 4.1 顺带得到的大数组事实

- **`np.fft.rfft` 需要约 2× 输入的额外内存**: 1 GB 输入实测提交峰值 4106 MB
  (输入 1 GB + pocketfft 内部 N 点复数 scratch 2.1 GB + 半谱 1 GB)。做库内大数组预算时
  必须按"输入 ×2 + 输出"估, 而不是"输入 + 输出"。
- 这条直接解释了 `docs/VMD_Large_Array_Iteration_Report.md` 里 VMD 的"FFT 内存地板"。

---

## 5. Memory 检测失效的修复(第 3 项任务)

### 5.1 旧版为什么失效

旧 `should_use_memmap` 用 `0.6 × psutil.virtual_memory().available` 判定。实测这台机器:

| 口径 | 值 |
|---|---|
| 物理可用 | 3.38 GB |
| 页面文件空闲 | 15.28 GB |
| **可提交余量**(物理可用 + 页面文件空闲) | **18.66 GB** |
| 本进程私有提交 vs 工作集(同一次探测) | 764 MB vs 37 MB |

于是"提交 10 GB 而物理只剩 3 GB"的调用**始终被判为内存充足** —— VMD 300 MB 用例真的提交到
10148 MB 并让整机换页(`MemCompression` 1.4 GB)。失效有三个成分, 逐条修:

1. **只看物理可用, 不看可提交** → 新增 `get_commit_available()`, 预算取两者较小值;
2. **不扣本进程已占用** → 新增 `get_process_memory()`, `memory_budget(nbytes=...)` 把它算进去;
3. **没有系统余量** → 新增 `MEMORY_RESERVE_RATIO`(默认 0.25) 与 `COMMIT_RATIO_LIMIT`(0.75)。

### 5.2 新口径与效果

```
budget = min(物理可用, 可提交余量) × (1 − MEMORY_RESERVE_RATIO) − 调用方额外占用
       = min(2.53 GB, 10.49 GB) = 2.53 GB        # limited_by = physical
```

| `extra` | 旧判定 | 新判定 |
|---|---|---|
| 0 | "充足" | `budget = 2.53 GB`, ok |
| 1 GB | "充足" | `budget = 1.53 GB`, ok |
| **4 GB** | **"充足"(实际会把机器拖入换页)** | **`budget = −1.47 GB`, ok = False** |
| 10 GB | "充足" | `budget = −7.47 GB`, ok = False |

新增 API: `get_memory_snapshot` / `get_commit_available` / `get_process_memory` /
`memory_budget` / `set_memory_reserve` / `format_bytes`; 快照 TTL 由 1.0 s 收紧到 0.25 s
(大数组分配期间内存变化很快)。`should_use_memmap` 的**语义随之变严**(更早落盘/分块),
既有签名与 `set_memmap_ratio` / `set_absolute_limit` 保持不变。

### 5.3 已知残留局限

- `should_use_memmap(nbytes)` 只知道**这一次**要分配多少; 算法侧的工作集(镜像/模态谱)仍需
  调用方用 `extra=` 传进来, 否则输入层仍会低估 —— 例如 1 GB 输入配 K=3 的模态谱,
  输入层只看到 1 GB。**建议**后续把 `_project_bytes` 类估算接入 `Utils.Check`。
- 预算闸门是"跑前一次判定"; 长任务中途内存被别的进程抢走仍会退化(此时只能靠 OS 换页)。

---

## 6. 复现方式

```powershell
# 小尺寸(秒级)
python tests/comparison/bench_fft.py --sizes 1 10 --with-fft

# 大尺寸(分钟级; 逐条带 3.6GB 提交上限与预算闸门; reps=3 才能同时看到 cold 与 warm)
python tests/comparison/bench_fft.py --sizes 100 500 1024 --with-fft --reps 3 --commit-cap-gb 3.6 --time-cap-s 600

# 只补 pyfftw (批量场景评估)
python tests/comparison/bench_fft.py --sizes 1 10 100 500 --configs pyfftw-w1 pyfftw-w8 pyfftw-all --reps 3

# 单条
python tests/comparison/bench_fft.py --worker '{"config":"numpy","size_mb":100,"reps":1}'
```

结果**合并写入** `docs/FFT_Backend_Results.json`(键 = config + size + op), 因此分多次跑不会互相覆盖。

---

## 7. 局限与后续

1. **pyfftw 的 ≥500 MB 反转可能是"内存压力的产物"**: 本机可用内存 3–8 GB, 其多线程工作区
   先触发换页。**在 ≥32 GB 的机器上复测**才能判断"大数组到底谁赢"。
2. **`pyfftw.interfaces.cache` 的容量未调**: 默认缓存条目有限, 形状很多时会反复重新规划;
   批量场景应显式 `set_cache_size`(未测)。
3. **`workers` 只在 scipy 上测了 1/4/8/全核**(差异 <10%), pyfftw 侧测了 1/8/全核
   (差异 6×); MEASURE 规划除 1M 点外未跑完。
4. **cupy 只测到 100 MB**: 已能判定"传输主导、全尺寸慢于 numpy"; 若真要用 GPU, 需要把设备
   数组留在显存(改返回契约)才可能见效; 未测显存峰值。
5. **float32 路径未单独测**: NumPy 2.x 与 pyfftw 的 `rfft(float32) → complex64` 都能省一半内存
   (pyfftw 实测返回 complex64 ✓), 对 1 GB 级输入的意义大于换后端; 建议后续补测
   `{float32, float64} × {numpy, pyfftw}`。
6. `tiled` 目前对 `rfft` 回落直算(已在 docstring 与 §4 说明原因); 真正需要"超内存 FFT"时,
   方向是**外存 + 分块频谱读写**, 而不是四步分解。
7. 基准脚本的提交内存上限是**采样式**的(父进程 50 ms 轮询 + worker 内 5 ms 采样线程),
   不是硬限: 1 GB 行实测峰值 4106 MB 超过了 3.2 GB 的设定值而未被拦下。要硬限需要
   Windows Job Object / cgroup, 本轮未做 —— 使用大尺寸时请自行确认机器余量。

---

## 8. 本轮顺带发现(未修, 待决策)

`tests/test_utils.py` 有 **5 个既有失败**(与本次改动无关, `git status` 显示 `Utils/Check.py`
未被修改): `Check_Time_and_Signal` 的 `_to_default_float` / `_to_keep_dtype` 只作用于 `T`,
**从未作用于 `S`**, 因此文档承诺的 `dtype="float64"` 转换、`squeeze`、非数值 dtype 拒绝
都没有实现:

```
test_check_time_and_signal_non_f64_large_goes_disk   # 期望 memmap, 实际拿到 f32 原数组
test_check_explicit_float64_converts                 # 期望 f64, 实际 f32
test_check_dtype_invalid_raises                      # 期望 ValueError, 实际不报
test_check_dtype_keep_squeezes_and_keeps_dtype       # 期望 squeeze, 实际 (1,N)
test_check_dtype_keep_rejects_non_numeric            # 期望 ValueError, 实际不报
```

修法是让 `Check_Time_and_Signal` 按 `dtype` 走 `_to_default_float(S)` / `_to_keep_dtype(S)`,
但这会改变全部 15 个方法的输入行为(它们目前都不传 `dtype`, 即走"保留 dtype 且不 squeeze"),
属于行为变更, 需单独决策。
