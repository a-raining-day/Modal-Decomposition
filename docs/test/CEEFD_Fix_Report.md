# CEEFD 缺陷修复报告

- **实现**: `src/Modal_Decomposition/CEEFD.py`
- **测试**: `tests/test_ceefd.py` (**92 passed**, 0.67 s)
- **修复日期**: 2026-09-16 · Python 3.10 · numpy 2.2.6 · scipy 1.15.3

---

## 1. 结论摘要

修复前 **`CEEFD` 完全不可用** —— 它在任何 `N > 37` 的信号上抛 `TypeError`,
在 `N ≤ 37` 上抛 `ValueError`。也就是说 `tests/_cases.py` 里为它配的那条用例
从来就跑不通, 而 `CEEFDConfig.fs` 是个从不参与计算的死参数。

修复后: 全长度区间可跑、`ΣIMFs + Res == S` 精确成立、参数校验到位、
`fs` 有了实际作用。

| # | 缺陷 | 修复前症状 | 状态 |
|---|---|---|---|
| 1 | 频带下标是 `numpy.float64` 却喂给内建 `range()` | **`N > 37` 全部抛 `TypeError`** | 已修 |
| 2 | 谱包络盒式窗长裸用 `int(0.05*n)` | `N < 20` 得空窗 → `ValueError: v cannot be empty`; 部分 `N` 下窗长非整型 → `TypeError` | 已修 |
| 3 | `len(peaks) == 0` 分支不可达 | 死代码 (前面已 `return`) | 已删 |
| 4 | `fs` 被接受但完全不参与计算 | 死参数 (实测 `fs=1` 与 `fs=1e5` 逐位相同) | 已赋予作用 |
| 5 | 无参数字段校验 | `envelop_iter=-1`、`min_peak_distance=-5` 等一律静默接受, 到 `decompose` 才炸 | 已加 |
| 6 | 零峰早退契约不一致 | 返回 `IMFs = S.reshape(1,-1)` + `Res = 0` (与库内其余方法矛盾) | 已统一 |
| 7 | 每模态重算一次 FFT | `_extract_imf` 内 `np.fft.fft(signal)` 对每个频带重复 | 已消除 |
| 8 | 频带 bin 约定半开, 覆盖率不闭合 | 潜在能量遗漏 | 已改为闭区间 |

## 2. 缺陷 1: 全长度崩溃 (最严重)

`CEEFD.py` 原第 106 行:

```python
boundaries = np.zeros(len(peaks) + 1)          # <- float64 数组
...
boundaries[1:-1] = (peaks[1:] + peaks[:-1]) // 2
...
freq_bins = list(range(start_bin, end_bin))    # <- start_bin 是 numpy.float64
```

`range()` 只接受 Python 整数 (或实现 `__index__` 的类型)。`numpy.float64`
**不实现** `__index__`, 于是:

```
TypeError: 'numpy.float64' object cannot be interpreted as an integer
```

实测崩溃范围 (修复前):

| N | 结果 |
|---:|---|
| 8, 16, 19, 20, 21 | `ValueError: v cannot be empty` (窗长为 0) |
| 37, 38, 39, 40, 41, 64, 128, 256, 512 | `TypeError: 'numpy.float64' object ...` |

**即只要信号长到能通过窗长检查, 就必然走进 `range()` 而崩。** 这个方法是死的。

修复: `boundaries` 直接建成整型数组 (`np.empty(..., dtype=np.int64)`),
并用 `np.arange` 取代 `list(range(...))`。

## 3. 缺陷 2: 谱包络窗长

原实现:

```python
window_size = int(0.05 * n)
window = windows.boxcar(window_size)
```

两个独立故障:

- `n < 20` 时 `window_size == 0` → `windows.boxcar(0)` 返回**空窗** →
  `np.convolve` 抛 `ValueError: v cannot be empty`;
- `0.05 * n` 是二进制浮点乘法, 某些 `n` 下结果带误差, `int()` 之后仍可能
  不是"干净的"整数 —— 与缺陷 1 叠加后症状更难定位。

修复: `window_size = max(1, int(_ENVELOP_WINDOW_FRACTION * n))`, 并把窗
**归一化** (`window / window.sum()`) 以保持与旧实现 `/ window_size` 等价的
加权平均语义。同时把比例提为模块常量 `_ENVELOP_WINDOW_FRACTION`, 不再写死。

## 4. 缺陷 3–8

### 4.1 死代码

```python
if len(peaks) == 0:
    return ...          # 这里已经返回

if len(peaks) == 0:     # <- 永远不可达
    boundaries = np.array([0, N // 2])
```

第二段是死代码, 已删。

### 4.2 `fs` 死参数

修复前 `fs` 只被存进 `self.fs` 与配置快照, 不参与任何计算 —— 实测
`CEEFD(fs=1.0)` 与 `CEEFD(fs=1e5)` 输出逐位相同。

`CEEFD` 的分割本身在**归一化频率**上进行 (边界就是 `0 … N//2` 的 bin 下标),
这是算法性质, 不应改变。故修复方式是**赋予它诊断作用**: 新增
`info["boundaries_hz"] = boundaries * (fs / N)`, 把 bin 下标换算成物理频率。

```
fs=1000, N=1024 -> boundaries_hz = [0, 41, 91.8, 127, ..., 500]
```

`hz[-1] == fs/2` (Nyquist) 是自检点, 已写成断言。

> 说明: 没有把 `fs` 改成"参与分割"——那会改变算法语义。若你希望分割直接在 Hz
> 轴上做 (例如支持非均匀频率轴), 那是**功能变更**而非缺陷修复, 需另行定夺。

### 4.3 参数校验

修复前全部静默接受, 直到 `decompose` 内部才炸:

| 参数 | 修复前 | 修复后 |
|---|---|---|
| `fs` | `0` / `-1` / `nan` / `inf` / `True` / `"1000"` 全接受 | 必须有限且 > 0, 拒绝 bool 与非数值 |
| `min_peak_distance` | `0` / `-5` / `1.5` 全接受 | 必须为 `int >= 1` |
| `envelop_iter` | `-1` / `1.5` 全接受 | 必须为 `int >= 0` |

### 4.4 零峰早退契约

修复前:

```python
return DecompositionResult(S.reshape(1, -1), np.zeros_like(S), ...)
```

把整条信号当成**一个模态**并令残差为 0。这与库内契约冲突 (`Res` 应是
`S − ΣIMFs` 的余量, 且"无可分解内容"应表现为 0 个模态)。修复后:

```python
IMFs = np.empty((0, N)); Re = S.copy()
```

与 `EMD` / `LMD` / `FMD` 在无内容输入上的行为一致。

### 4.5 重复 FFT

`_extract_imf` 内部对每个频带都重算一次 `np.fft.fft(signal)`, 而外层已有
`fft_signal`。现已把 `fft_signal` 直接传入 `_extract_mode`, 只算一次。

### 4.6 频带 bin 约定 (闭区间)

原实现用半开区间, 各带覆盖 `[a, b)`。为保证"各带恰好覆盖 `0 … N//2` 一次,
既无重复也无遗漏", 现改为**闭区间** `[a, b]` (末带含 Nyquist), 且 DC 与
Nyquist 作为自共轭点只计一次 (掩码用 `(N - bins) % N`, 在 `k=0` 与 `k=N//2`
时回落到自身, 天然不重复)。

实测该约定下重构精确:

| N | 形态 | K | 重构误差 |
|---:|---|---:|---:|
| 4 / 8 / 16 / 19 / 20 / 21 | 短信号 | 1 | 0.00e+00 |
| 37 / 38 / 39 / 41 | 短信号 | 2 | 0.00e+00 |
| 40 / 64 / 1000 / 1001 | | 2–37 | ≤ 1.39e-17 |
| 128 / 256 | | 4 / 9 | ≤ 2.78e-17 |
| 512 (纯噪声) | | 16 | 3.47e-17 |

## 5. 验收

```
tests/test_ceefd.py                     92 passed (0.67 s)
  · 17 个长度区间不再崩溃 (含修复前两段崩溃区)
  · 23 个长度的包络窗非退化
  · 重构精确性 (含 DC 分量) ≤ 1e-9
  · 频带精确覆盖 0…N//2 一次
  · 13 组非法参数全部 ValueError / 4 组合法边界值接受
  · fs 映射到物理频率 (hz[-1] == fs/2) 且不改变模态
  · facade 等价 / 确定性 / 不改写输入 / 多通道拒绝
```

## 6. 未改动 (属算法行为, 非缺陷)

`CEEFD` 在默认参数下**分得偏碎**: 512 点双音信号给出 17 个模态。原因是
`min_peak_distance=10` 与 `envelop_iter=3` 的默认组合在平滑后的包络上仍允许
较多峰。这是算法参数选择的后果, 不是崩溃类缺陷, 故本次**未改动默认值**
(改动会改变所有既有结果)。若需要更粗的分割: 增大 `min_peak_distance` 或
`envelop_iter` (实测 `envelop_iter=0/3/10` → K=20/17/9, 单调递减)。

## 7. 复现

```bash
python -m pytest tests/test_ceefd.py -q

python -c "
import numpy as np
from Modal_Decomposition import Class
for n in (19, 40, 256, 1000):
    S = np.sin(2*np.pi*5*np.linspace(0,1,n,endpoint=False)) + 0.3*np.random.default_rng(0).standard_normal(n)
    r = Class.CEEFD(fs=1000.0).decompose(S)
    print(n, r.IMFs.shape, float(np.max(np.abs(r.reconstruct()-S))))
"
```
