# `Utils.Envelope` 模块报告

> 由 `.scratch/eemd_new/gen_tool_doc.py` 从**当前代码内省**生成；契约与 `tests/utils/test_Envelope.py` 一一对应。

## 1. 模块定位

Envelope extraction (包络提取工具)。

## 2. 公开 API

| 名字 | 签名 | 说明 |
|---|---|---|
| `envelope` | `(S: numpy.ndarray, method: Union[Literal['Hilbert', 'Lowpass', 'IQ', 'PeakInterpolation'], Modal_Decomposition.Utils.Envelope.ENVELOPE_MODS] = 'Hilbert', fs: float = None, fc: float = None, cf: float = None, spline: Callable = None, **kwargs) -> numpy.ndarray` | Compute the amplitude envelope of a 1-D signal. |

## 3. 枚举

| 枚举 | 成员 |
|---|---|
| `ENVELOPE_MODS` | Hilbert / Lowpass / IQ / Peak |

## 4. 契约（由测试验证）

- 可导入且 `__all__` 与实际导出一致
- `mod` / `method` 等参数同时接受字符串与 Enum 成员
- 非法 `mod` / `backend` 抛 `ValueError`
- 返回值契约（形状 / dtype / 复数性 / 元组长度）
- 与参考实现（scipy / numpy）数值一致
- 边界输入不崩（空 / 单点 / 常量 / NaN）
- 确定性（同输入两次逐位相同）且不改写调用方输入

## 5. 复现

```
.\.venv\Scripts\python.exe -m pytest tests/utils/test_Envelope.py -q
```

`__all__` 共 3 项。

## 6. 深度测试（本轮）

> 测试：`tests/utils/test_Envelope_deep.py`（**20 条**，全绿）；数据：
> `table/utils/envelope/envelope_modes.csv`；图：`fig/utils/envelope/envelope_modes.png`。

### 6.1 四模式定量对比（重算 docstring 那张表）

`fs=10 kHz, fc=1 kHz, cf=200 Hz, N=2^14`，误差对**真实包络**（两端各裁 200 点）：

| 用例 | Hilbert | Lowpass | IQ | PeakInterpolation |
|---|---|---|---|---|
| clean AM（1+0.4cos 50 Hz） | 4.7e-4 | 1.7e-2 | **4e-6** | **5e-6** |
| deep AM（0.5+0.5cos 10 Hz） | 1.5e-4 | 1.0e-2 | **1e-6** | **3e-6** |
| two-tone AM（30+70 Hz） | 4.2e-4 | 1.7e-2 | **3.2e-5** | **9e-6** |
| noisy AM（σ=0.05） | 5.0e-2 | 2.4e-2 | **1.4e-2** | 7.5e-2 |

* 干净信号上 **IQ ≈ Peak < Hilbert < Lowpass**；
* 噪声下排序变成 **IQ < Lowpass < Hilbert < Peak** —— docstring 的"Lowpass 优于
  Hilbert（0.026 vs 0.051）"成立（实测 0.024 vs 0.050），但**真正最好的是 IQ**，
  而 Peak 因伪峰最差；
* 每模式耗时（N=2^14）：Hilbert 0.35 ms、Lowpass 0.44 ms、IQ 0.98 ms、Peak 0.48 ms。

### 6.2 发现

| 编号 | 现象 | 期望 |
|---|---|---|
| **F1（真 bug）·已修** | `hilbert_mod` 被透传进 `hilbert(S, mod=hilbert_mod, **kwargs)`，于是 `"FFT"`/`"FHT"` 撞上 `_hilbert() got an unexpected keyword argument` ⇒ **Hilbert 后端无法切换**，实际只能用 `"scipy"`。规模轴上体现为 600 MiB **12.7 s / 4.8 GiB**，而 `hilbert(S,"FFT")` 只要 3.7 s / 2.4 GiB | **已修**：`peak_mod` / `hilbert_mod` 改成 `kwargs.pop(...)`，不再往下透传。回归见 `test_hilbert_backend_switch_works`（scipy/FFT/FHT 三者一致，1e-9 内） |
| **F2 ·已修** | `peak_mod` 同理：`envelope(S, method="PeakInterpolation", peak_mod="numpy")` → `CubicSpline.__init__() got an unexpected keyword argument 'peak_mod'` | **已修**：同 F1；回归见 `test_peak_mod_kwarg_no_longer_leaks` |
| **F3** | `fs/fc/cf` 的校验只在**三者全为 None** 时触发：`Lowpass(cf=200)` 少给 `fs` → 在 `_lowpass` 里 `0.5*fs` 抛 `TypeError`（不是清晰的 `ValueError`） | 按模式分别校验必需参数 |
| **F4** | 用枚举传 `method` 时 `method in ["Lowpass","IQ"]` 恒 False ⇒ 参数校验被完全跳过 | 先归一化枚举再校验 |
| **F5** | 枚举成员叫 `Peak`，但字符串值必须是 `"PeakInterpolation"`；传 `"Peak"` 报 `Unknown envelope method` | 文档写明，或接受别名 |
| **F6** | 除 `not_numpy_arr`/`not_numpy_memmap` 外**无任何输入校验**：NaN 在 Hilbert 模式透传（输出 NaN），在 Peak 模式才因 `find_peaks` 抛错；空输入在 Hilbert 模式由 scipy 抛 `N must be positive` | 统一在入口做有限性检查 |

### 6.3 复现

```powershell
.\.venv\Scripts\python.exe -m pytest tests/utils/test_Envelope_deep.py -q
.\.venv\Scripts\python.exe tests/utils/test_Envelope_deep.py --export
.\.venv\Scripts\python.exe tests/size_axis/run_scale.py --tool envelope --export
```
