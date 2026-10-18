# `Utils.Hilbert` 模块报告

> 由 `.scratch/eemd_new/gen_tool_doc.py` 从**当前代码内省**生成；契约与 `tests/utils/test_Hilbert.py` 一一对应。

## 1. 模块定位

Hilbert backends — thin wrapper module.

## 2. 公开 API

| 名字 | 签名 | 说明 |
|---|---|---|
| `fht_forward` | `(x, normalize_order=True)` | 前向 FHT。 |
| `fht_inverse` | `(x, input_is_normal_order=True)` | 逆 FHT。 |
| `hilbert` | `(S: numpy.ndarray, mod: Union[Literal['scipy', 'Fourth-order Wave', 'FHT', 'SB-Hilbert', 'Kramers-Kronig', 'FIR', 'HST', 'FFT', 'Optional-order FIR'], Modal_Decomposition.Utils._Hilbert.HILBERT_MODS] = 'FFT', **kwargs) -> numpy.ndarray[..., numpy.complexfloating]` | Hilbert transform of a real signal with a selectable spline_kind. |
| `hilbert_transform` | `(real_signal, matlab_phase=False)` | 计算实信号的 Hilbert 变换，返回解析信号 (实部=原信号, 虚部=Hilbert 变换)。 |

## 3. 枚举

| 枚举 | 成员 |
|---|---|
| `HILBERT_MODS` | Scipy / FW / FHT / SBH / KK / FIR / HST / FFT / Oo_FIR |

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
.\.venv\Scripts\python.exe -m pytest tests/utils/test_Hilbert.py -q
```

`__all__` 共 6 项。
