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

## 6. 深度测试（本轮）

> 测试：`tests/utils/test_Hilbert_deep.py`（**27 条**，全绿）；数据：
> `table/utils/hilbert/{backend_agreement,contract_matrix,backend_timing,zeropad_artefact}.csv`；
> 图：`fig/utils/hilbert/hilbert_backends.png`。

### 6.1 后端契约矩阵（本轮的实测表）

| 输入/参数 | `"FFT"` | `"FHT"` | `"scipy"` |
|---|---|---|---|
| 1-D 任意长度（1,2,3,7,…,10000） | 精确 | 精确 | 基线 |
| 2-D | `ValueError` | `ValueError` | **接受**（沿末轴） |
| 空输入 | 返回空 | 返回空 | **`ValueError: N must be positive`** |
| 未知 kwarg | `TypeError` | `TypeError` | **静默忽略** |
| `matlab_phase=False` | 生效（共轭） | 生效（共轭） | **静默忽略** |
| 复数输入 | `ComplexWarning` + 丢虚部 | 同左 | 同左 |

* 与 scipy 的一致性：所有尺寸 `max|z−scipy| ≤ 1e-9 × scale`（pow2 上 ~1e-15）；
* `matlab_phase=False` ≡ `conj(True)`（FFT/FHT 两者都成立，≤1e-12）；
* 未实现的 6 个声明名（`KK`/`FIR`/`HST`/`SB-Hilbert`/`Fourth-order Wave`/`Optional-order FIR`）均 `ValueError`；
* FHT 原语：`fht_forward`/`fht_inverse` 在 pow2 上往返误差 ≤1e-12；非 pow2 零填充到下一个 pow2
  （`DHT` 定义用 O(N²) 直算逐点核对）；`hilbert_transform` 的零填充伪影相对 L2 约 1e-3–1e-1（§6.3）。

### 6.2 发现

| 编号 | 现象 | 期望 |
|---|---|---|
| **F1** | 三个后端的**契约不统一**（见上表：2-D/空输入/未知 kwarg/`matlab_phase`），调用方无法"换后端不改代码" | 要么统一在 `hilbert()` 入口做校验/归一化，要么在 docstring 里给出这张矩阵 |
| **F2** | `"scipy"` 后端把 `matlab_phase` 静默吞掉（`_scipy_backend._hilbert(**kwargs)` 只读 `verbose`）⇒ 传 `False` 也拿不到共轭 | `matlab_phase=False` 时对结果取共轭，或明确报 `TypeError` |
| **F3** | 文档自相矛盾：模块 docstring 说 FHT"**power-of-two lengths only**"，而 `_hilbert` 实际是**自适应分发**（pow2 且 ≤512 走 SAO 内核，其余走实半谱），任意长度都精确 | 以自适应为准，改掉"only"的措辞 |
| **F4** | **无输入校验**：NaN 直接透传（结果 NaN）、不做长度上限、只把 dtype 强制成 float64 | 与 `Check` 的分层保持一致（或至少在 docstring 声明"本函数不校验"） |
| **F5** | `hilbert_transform`（原语）默认 `matlab_phase=False`（SAO 的 +90°），而 `hilbert()` 默认 `True`；两者默认相反，容易混用 | 文档已在两处写明；建议给原语改成 `matlab_phase=True` 或改名 |

### 6.3 零填充伪影（原语 `hilbert_transform`）

| n | 100 | 255 | 256 | 512 | 513 | 1000 | 1024 | 4000 |
|---|---|---|---|---|---|---|---|---|
| pow2 | ✗ | ✗ | ✓ | ✓ | ✗ | ✗ | ✓ | ✗ |
| 相对 L2 vs `scipy` | 5.3e-2 | **1.1e-1** | 5.1e-16 | 5.6e-16 | 3.3e-2 | 1.7e-2 | 6.7e-16 | 1.2e-2 |

（数据：`table/utils/hilbert/zeropad_artefact.csv`。）**2 的幂上 1e-16 量级，非 2 的幂上 1e-2–1e-1**
—— 最大 11.4%（n=255，零填充到 256）。这正是 `_hilbert` 只在小尺寸 pow2 上使用 SAO 内核、
其余长度走实半谱的原因。

### 6.4 复现

```powershell
.\.venv\Scripts\python.exe -m pytest tests/utils/test_Hilbert_deep.py -q
.\.venv\Scripts\python.exe tests/utils/test_Hilbert_deep.py --export
.\.venv\Scripts\python.exe tests/size_axis/run_scale.py --tool hilbert --export
```
