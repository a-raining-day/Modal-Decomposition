# `Utils.Peaks` 模块报告

> 由 `.scratch/eemd_new/gen_tool_doc.py` 从**当前代码内省**生成；契约与 `tests/utils/test_Peaks.py` 一一对应。

## 1. 模块定位

Peak detection (统一峰检测)。

## 2. 公开 API

| 名字 | 签名 | 说明 |
|---|---|---|
| `find_peaks` | `(S: numpy.ndarray, mod: Union[Literal['max', 'min', 'both'], Modal_Decomposition.Utils.Peaks.PEAKS_MODS] = 'max', backend: Union[Literal['scipy', 'numpy', 'numba'], Modal_Decomposition.Utils.Peaks.PEAKS_BACKEND] = 'numpy', safe: Union[Literal['safe', 'unknown'], Modal_Decomposition.Define.ConstDefine.SAFE_MODS] = 'unknown', **kwargs) -> tuple[numpy.ndarray, numpy.ndarray] | tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]` | Find peaks in a 1-D signal. |

## 3. 枚举

| 枚举 | 成员 |
|---|---|
| `PEAKS_BACKEND` | Scipy / Numpy / Numba |
| `PEAKS_MODS` | Max / Min / Both |
| `PEAKS_SUPPORTED_FILTERS` | Height / Threshold / Distance |

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
.\.venv\Scripts\python.exe -m pytest tests/utils/test_Peaks.py -q
```

`__all__` 共 7 项。

## 6. 深度测试（本轮）

> 测试：`tests/utils/test_Peaks_deep.py`（**39 条**，全绿）；数据：
> `table/utils/peaks/{backend_agreement,peak_counts,backend_timing}.csv`；
> 图：`fig/utils/peaks/peaks_backends.png`。

### 6.1 三后端一致性（本轮核心）

| 信号 | numpy vs numba | numpy vs scipy |
|---|---|---|
| smooth / random / constant / ramp | **逐位相同** | **逐位相同** |
| plateau（人工平台） | 逐位相同 | **不同**（numpy 取平台左沿，scipy 取中点） |

* 过滤器（`height` 标量/区间、`threshold`、`distance`）在 numpy / scipy / numba 上
  逐一比对：**全部一致**（含 `mod="min"` 的镜像语义）；
* `both` 的 `distance` 按类型分别施加（极大只与极大比距离），与 docstring 一致；
* `numba` 内核每进程只编译一次（`lru_cache` 计数验证）。

### 6.2 发现

| 编号 | 现象 | 期望 |
|---|---|---|
| **F1（陷阱）** | numpy/numba **静默忽略**未知过滤键，scipy 会真的施加。`plateau_size=2` 时 numpy 返回全部 9 个峰、scipy 返回 0 个 —— 同一调用换个后端结果就不同 | 对未识别的键发 `UserWarning`（或显式白名单校验） |
| **F2** | 平台规则不同：numpy/numba 用 `S[i-1] < S[i] >= S[i+1]`（取**左沿**），scipy 取**平台中点**；docstring 已声明，但没有任何提示/告警 | 同 F1，或提供 `plateau="scipy"` 开关 |
| **F3** | `safe="suspicious"` 明确被拒（`ValueError`），但错误信息只说 "the `suspicious` mod is not supported in this function" 而不说该用哪个 | 文案补上 `"safe"/"unknown"` |
| **F4** | 严格无意义的键（`no_such_filter=1`）在 numpy/numba 下静默通过、scipy 下 `TypeError` | 同 F1 |

### 6.3 正常面

* 返回契约：`int64` 下标、`values is S[idx]`（`mod="min"` 也不取负）、`both` 归并后严格升序、
  掩码与下标等长且"极大子集 == 单独 `mod='max'` 的结果"；
* 短输入（size 0/1/2）：单峰型返回全部下标、`both` 返回空集（docstring 的语义已实现）；
* 常量/斜坡 → 0 峰；NaN 在 `safe="unknown"` 下 `ValueError`、在 `safe="safe"` 下不成峰；
* scipy/numba 缺失 → `UserWarning` + 回退 numpy（monkeypatch 验证）；
* 规模轴：600 MiB 上 `numpy_max` 176 ms、`both` 324 ms、`scipy_max` 156 ms、`numba_both` 156 ms。

### 6.4 复现

```powershell
.\.venv\Scripts\python.exe -m pytest tests/utils/test_Peaks_deep.py -q
.\.venv\Scripts\python.exe tests/utils/test_Peaks_deep.py --export
.\.venv\Scripts\python.exe tests/size_axis/run_scale.py --tool peaks --export
```
