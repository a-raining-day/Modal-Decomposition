# `Utils.Monotonicity` 模块报告

> 由 `.scratch/eemd_new/gen_tool_doc.py` 从**当前代码内省**生成；契约与 `tests/utils/test_Monotonicity.py` 一一对应。

## 1. 模块定位

Monotonicity checks.

## 2. 公开 API

| 名字 | 签名 | 说明 |
|---|---|---|
| `give_monotony` | `(arr, safe: Union[Literal['safe', 'suspicious', 'unknown'], Modal_Decomposition.Define.ConstDefine.SAFE_MODS] = 'unknown', chunk_size: int | None = None, chunk_mod: Optional[Literal['memory', 'exo-memory']] = None) -> Modal_Decomposition.Utils.Monotonicity.Monotony` | Classify a 1-d array into a ``Monotony`` member. |
| `monotonic` | `(arr, strict: bool = False, mod: Union[Literal['increasing', 'decreasing', 'monotonic'], Modal_Decomposition.Utils.Monotonicity.MONOTONY_MODS] = 'monotonic', safe: Union[Literal['safe', 'suspicious', 'unknown'], Modal_Decomposition.Define.ConstDefine.SAFE_MODS] = 'unknown', chunk_size: int | None = None, chunk_mod: Optional[Literal['memory', 'exo-memory']] = None) -> bool` | Check whether a 1-d array is monotonic. |

## 3. 枚举

| 枚举 | 成员 |
|---|---|
| `MONOTONY_MODS` | Increasing / Decreasing / Monotonic |
| `Monotony` | Equal / MixedIncreasing / MixedDecreasing / StrictIncreasing / StrictDecreasing / No |

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
.\.venv\Scripts\python.exe -m pytest tests/utils/test_Monotonicity.py -q
```

`__all__` 共 5 项。
