# `Utils.Poly` 模块报告

> 由 `.scratch/eemd_new/gen_tool_doc.py` 从**当前代码内省**生成；契约与 `tests/utils/test_Poly.py` 一一对应。

## 1. 模块定位



## 2. 公开 API

| 名字 | 签名 | 说明 |
|---|---|---|
| `fast_poly` | `(x_ref: numpy.ndarray, y_ref: numpy.ndarray, poly_order: int, x_new: numpy.ndarray, safe: Literal['safe', 'suspicious', 'unknown'] = 'unknown') -> numpy.ndarray` | 在 ``(x_ref, y_ref)`` 上拟合 ``poly_order`` 次多项式, 求 ``x_new`` 处的值。 |

## 3. 契约（由测试验证）

- 可导入且 `__all__` 与实际导出一致
- `mod` / `method` 等参数同时接受字符串与 Enum 成员
- 非法 `mod` / `backend` 抛 `ValueError`
- 返回值契约（形状 / dtype / 复数性 / 元组长度）
- 与参考实现（scipy / numpy）数值一致
- 边界输入不崩（空 / 单点 / 常量 / NaN）
- 确定性（同输入两次逐位相同）且不改写调用方输入

## 4. 复现

```
.\.venv\Scripts\python.exe -m pytest tests/utils/test_Poly.py -q
```

`__all__` 共 1 项。
