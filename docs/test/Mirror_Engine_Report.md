# `Utils.Mirror` 模块报告

> 由 `.scratch/eemd_new/gen_tool_doc.py` 从**当前代码内省**生成；契约与 `tests/utils/test_Mirror.py` 一一对应。

## 1. 模块定位

Extrema endpoint mirror extension (端点镜像外拓, 全库共享)。

## 2. 公开 API

| 名字 | 签名 | 说明 |
|---|---|---|
| `mirror` | `(S: numpy.ndarray | numpy.memmap, T: numpy.ndarray = None, mod: Union[Literal['mirror', 'extrema', 'polynomial', 'parallel', 'BLCSE'], Modal_Decomposition.Utils.Mirror.MIRROR_MODS] = 'mirror', nbsym: int = 3, poly_order: int = 2, time_poly_order: int = 2, *, verbose: bool = False, peaks_backend: Union[Literal['scipy', 'numpy', 'numba'], Modal_Decomposition.Utils.Peaks.PEAKS_BACKEND] = 'numpy', safe: Union[Literal['safe', 'suspicious', 'unknown'], Modal_Decomposition.Define.ConstDefine.SAFE_MODS] = 'unknown', **kwargs) -> numpy.ndarray | tuple[numpy.ndarray, numpy.ndarray]` | 信号端点外拓: 端点镜像, 或"检出极值 → 外延极值表"。 |

## 3. 枚举

| 枚举 | 成员 |
|---|---|
| `MIRROR_MODS` | Mirror / Extrema / Polynomial / Parallel / BLCSE |

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
.\.venv\Scripts\python.exe -m pytest tests/utils/test_Mirror.py -q
```

`__all__` 共 3 项。
