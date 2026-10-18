# `Utils.Slepian` 模块报告

> 由 `.scratch/eemd_new/gen_tool_doc.py` 从**当前代码内省**生成；契约与 `tests/utils/test_Slepian.py` 一一对应。

## 1. 模块定位

Slepian (DPSS) 序列生成的统一入口 —— 三后端分发 (numpy / scipy / C)。

## 2. 公开 API

| 名字 | 签名 | 说明 |
|---|---|---|
| `available_backends` | `() -> dict` | 各后端可用性: ``{"numpy": True, "scipy": True/False, "C": True/False}``。 |
| `cache_info` | `() -> dict` | 缓存统计: ``{version, size, bytes, hits, misses, evictions, skipped}``。 |
| `clear_cache` | `(reset_stats: bool = True) -> None` | 清空缓存 (并可重置统计)。 |
| `resolve_backend` | `(N: int, mod: Literal['numpy', 'scipy', 'C', 'auto'] = 'auto', halfBW: float = None) -> str` | 解析生效后端名。 |
| `slepian` | `(N: int, halfBW: float, nTapers: int = None, mod: Literal['auto', 'numpy', 'scipy', 'C'] = 'auto', return_ratios: bool = False, norm=None, sym: bool = True, cache=None) -> numpy.ndarray` | 生成 Slepian (DPSS) 序列 —— 统一入口。 |

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
.\.venv\Scripts\python.exe -m pytest tests/utils/test_Slepian.py -q
```

`__all__` 共 5 项。
