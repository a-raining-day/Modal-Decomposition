# `Base.Seed` 模块报告

> 由 `.scratch/eemd_new/gen_tool_doc.py` 从**当前代码内省**生成；契约与 `tests/base/test_Seed.py` 一一对应。

## 1. 模块定位

Global Seed

## 2. 公开 API

| 名字 | 签名 | 说明 |
|---|---|---|
| `Seed` | `(*args, **kwargs)` | 全局仅该唯一类可以设置全局种子，该种子对全局覆盖且唯一 |
| `Seed.get` | `() -> int` |  |
| `Seed.set` | `(seed: int, lock: bool = False, verbose: bool = True) -> int` | Set the seed. |
| `Seed.temporary_seed` | `(seed: int) -> Generator[NoneType, NoneType, NoneType]` | 在多线程并发时，该窗口会带来隐式的复现性破坏。虽然实现了，但是不建议使用 |
| `Seed.validate` | `(seed: int) -> int` |  |

## 3. 契约（由测试验证）

- 可导入；公开符号能从 `Utils` 顶层取到（`Utils` 包本身无 `__all__`）
- 公开可调用对象的签名带合理默认值（无必填隐藏参数）
- 枚举/Literal 参数非法值抛 `ValueError`
- 功能正确性（各模块在此文件里各自展开）
- 边界输入不崩（空 / 极短 / 极端尺寸）
- 副作用可控（测试完恢复全局状态）

## 4. 复现

```
.\.venv\Scripts\python.exe -m pytest tests/base/test_Seed.py -q
```

`__all__` 共 2 项。
