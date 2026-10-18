# `Base.Cache` 模块报告

> 由 `.scratch/eemd_new/gen_tool_doc.py` 从**当前代码内省**生成；契约与 `tests/base/test_Cache.py` 一一对应。

## 1. 模块定位

store of the import cache

## 2. 公开 API

| 名字 | 签名 | 说明 |
|---|---|---|
| `Cache` | `()` | Import cache: unique-name registry of imported modules. |
| `Cache.add` | `(name, api, verbose: bool = False, description: Optional[str] = None) -> bool` | add the api of the module or function or class to the cache |
| `Cache.check` | `(name) -> bool` | return the result of the query to cache |
| `Cache.clear` | `() -> None` | Remove every registered entry. |
| `Cache.describe` | `(name) -> str` | Return the description of a registered module. |
| `Cache.descriptions` | `() -> Dict[str, str]` | Snapshot dict of ``{name: description}``. |
| `Cache.get` | `(name)` | return the cached api |
| `Cache.import_module` | `(name: str, description: Optional[str] = None, verbose: bool = False)` | 抽象并集中管理经典惰性导入样板:: |
| `Cache.items` | `() -> List[Tuple[str, Any]]` | List of ``(name, api)`` pairs in sorted-key order. |
| `Cache.keys` | `() -> List[str]` | Alias of :meth:`cache.names`. |
| `Cache.lazy` | `(name: str, description: Optional[str] = None, verbose: bool = False)` | 装饰器: 把零参加载函数变成"进程内恰好执行一次"的惰性取用器。 |
| `Cache.names` | `() -> List[str]` | Sorted list of all registered keys. |
| `Cache.remove` | `(name) -> bool` | Remove one entry (api and description). |
| `Cache.values` | `() -> List[Any]` | List of cached apis in sorted-key order. |
| `cache` | `()` | (别名, 同 `Cache`) |
| `cache.add` | `(name, api, verbose: bool = False, description: Optional[str] = None) -> bool` | (别名, 同 `Cache.add`) |
| `cache.check` | `(name) -> bool` | (别名, 同 `Cache.check`) |
| `cache.clear` | `() -> None` | (别名, 同 `Cache.clear`) |
| `cache.describe` | `(name) -> str` | (别名, 同 `Cache.describe`) |
| `cache.descriptions` | `() -> Dict[str, str]` | (别名, 同 `Cache.descriptions`) |
| `cache.get` | `(name)` | (别名, 同 `Cache.get`) |
| `cache.import_module` | `(name: str, description: Optional[str] = None, verbose: bool = False)` | (别名, 同 `Cache.import_module`) |
| `cache.items` | `() -> List[Tuple[str, Any]]` | (别名, 同 `Cache.items`) |
| `cache.keys` | `() -> List[str]` | (别名, 同 `Cache.keys`) |
| `cache.lazy` | `(name: str, description: Optional[str] = None, verbose: bool = False)` | (别名, 同 `Cache.lazy`) |
| `cache.names` | `() -> List[str]` | (别名, 同 `Cache.names`) |
| `cache.remove` | `(name) -> bool` | (别名, 同 `Cache.remove`) |
| `cache.values` | `() -> List[Any]` | (别名, 同 `Cache.values`) |

## 3. 契约（由测试验证）

- 可导入；公开符号能从 `Utils` 顶层取到（`Utils` 包本身无 `__all__`）
- 公开可调用对象的签名带合理默认值（无必填隐藏参数）
- 枚举/Literal 参数非法值抛 `ValueError`
- 功能正确性（各模块在此文件里各自展开）
- 边界输入不崩（空 / 极短 / 极端尺寸）
- 副作用可控（测试完恢复全局状态）

## 4. 复现

```
.\.venv\Scripts\python.exe -m pytest tests/base/test_Cache.py -q
```

`__all__` 共 2 项。
