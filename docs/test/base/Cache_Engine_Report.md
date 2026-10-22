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

## 5. 深度测试（本轮）

> 测试：`tests/base/test_Cache_deep.py`（**29 条**，全绿）；数据：
> `table/base/cache/bench.csv`；图：`fig/base/cache/cache_bench.png`。

### 5.1 逐分支覆盖

* **校验**：`_validate_name`（非 str → `TypeError`；空/纯空白 → `ValueError`）、
  `_validate_api`（`None` 拒绝）、`_validate_description`（空串/空白 ≡ None）；
* **注册表**：`add` 永不覆盖（返回 False）、重复注册时不改描述、缺省描述自动生成、
  `get` 缺键的 `KeyError` 消息**列出已注册键**、`describe`/`remove`/`clear`/`names`/
  `keys`/`values`/`items`/`descriptions` 一致；
* **`import_module`**：正命中只 import 一次（monkeypatch 计数）、负缓存不再重试、
  `verbose=True` 二次调用仍抛 `ImportError`、已有条目优先于 import、`str` 子类走慢路径；
* **`lazy`**：恰好执行一次、带参 → `TypeError`、loader 返回 `None` → `TypeError`、
  loader 抛异常**不缓存**（可重试）、键已存在时不执行 loader、`functools.wraps` 保留元信息；
* **并发**：8 线程 × 200 次 → `importlib.import_module` 恰好 **1** 次（正/负两条路径），
  1600 个返回值同一对象；并发 `add` 竞态下只有 1 个 True。

### 5.2 快路径性能（本机，20000 次平均，含 ~25 ns 调用开销）

| 操作 | ns/call |
|---|---|
| `import_module` 命中 | **112** |
| `import_module` 负命中 | **136** |
| `get` 命中 | 266 |
| `check` 命中 | 256 |
| `"x" in cache()` | 304 |
| `len(cache())` | 185 |

命中路径在**无锁**读 `_entries`（名字入表时已校验，CPython 下 dict 读原子），
比改造前（315 ns / 309 ns）快 2.5–2.9×。`src/Modal_Decomposition/Base/Cache.py`
里"315 ns → ~55 ns"的注释已按实测改为"~110 ns"（55 ns 是省掉的校验+锁开销，
不是端到端耗时）。

### 5.3 发现

| 编号 | 现象 | 期望 |
|---|---|---|
| **F1（不一致）** | `__contains__` / `__len__` / `__getitem__` / `__repr__` **没有 `@classmethod`**，所以在**类上直接用会 TypeError**（`"x" in cache`、`len(cache)`、`cache["x"]` 全炸），必须实例化 `cache()`；而 `add`/`get`/`check` 是 classmethod、类上可用 | 同一个类的 API 有一半类级、一半实例级，易踩；建议 dunder 也改成 classmethod（或全部要求实例） |
| **F2** | 快路径用 `type(name) is str` 跳过校验：`str` 子类走慢路径（行为正确，但两条路径的异常时序不同） | 已按设计验证；建议在注释里点明"子类不是 bug" |
| **F3** | 负命中时 `verbose=True` 抛 `ImportError`，但**不刷新**负缓存（正确），只是错误信息 `"the {name} module is not pip!"` 语法有误 | 文案改成 "is not importable" |

### 5.4 复现

```powershell
.\.venv\Scripts\python.exe -m pytest tests/base/test_Cache_deep.py -q
.\.venv\Scripts\python.exe tests/base/test_Cache_deep.py --export
```
