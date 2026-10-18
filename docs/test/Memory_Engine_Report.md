# `Base.Memory` 模块报告

> 由 `.scratch/eemd_new/gen_tool_doc.py` 从**当前代码内省**生成；契约与 `tests/base/test_Memory.py` 一一对应。

## 1. 模块定位

Memory

## 2. 公开 API

| 名字 | 签名 | 说明 |
|---|---|---|
| `MemoryPolicy` | `()` |  |
| `MemoryPolicy.policy` | `() -> dict` |  |
| `MemoryPolicy.set_memory_reserve` | `(ratio: float) -> float` | 设置永不参与分配的系统余量比例。 |
| `cleanup_memmaps` | `(try_times: int = 5, force: bool = False) -> bool` | Close and delete every temporary memmap backing file created here. |
| `drop_memmap` | `(mm: numpy.memmap, try_times: int = 5, force: bool = False) -> bool` |  |
| `get_available_memory` | `(force: bool = False) -> int` | Return the remaining **physical** available memory in bytes. |
| `get_commit_available` | `(force: bool = False) -> int` | Return the approximate bytes still available to **commit** (physical |
| `get_memory_snapshot` | `(force: bool = False) -> dict` | 返回内存/提交状态的快照 (字节), 带 ``_SNAPSHOT_TTL`` 秒缓存。 |
| `get_process_memory` | `(force: bool = False) -> dict` | 本进程当前占用: ``{"rss": bytes, "private": bytes}`` (提交量口径为 vms)。 |
| `memory_budget` | `(extras_bytes: int = 0, force: bool = False) -> dict` | 计算调用方**一次分配**的安全预算与判定依据 |
| `should_use_memmap` | `(nbytes: int, extra: int = 0) -> bool` | Whether the projected memory usage should be served from a memmap. |
| `temp_memmap` | `(shape, dtype=<class 'numpy.float64'>, temp_dir: Optional[str] = None) -> numpy.memmap | None` | Create a writable temporary memmap in the system temp dir. |

## 3. 契约（由测试验证）

- 可导入；公开符号能从 `Utils` 顶层取到（`Utils` 包本身无 `__all__`）
- 公开可调用对象的签名带合理默认值（无必填隐藏参数）
- 枚举/Literal 参数非法值抛 `ValueError`
- 功能正确性（各模块在此文件里各自展开）
- 边界输入不崩（空 / 极短 / 极端尺寸）
- 副作用可控（测试完恢复全局状态）

## 4. 复现

```
.\.venv\Scripts\python.exe -m pytest tests/base/test_Memory.py -q
```

`__all__` 共 10 项。
