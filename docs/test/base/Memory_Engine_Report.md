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

## 5. 深度测试（本轮）

> 测试：`tests/base/test_Memory_deep.py`（**20 条**，全绿）；数据：
> `table/base/memory/budget_table.csv`。

### 5.1 预算公式（逐项核对）

`budget = int(min(available·(1−reserve), commit_available·(1−reserve)·0.6)) − extras`，
`reserve` 默认 0.25。本机实测（测试时会随其他进程浮动，±1% 容差）：

| `memory_reserve_ratio` | budget | limited_by |
|---|---|---|
| 0.10 | 6144 MiB | physical |
| 0.25 | 5442 MiB | physical |
| 0.50 | 3628 MiB | physical |

⇒ 在本机（15.6 GB / 无大页面文件余量）**永远是物理内存先触顶**，`commit` 不是瓶颈。

### 5.2 覆盖

* 快照键集合、`commit_available == available + swap_free`、TTL 缓存与 `force=True`；
* `get_available_memory` / `get_commit_available` / `get_process_memory`；
* `should_use_memmap`：阈值上下、`extra` 参数、传数组、非 int 报错；
* `MemoryPolicy.set_memory_reserve` 的合法/非法取值与恢复；
* `temp_memmap`：注册与否（`persist`）、`temp_dir`、坏 dtype、读写往返；
* `drop_memmap`：非 memmap → `TypeError`、未注册 → `ValueError`、成功删除；
* `cleanup_memmaps`：全删、幂等、参数类型校验。

### 5.3 发现

| 编号 | 现象 | 期望 |
|---|---|---|
| **F1** | 坏 dtype 不报错：`temp_memmap((4,), "not-a-dtype")` 走 `except BaseException` 分支，**warn + 返回 None** | 要么抛 `TypeError`，要么 docstring 明写返回 None |
| **F2 ·已修** | `cleanup_memmaps()` 在"还有外部引用（Windows 删不掉）"时走到 `_ERROR_FILES.add(unexpected_files)`，加的是 **list** ⇒ `TypeError: unhashable type: 'list'` | **已修**：`_ERROR_FILES.update(...)`；现在告警 + 返回 `False`，与 `Chunk` 报告 F5 同一处 |
| **F3** | `snapshot_ttl=0.2 s` 意味着连续调用可能拿到**不同**的 available（本报告的 `test_budget_formula` 因此用 1% 容差） | 文档写明 TTL 语义（已写）；需要严格一致时用 `force=True` |

### 5.4 复现

```powershell
.\.venv\Scripts\python.exe -m pytest tests/base/test_Memory_deep.py -q
.\.venv\Scripts\python.exe tests/base/test_Memory_deep.py
```
