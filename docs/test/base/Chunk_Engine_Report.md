# `Base.Chunk` 模块报告

> 由 `.scratch/eemd_new/gen_tool_doc.py` 从**当前代码内省**生成；契约与 `tests/base/test_Chunk.py` 一一对应。

## 1. 模块定位

Chunked processing utilities (统一分块工具)

## 2. 公开 API

| 名字 | 签名 | 说明 |
|---|---|---|
| `Chunk` | `()` | Class facade over the chunking functions. |
| `Chunk.chunk` | `(S: numpy.ndarray, *, chunk_size: int = None, mod: Union[Literal['memory', 'exo-memory'], Modal_Decomposition.Base.Chunk.CHUNK_MOD] = 'memory', axis: int = 0, dtype=None, temp_dir: Optional[str] = None, **kwargs) -> Iterator[numpy.ndarray]` | Split ``S`` according to ``mod`` — always an iterator (in ``"None"`` |
| `Chunk.chunk_size` | `(nbytes: int, *, chunk_ratio: float = 0.75, extras_bytes: int = 0, **kwargs) -> int` | If `nbytes` <= `DEFAULT_FILL_CHUNK_SIZE`, use `DEFAULT_FILL_CHUNK_SIZE`. |
| `Chunk.chunked_fill` | `(dst: numpy.ndarray \| numpy.memmap, src: numpy.ndarray \| numpy.memmap, chunk_size: int = 8388608) -> numpy.ndarray` | Stream ``src`` into ``dst`` chunk-wise (peak memory = one chunk). |
| `Chunk.chunked_map` | `(func: Callable, S, chunk_size: Optional[int] = None, out: Optional[numpy.ndarray] = None, dtype=None) -> numpy.ndarray` | Apply ``func`` chunk-by-chunk along axis 0, returning one array. |
| `Chunk.fill_memmap` | `(fp: numpy.memmap, source: numpy.ndarray) -> None` | Stream ``source`` into ``fp`` chunk-wise so the peak memory stays at |

## 3. 枚举

| 枚举 | 成员 |
|---|---|
| `CHUNK_MOD` | Memory / ExoMemory |

## 4. 契约（由测试验证）

- 可导入；公开符号能从 `Utils` 顶层取到（`Utils` 包本身无 `__all__`）
- 公开可调用对象的签名带合理默认值（无必填隐藏参数）
- 枚举/Literal 参数非法值抛 `ValueError`
- 功能正确性（各模块在此文件里各自展开）
- 边界输入不崩（空 / 极短 / 极端尺寸）
- 副作用可控（测试完恢复全局状态）

## 5. 复现

```
.\.venv\Scripts\python.exe -m pytest tests/base/test_Chunk.py -q
```

`__all__` 共 3 项。

## 6. 备注（人工补充：生成器内省之外的已知事实）

- **导出面**：`Modal_Decomposition/Base/__init__.py` 本身**没有** `__all__`；`__all__` 定义在 `Base/Chunk.py`（3 项：`Chunk` / `CHUNK_MOD` / `CHUNK_MOD_ARGS`）。这三者由 `Base/__init__.py` 以**同一对象**转发，测试用 `is` 校验。第 4 节第 1 条模板里的 “`Utils` 顶层” 在 Base 场景下应读作 “`Base` 包 / `Base.Chunk` 子模块”。
- **枚举取值**：`CHUNK_MOD.Memory.value == "memory"`、`CHUNK_MOD.ExoMemory.value == "exo-memory"`；均可按值往返（`CHUNK_MOD("memory") is CHUNK_MOD.Memory`），枚举成员与字符串在 `chunk(mod=...)` 中行为等价。
- **Literal 别名**：`CHUNK_MOD_ARGS = Literal["memory", "exo-memory"]`，其成员与上述两个 `.value` 完全一致；生成器按 `__module__ == "typing"` 过滤，故未在正文表格中出现。
- **块长默认值**：`DEFAULT_FILL_CHUNK_SIZE = 8388608` 元素（8 Mi 元素）；既是 `chunked_fill` 的默认块长，也是 `chunk_size(nbytes)` 在 `nbytes <= 8388608` 时的返回值（`chunk(S)` 不传 `chunk_size` 时因此一块即可覆盖 ≤8 Mi 元素的输入）。
- **异常契约**：块长 `0` / 负数 → `ValueError`；块长 `bool` / `float` / `np.int64` → `TypeError`（`validate_chunk_size` 明确拒绝非 `int`）。非法 `mod`、`axis` 越界、空目标维、非数值 dtype 均抛 `ValueError`。
- **memmap 路径已实测**（各一块）：`mod="exo-memory"`（`temp_dir` 缺失时自动创建、`dtype` 覆盖生效、迭代结束后临时 `.dat` 被清理）、`chunked_fill` 写 `np.memmap` 目标后可重新映射读出、`fill_memmap` 流式落盘。
- **列表输入**：NumPy ≥ 2 下 `np.asarray(list, copy=False)` 无法避免拷贝，`chunk` 会原样抛 `ValueError`；测试按 NumPy 版本自适应断言（1.x 下可直接接受列表）。
- **用例规模**：`tests/base/test_Chunk.py` 共 56 条（参数化展开后），仅使用 ≤128 元素的小数组，不分配大数组。

## 7. 深度测试（本轮）

> 测试：`tests/base/test_Chunk_deep.py`（**49 条**，全绿）；数据：
> `table/base/chunk/{chunk_size_table,chunk_cases}.csv`；图：`fig/base/chunk/chunk_size.png`。

### 7.1 发现

| 编号 | 现象 | 期望 |
|---|---|---|
| **F1（核心）·已修** | `Chunk.chunk_size(nbytes)` **单位混用**：入参是**字节**，却与 `DEFAULT_FILL_CHUNK_SIZE`（8 Mi **元素**）比较；中间分支 `return nbytes` 把字节数当元素数返回 ⇒ 超出预算的输入会拿到 8 倍过大的块（5 GiB 输入 → 40 GiB 块） | **已修**：两处比较先换算到字节、最后按 `itemsize` 换回元素；新增 `itemsize=` 形参，`chunk()` / `fill_memmap()` 传入真实 `dtype.itemsize`。回归见 `test_chunk_size_units_are_bytes_in_elements_out` / `_above_budget_is_capped` |
| **F2（危险）·已修** | `_exo_chunks` 在 `finally` 里关掉**所有**块的映射并删文件 ⇒ 迭代结束后再 dereference 块会让解释器 **access violation 崩溃**（不是异常）。开发期实测：`np.array_equal(chunks[0], ...)` 处 `Windows fatal exception` | **已修**：每块改由 `Memory.temp_memmap` 拥有，迭代结束**不关闭**映射；临时文件登记在 `Memory` 注册表，由 `cleanup_memmaps()` / 解释器退出统一回收。回归见 `test_exo_memory_chunks_stay_valid_after_iteration` |
| **F3** | `chunked_fill(dst, src)` 不校验长度：`len(src) > len(dst)` 时**静默截断**（尾部丢弃），反方向才抛广播错误 | 显式长度校验 |
| **F4** | `fill_memmap(fp, list)` 必抛 `TypeError: src must be an ndarray or a memmap`（`Check._load` 的超长 list 分支因此从不生效） | 先 `np.asarray` |
| **F5 ·已修** | `cleanup_memmaps()` 的错误分支 `_ERROR_FILES.add(unexpected_files)` 加的是 **list**（集合）⇒ `TypeError: unhashable type: 'list'` | **已修**：改成 `update()`；同一轮还把 `FileNotFoundError` 视作"已删除"（POSIX 允许删除已映射文件、tmp 清理器也可能先删），现在"还有活引用"时 Windows 告警 + `False`、POSIX 直接 `True`，块两种情况下都可读 |
| **F6** | 死代码：`chunk` 里 `S.ndim == 0 and S.size == 0` 不可达（0-d 数组 size 恒 1） | 删或改注释 |
| **F7** | 列表输入在 NumPy ≥ 2 下 `np.asarray(list, copy=False, dtype=float64)` 直接 `ValueError`（原样抛出，未包装） | 已在 `Chunk_Engine_Report` §6 记录；建议 `np.asarray(list)` 不带 `copy=False` |

### 7.1.1 修好后的 exo-memory 契约

| 事项 | 行为 |
|---|---|
| 迭代结束后读块 | **可读**（不再崩溃）——映射随块对象存活 |
| 临时文件何时删 | `Memory.cleanup_memmaps()` 或解释器退出；`temp_dir` 不存在时自动创建 |
| 还有活引用时调 `cleanup_memmaps()` | 告警 + 返回 `False`（文件删不掉），块仍可用；`del` 掉块后再调即返回 `True` |
| 峰值内存 | 仍是"一块"（`chunk_size` 按预算封顶） |

### 7.2 正常面

* 内存视图分块：`chunk_size=4` on `arange(10)` → `[4,4,2]`，拼接等于原数组（视图，非拷贝）；
* `axis=0/1`、`dtype=` 覆盖、`temp_dir` 自动创建、**提前 break 也会清理**临时文件；
* `chunked_map`：`out`/`dtype`/空输入/返回 `out` 本身/memmap 输出；恒等映射下峰值 ≈ 输出（20 MiB → < 24 MiB）；
* `chunked_fill` 写 memmap + `flush`；`fill_memmap` 走 `chunk_size` 自适应（monkeypatch 验证调用链）；
* 非法块长（0/-1/True/2.0/np.int64）、非法 `mod`、`axis` 越界/非 int、bool/complex dtype 均按契约报错。

### 7.3 复现

```powershell
.\.venv\Scripts\python.exe -m pytest tests/base/test_Chunk_deep.py -q
.\.venv\Scripts\python.exe tests/base/test_Chunk_deep.py --export
.\.venv\Scripts\python.exe tests/size_axis/run_scale.py --tool chunk --export
```
