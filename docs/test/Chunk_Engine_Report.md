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
