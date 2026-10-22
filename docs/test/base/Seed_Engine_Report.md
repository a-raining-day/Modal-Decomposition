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

## 5. 深度测试（本轮）

> 测试：`tests/base/test_Seed_deep.py`（**18 条**，全绿）。

### 5.1 覆盖

* 单例（`Seed() is Seed()`）、默认值 42、`get()` / `.seed` / `.status` 一致；
* `set` 返回新值、覆盖时 `UserWarning`（`verbose=False` 静默）、非法值（`-1` → `ValueError`；
  `"42"` / `3.5` / `None` → `TypeError`）；
* `temporary_seed` 正常/异常路径都恢复原值，非法种子在进入上下文时即报错；
* **锁**是进程级且不可解除 —— 该用例放在**子进程**里跑（`Seed.set(5, lock=True)` 后
  `Seed().status is True`、再 `set` 抛 `ValueError`），避免污染整个 pytest 会话；
* 多线程 `set`/`get` 烟测（4 线程 × 200 次，无异常、值恒为 int）。

### 5.2 发现

| 编号 | 现象 | 期望 |
|---|---|---|
| **F1** | `validate` 先 `isinstance(seed, int)`，所以 `np.int64(5)` **被拒**（尽管它有 `__index__`）；而 `bool` 通过 `isinstance` 被接受（`validate(True) == 1`） | 用 `operator.index()` 接受所有整数类型，并显式拒绝 bool |
| **F2** | `Global_Seed` 是**导入时快照**（`Seed.set()` 之后不更新）；docstring 已注明，但名字看不出是快照 | 改名为 `IMPORT_TIME_SEED` 或在文档首行强调 |
| **F3** | 实例属性遮蔽坑：`Seed().GLOBAL_SEED = 7` 只给单例加属性，`Seed.get()` 仍读类属性（docstring 已警告） | 建议 `__setattr__` 禁止写 `GLOBAL_SEED` |
| **F4** | 锁一旦打开**没有解锁 API**（`Seed._locked` 只能直接改类属性） | 提供 `unlock()`（测试/交互场景需要） |

### 5.3 复现

```powershell
.\.venv\Scripts\python.exe -m pytest tests/base/test_Seed_deep.py -q
```
