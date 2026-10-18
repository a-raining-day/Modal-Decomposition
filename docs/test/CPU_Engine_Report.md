# `Base.CPU` 模块报告

> 骨架由 `.scratch/eemd_new/gen_tool_doc.py Base CPU` 从**当前代码内省**生成；
> 该模块没有模块级 docstring，生成器的"仅可调用对象"表格会漏掉 int 常量、并误把
> 导入的 `PYTHON_VERSION` 当成 API，因此第 1/2/3/5 节按**实测**人工校正。
> 契约与 `tests/base/test_CPU.py` 一一对应。

## 1. 模块定位

`src/Modal_Decomposition/Base/CPU.py`（**没有 module docstring**，故生成器此节为空）。
职责：在进程级维护"本机逻辑 CPU 数"与"可分配 CPU 预算"两个模块级 `int` 全局量，
并在导入时按 `Define.DEFAULT_ALLOCABLE_CPU_RATIO` 完成一次初始化；对外只暴露
`cpu_num` / `allocable_cpu_num` / `total_cpu` / `set_allocable` 四项。

关键事实（与常见预期不同，务必注意）：

- `Base.CPU` **是模块，不是类**。模块内没有任何类；`Modal_Decomposition.Base.CPU`
  与 `Modal_Decomposition.CPU` 解析到的都是这个模块对象本身。
- `cpu_num` / `allocable_cpu_num` 是**导入时求值一次的 `int` 常量**（不是函数）。
- `total_cpu` 是**函数** `() -> int`（首次调用探测并把结果缓存在 `cpu_num`），不是常量。
- 初始化顺序：`total_cpu()` → `set_allocable(DEFAULT_ALLOCABLE_CPU_RATIO)`。

## 2. 公开 API

`__all__` 共 4 项，与实测导出完全一致：

| 名字 | 种类 | 签名 / 值 | 说明 |
|---|---|---|---|
| `cpu_num` | `int` 常量 | 实测 `32` | 本机逻辑 CPU 数，导入时缓存 |
| `allocable_cpu_num` | `int` 常量 | 实测 `21` | 当前可分配预算，初始化 = `max(1, int(32 × 2/3))` |
| `total_cpu` | 函数 | `() -> int` | 探测并缓存 `os.cpu_count()`（3.13+ 优先 `os.process_cpu_count`） |
| `set_allocable` | 函数 | `(value: int \| float) -> int` | 设置预算并返回生效值 |

非公开 API（导入依赖，**不在** `__all__` 中）：`os`、`Define.PYTHON_VERSION`、
`Define.DEFAULT_ALLOCABLE_CPU_RATIO`。

## 3. 实测行为

环境：Python 3.10.11 / Windows-10-10.0.26200 / `os.cpu_count() = 32`，
无 `os.process_cpu_count`（3.13+ 才提供）。

| 调用 | 结果 |
|---|---|
| `set_allocable(1)` | `1` |
| `set_allocable(0.5)` | `16`（`int(32 × 0.5)`） |
| `set_allocable(0.01)` | `1`（floor 后为 0，被夹到下限 1） |
| `set_allocable(1.0)` | `32` |
| `set_allocable(-1)` | `32` —— docstring 写明的 "all" 哨兵值，**不抛异常** |
| `set_allocable(0)` | `ValueError` |
| `set_allocable(-2)` / `set_allocable(33)` | `ValueError`（只有 `-1` 是哨兵） |
| `set_allocable(1.5)` / `0.0` / `-0.5` | `ValueError`（比例须在 `(0, 1]`） |
| `set_allocable("x")` / `None` / `[1]` | `TypeError` |
| `set_allocable(True)` | `TypeError`（`bool` 是 `int` 子类，被显式拒绝） |

## 4. 契约（由 `tests/base/test_CPU.py` 验证，共 25 个用例）

- `__all__ == ["cpu_num", "allocable_cpu_num", "total_cpu", "set_allocable"]`，
  每项都存在且可 `import *`；模块内自定义函数与模块级 int 常量无遗漏。
- 导入的内部依赖（`os` / `PYTHON_VERSION` / `DEFAULT_ALLOCABLE_CPU_RATIO`）不在 `__all__` 里。
- `cpu_num` 是 `int` 且 `>= 1`；`total_cpu` 是函数，返回 `int >= 1` 且等于缓存的 `cpu_num`。
- `1 <= allocable_cpu_num <= cpu_num`；`DEFAULT_ALLOCABLE_CPU_RATIO` 在 `(0, 1]` 且套用后落在合法区间。
- `set_allocable(1) == 1`；`set_allocable(0.5)` 返回 `[1, cpu_num]` 内的 `int`，等于 `max(1, int(cpu_num × 0.5))`。
- 比例极小（`0.01`）被夹到 `1`；`1.0` 取满 `cpu_num`；`-1` 是 "all" 哨兵。
- 非法输入异常类型按**实测**断言：`0` / 越界 `int` / 越界比例 → `ValueError`；
  `str` / `None` / `list` / `bool` → `TypeError`；抛错时全局状态不变。
- 副作用可控：所有改写全局的用例经 `try/finally` 恢复 `allocable_cpu_num` 到测试前的值，
  含"try 块内抛异常"的恢复路径。
- 模块内不存在类；`Base.CPU is Modal_Decomposition.CPU is` 该模块对象（类/属性存在性烟测）。

## 5. 与任务背景的差异（以代码为准）

| 背景描述 | 实测 | 处理 |
|---|---|---|
| `Base` 导出 `CPU` **类** | `CPU` 是子模块，模块内无类 | 烟测改为"模块对象同一性 + 无类"断言 |
| `total_cpu` 是模块级 `int` 常量 | 是函数 `() -> int` | 断言 `callable(total_cpu)` 且返回值 `== cpu_num` |
| `set_allocable(-1)` 抛异常 | 返回 `cpu_num`（"all" 哨兵） | 断言返回 `cpu_num`；改用 `-2` 覆盖非法负数 |
| 非法输入统一 `ValueError` | `0`→`ValueError`，`"x"`→`TypeError` | 按实测类型分别断言 |

## 6. 复现与验收

```
.\.venv\Scripts\python.exe -m pytest tests/base/test_CPU.py -q
```

实测结果：`25 passed, 25 warnings`（warning 来自 `tests/conftest.py` 的
autouse 全局种子重置，与模块本身无关）。

生成/刷新本报告：

```
.\.venv\Scripts\python.exe -u .scratch\eemd_new\gen_tool_doc.py Base CPU
```

> 注意：生成器只产出骨架（本模块无 docstring 时第 1 节为空、API 表会误收
> `PYTHON_VERSION` 且漏掉两个 int 常量），**第 1/2/3/5 节的人工校正需在重跑生成器后重新应用**。
