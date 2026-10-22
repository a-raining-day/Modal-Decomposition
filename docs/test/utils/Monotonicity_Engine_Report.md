# `Utils.Monotonicity` 模块报告

> 由 `.scratch/eemd_new/gen_tool_doc.py` 从**当前代码内省**生成；契约与 `tests/utils/test_Monotonicity.py` 一一对应。

## 1. 模块定位

Monotonicity checks.

## 2. 公开 API

| 名字 | 签名 | 说明 |
|---|---|---|
| `give_monotony` | `(arr, safe: Union[Literal['safe', 'suspicious', 'unknown'], Modal_Decomposition.Define.ConstDefine.SAFE_MODS] = 'unknown', chunk_size: int | None = None, chunk_mod: Optional[Literal['memory', 'exo-memory']] = None) -> Modal_Decomposition.Utils.Monotonicity.Monotony` | Classify a 1-d array into a ``Monotony`` member. |
| `monotonic` | `(arr, strict: bool = False, mod: Union[Literal['increasing', 'decreasing', 'monotonic'], Modal_Decomposition.Utils.Monotonicity.MONOTONY_MODS] = 'monotonic', safe: Union[Literal['safe', 'suspicious', 'unknown'], Modal_Decomposition.Define.ConstDefine.SAFE_MODS] = 'unknown', chunk_size: int | None = None, chunk_mod: Optional[Literal['memory', 'exo-memory']] = None) -> bool` | Check whether a 1-d array is monotonic. |

## 3. 枚举

| 枚举 | 成员 |
|---|---|
| `MONOTONY_MODS` | Increasing / Decreasing / Monotonic |
| `Monotony` | Equal / MixedIncreasing / MixedDecreasing / StrictIncreasing / StrictDecreasing / No |

## 4. 契约（由测试验证）

- 可导入且 `__all__` 与实际导出一致
- `mod` / `method` 等参数同时接受字符串与 Enum 成员
- 非法 `mod` / `backend` 抛 `ValueError`
- 返回值契约（形状 / dtype / 复数性 / 元组长度）
- 与参考实现（scipy / numpy）数值一致
- 边界输入不崩（空 / 单点 / 常量 / NaN）
- 确定性（同输入两次逐位相同）且不改写调用方输入

## 5. 复现

```
.\.venv\Scripts\python.exe -m pytest tests/utils/test_Monotonicity.py -q
```

`__all__` 共 5 项。

## 6. 深度测试（本轮）

> 测试：`tests/utils/test_Monotonicity_deep.py`（**29 条**，全绿）；数据：
> `table/utils/monotonicity/{semigroup_table,monotonic_truth_table,give_monotony_exhaustive,chunk_invariance}.csv`；
> 图：`fig/utils/monotonicity/monotonicity_matrix.png`。

### 6.1 覆盖

| 组 | 内容 | 规模 |
|---|---|---|
| 编码层 | 6 个成员的 3 位编码 + 6×6 合并表，与"按定义重算"的独立实现逐格比对 | 36 格 |
| 合并代数 | 交换律（36 对）、结合律（216 三元组）、`No` 吸收元 | 252 |
| 判定层 | **穷举**长度 2–4、符号表 {0,1,2} 的全部序列 × `give_monotony` / `monotonic(strict×mod)` | 117 条 / 702 格 |
| 分块层 | 9 种形态 × 7 种 `chunk_size`（None/1/2/3/5/7/16）× 严格与非严格 × 两个方向 | 63 格 |
| 边界 | 空/单点/常数/NaN/整型/bool/2-D/非法 safe/非法 mod/不改输入/确定性/O(N) | 12 |

### 6.2 发现

| 编号 | 现象 | 期望 |
|---|---|---|
| **F1（真 bug）·已修** | `_give_monotony` 的分块分支把边界增量 `delta` 只用于"是否已矛盾"的判定，**从未折进累加器**（`acc = m`）⇒ `chunk_size=1` 时任何序列退化成 `Equal`；块长不整除时 `StrictIncreasing → MixedIncreasing`（实测 `give_monotony(np.arange(10.), chunk_size=3)` 曾返回 `MixedIncreasing`） | **已修（两处）**：① 边界增量先折进累加器；② **单元素块不含相邻对**，它的 `Equal` 是空信息，不再参与合并（否则会清零严格位）。回归见 `test_give_monotony_chunked_matches_whole`（chunk 1/2/3/4/5/9/10/11 全等于整块）与旧的 `test_chunked_give_monotony_matches_whole` |
| **F2** | `Equal` 不是**严格性**的单位元：合并 `Equal × StrictIncreasing` 会清零严格位（`MixedIncreasing`）。语义上对，但与模块 docstring "Equal 为 00，起单位元作用" 的措辞不一致 | 改措辞为"方向单位元 + 严格性吸收元" |
| **F3** | `safe` 的 docstring 写 `Literal[0, 1, 2]`，实现只认 `"safe"/"suspicious"/"unknown"` 字符串，传 `0/1/2` 直接 `ValueError` | 统一为字符串（或加数字映射） |
| **F4** | `safe="safe"` 下 NaN 参与比较恒 False ⇒ 被归为 `Equal` ⇒ non-strict 单调判 `True` | 文档说明；或在 `safe` 档仍拒绝 NaN |

### 6.3 正常面

* 方向路径（`mod="increasing"/"decreasing"`）的**分块不变性 63 格全过** —— F1 只影响
  `give_monotony`（以及走它的 `mod="monotonic"`），不影响方向判定；
* 117 条序列的分类与 oracle 逐条一致；36 格 strict×mod 真值表全一致；
* dtype 保持（int8/int32/int64/float16/32/64/bool 结论相同）、不改输入、确定性、1 Mi 元素 < 1 s。

### 6.4 复现

```powershell
.\.venv\Scripts\python.exe -m pytest tests/utils/test_Monotonicity_deep.py -q
.\.venv\Scripts\python.exe tests/utils/test_Monotonicity_deep.py --export
.\.venv\Scripts\python.exe tests/size_axis/run_scale.py --tool monotonicity --export
```
