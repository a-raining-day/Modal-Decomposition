# `Utils.Check` 模块报告

> 由 `.scratch/eemd_new/gen_tool_doc.py` 从**当前代码内省**生成；契约与 `tests/utils/test_Check.py` 一一对应。

## 1. 模块定位

本模块仅 Check_Time_and_Signal 主要负责对上级使用
Check 主要用于内部

## 2. 公开 API

| 名字 | 签名 | 说明 |
|---|---|---|
| `Check` | `()` | 针对已载入数组 (np.ndarray | np.memmap) 的原子检测与规范化。 |
| `Check.is_0d` | `(S: numpy.ndarray) -> bool` | 与 is_1d, is_2d 不同, is_0d 是指标量, 如 np.array(42) |
| `Check.is_1d` | `(S: numpy.ndarray) -> bool` |  |
| `Check.is_2d` | `(S: numpy.ndarray) -> bool` |  |
| `Check.is_duplicates` | `(S: numpy.ndarray, axis: int = 0, temp_dir: Optional[str] = None) -> bool` | 是否存在相邻重复值。分块遍历, 含跨块边界 |
| `Check.is_empty` | `(S) -> bool` |  |
| `Check.is_infinite` | `(S: numpy.ndarray | numpy.memmap, chunk_size: int = None) -> bool` | `True` means finite, `False` means infinite. |
| `Check.is_numeric` | `(S) -> bool` | dtype 为 bool / int / uint / float 之一。 |
| `Check.is_real` | `(S) -> bool` | dtype 为 int / uint / float 之一 (不含 bool)。 |
| `Check.is_uniform` | `(S: numpy.ndarray, *, rtol: float = 1e-10, atol: float = 1e-14, temp_dir: Optional[str] = None) -> bool` | 是否等间距。分块遍历, 含跨块边界差 |
| `Check.not_numpy_arr` | `(x: Any, method: str = '', raise_error: bool = True) -> bool` |  |
| `Check.not_numpy_memmap` | `(x: Any, method: str = '', raise_error: bool = True) -> bool` | 断言 ``x`` **不是** ``np.memmap`` (普通 ndarray 通过)。 |
| `Check.require_1d` | `(S, method: str = '') -> None` |  |
| `Check.require_length_match` | `(T, S, method: str = '') -> None` |  |
| `Check.require_ndim` | `(S, allowed, method: str = '') -> None` |  |
| `Check.require_nonempty` | `(S, method: str = '') -> None` |  |
| `Check.require_numeric` | `(S, method: str = '', *, allow_bool: bool = True) -> None` |  |
| `Check.return_dtype_kind` | `(S: numpy.ndarray | numpy.memmap) -> str` |  |
| `Check.to_array` | `(S, *, dtype: type | numpy.dtype = None, safe: str | Modal_Decomposition.Define.ConstDefine.SAFE_MODS = 'unknown', ndim: int | set | tuple | None = 1, need_squeeze: bool = True) -> numpy.ndarray` | 规范化任意 array-like 为 ndarray, 可选转 dtype。 |
| `Check.to_dtype` | `(S: numpy.ndarray | numpy.memmap, dtype: numpy.dtype) -> numpy.ndarray | numpy.memmap` | 转 dtype。memmap 走 temp_memmap + Chunk.fill_memmap 分块写, |
| `Check.validate_mod` | `(mod: Union[str, Type[enum.Enum]], MOD: Type[enum.Enum])` |  |
| `Check.validate_safe` | `(safe: str | Modal_Decomposition.Define.ConstDefine.SAFE_MODS) -> Modal_Decomposition.Define.ConstDefine.SAFE_MODS` |  |
| `Check_Time_and_Signal` | `(S, T=None, safe: Union[Literal['safe', 'suspicious', 'unknown'], Modal_Decomposition.Define.ConstDefine.SAFE_MODS] = 'unknown', ndim=None, method: str = '', default_T: bool = True, dtype: numpy.dtype = None, axis: int = 0, need_T: bool = True, chunk_size: int = None, temp_dir: Optional[str] = None, copy: bool = True) -> tuple[numpy.ndarray, typing.Optional[numpy.ndarray], int] | tuple[numpy.ndarray, int]` | Validate the signal and its now axis and return them in canonical form. |

## 3. 契约（由测试验证）

- 可导入且 `__all__` 与实际导出一致
- `mod` / `method` 等参数同时接受字符串与 Enum 成员
- 非法 `mod` / `backend` 抛 `ValueError`
- 返回值契约（形状 / dtype / 复数性 / 元组长度）
- 与参考实现（scipy / numpy）数值一致
- 边界输入不崩（空 / 单点 / 常量 / NaN）
- 确定性（同输入两次逐位相同）且不改写调用方输入

## 4. 复现

```
.\.venv\Scripts\python.exe -m pytest tests/utils/test_Check.py -q
```

`__all__` 共 2 项。

## 5. 深度测试（本轮）

> 测试：`tests/utils/test_Check_deep.py`（**89 条**，全绿）；数据：
> `table/utils/check/{check_dtype_matrix,check_cases,check_ctas_matrix}.csv`；
> 图：`fig/utils/check/check_matrix.png`。

### 5.1 覆盖

| 组 | 内容 | 条数 |
|---|---|---|
| dtype 矩阵 | 17 种 dtype（bool/int/uint/float16-64/complex/datetime64/timedelta64/object/str/unicode/void）× `is_numeric`/`is_real`/`return_dtype_kind` | 18 |
| 形状谓词 | 7 种 shape（0-d/空/单点/1-D/2-D/空 2-D/3-D）× `is_0d`/`is_1d`/`is_2d`/`is_empty` | 1 |
| `is_duplicates` | 相邻重复、非相邻、NaN、0-d、axis 0/1、memmap、**跨块**（monkeypatch 块长） | 6 |
| `is_uniform` | 等距/非等距、rtol 灵敏度、0-d、2-D squeeze、2×3 拒绝、**跨块** | 5 |
| `is_infinite` | 有限/inf/NaN/整型、强制分块分支（`chunk_size=7`） | 2 |
| `require_*` | 6 个函数的通过/拒绝 + `method` 前缀 | 5 |
| `to_array` | 序列/0-d/空/NaN/三档 safe/ndim 形式/squeeze/dtype 转换 | 17 |
| `to_dtype` | 同 dtype 恒等、ndarray、坏 dtype、memmap 不物化 | 4 |
| `Check_Time_and_Signal` | 默认 arange/need_T/default_T/copy/empty/squeeze/ndim/T 升序/重复/长度/2-D T/NaN/降序重排/axis/int 信号 + 浮点 T/safe 旁路/memmap/npy/坏路径/长 list | 25 |

### 5.2 发现（编号在 `utils/test_Check_deep.py` 里逐条对应）

| 编号 | 现象 | 期望 |
|---|---|---|
| **F1** | `return_dtype_kind` 第一步就用 `is_numeric` 拦住非 `biuf`，所以 `match` 里的 complex/datetime/timedelta/object/str/unicode/void **8 个分支是死代码**；错误信息还自相矛盾：`"must be np.ndarray, not <class 'numpy.ndarray'>"` | 要么允许这 8 类（去掉前置 `is_numeric`），要么删分支并修文案 |
| **F2** | `is_duplicates` 的 `axis` 只影响**切块方向**，diff 恒在最后一轴 ⇒ 相邻整行相同 / 同列相同都判不出；相邻 NaN 也不判重 | docstring 明确"只查最后一轴"，或按 axis 真正比较 |
| **F3** | `is_duplicates` 2-D 且**真的跨块**时边界比较是数组比较 → `ValueError: truth value ... is ambiguous`（现因 F4 不可达，块切分修好后就会暴露） | 边界用 `.any()` / 逐元素比较 |
| **F4** | `is_uniform` 对空 1-D 抛 `ValueError("chunking for goal dim requires at least 1")`，不是 vacuous True；且**峰值分配 3× 输入**（600 MiB → 1800 MiB，见规模轴） | 空数组提前返回 True；`np.diff` 结果可复用 |
| **F5** | `is_infinite` 名字与语义相反（**True = 有限**），返回 `np.bool_` 而非 `bool` | 改名 `is_finite` 或至少返回 `bool(...)` |
| **F6** | `_load` 的 `LIST_MEM_THRESHOLD`（>10 Mi 元素的 list）分支**必抛 TypeError**：把 list 原样交给 `Chunk.fill_memmap`，后者要求 ndarray/memmap | 先 `np.asarray` 再填 |
| **F7** | `safe="safe"` 完全跳过规范化（list 输入在 `S.shape` 处 AttributeError）；`safe="suspicious"` 跳过 ndim/有限性/squeeze ⇒ 可让 2-D 绕过 `ndim={1}`、让 NaN 信号进入算法 | 文档已写"信任调用方"，但 `suspicious` 的旁路范围过宽 |
| **F8** | `Check_Time_and_Signal` 把 `T` 强转成 `S.dtype`：整型信号 + 小数时间轴 → 截断成重复值 → `ValueError: Time axis contains duplicate values`（真实陷阱）；`axis>=1` + 显式 T → `Chunk` 判越界 `ValueError`；`require_length_match` 错误信息把 T 也写成 S | T 用浮点、axis 上限 `min(axis, T.ndim-1)` |
| **F9** | `to_array` 的 `S.ndim == 0 and S.size == 0` 分支不可达（0-d 数组 size 恒 1）；`to_array` 本身**不**拒绝空数组（拒绝在 CTAS） | 删死代码 / 在文档里写清分层 |

### 5.3 正常面（已逐条验证，可作回归基线）

* 17 种 dtype 的 `is_numeric`/`is_real`/`return_dtype_kind`（`biuf` 四类）；
* 6 个 `require_*` 的成功/失败路径与 `method` 前缀；
* `copy=True/False`：前者把 memmap 物化成内存 ndarray，后者保持同一映射（`out is src`）；
* 降序/非单调 `T` → 告警 + 重排，且**不改调用方数组**；
* `.npy` 路径（小文件 `np.load`、强制 memmap 分支、非 `.npy` 拒绝、缺文件 `FileNotFoundError`）；
* 1 Mi 元素上三个谓词耗时 < 1 s（防 O(N²) 回归）。

### 5.4 复现

```powershell
.\.venv\Scripts\python.exe -m pytest tests/utils/test_Check_deep.py -q
.\.venv\Scripts\python.exe tests/utils/test_Check_deep.py --export
.\.venv\Scripts\python.exe tests/size_axis/run_scale.py --tool check --export
```
