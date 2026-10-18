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
