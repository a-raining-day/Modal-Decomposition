# `Utils.Mirror` 模块报告

> 由 `.scratch/eemd_new/gen_tool_doc.py` 从**当前代码内省**生成；契约与 `tests/utils/test_Mirror.py` 一一对应。

## 1. 模块定位

Extrema endpoint mirror extension (端点镜像外拓, 全库共享)。

## 2. 公开 API

| 名字 | 签名 | 说明 |
|---|---|---|
| `mirror` | `(S: numpy.ndarray | numpy.memmap, T: numpy.ndarray = None, mod: Union[Literal['mirror', 'extrema', 'polynomial', 'parallel', 'BLCSE'], Modal_Decomposition.Utils.Mirror.MIRROR_MODS] = 'mirror', nbsym: int = 3, poly_order: int = 2, time_poly_order: int = 2, *, verbose: bool = False, peaks_backend: Union[Literal['scipy', 'numpy', 'numba'], Modal_Decomposition.Utils.Peaks.PEAKS_BACKEND] = 'numpy', safe: Union[Literal['safe', 'suspicious', 'unknown'], Modal_Decomposition.Define.ConstDefine.SAFE_MODS] = 'unknown', **kwargs) -> numpy.ndarray | tuple[numpy.ndarray, numpy.ndarray]` | 信号端点外拓: 端点镜像, 或"检出极值 → 外延极值表"。 |

## 3. 枚举

| 枚举 | 成员 |
|---|---|
| `MIRROR_MODS` | Mirror / Extrema / Polynomial / Parallel / BLCSE |

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
.\.venv\Scripts\python.exe -m pytest tests/utils/test_Mirror.py -q
```

`__all__` 共 3 项。

## 6. 深度测试（本轮）

> 测试：`tests/utils/test_Mirror_deep.py`（**51 条**，全绿）；数据：
> `table/utils/mirror/mirror_modes.csv`；图：`fig/utils/mirror/mirror_modes.png`。

### 6.1 几何不变量（五模式 × nbsym 1–5）

| 不变量 | `mirror` | `extrema` | `polynomial` | `parallel` | `BLCSE` |
|---|---|---|---|---|---|
| `T` 严格递增、无重复时间戳 | — | ✓ | ✓ | ✓ | ✓ |
| 端部点落在中段之外 | — | ✓ | ✓ | ✓ | ✓ |
| 端部型别与边界极值相反、向外交替 | — | 镜像源 | ✓ | ✓ | ✓ |
| 中段与 `find_peaks(both)` 逐项相等 | — | ✓ | ✓ | ✓ | ✓ |
| 长度恒等式 | `N+2·nbsym` | `L+2·nbsym` | `L+2·nbsym` | `L+2·nbsym` | `L+4` |

`mirror` 还与 `np.pad(mode="reflect")` **逐位一致**（`nbsym = 1 … N-1`），
docstring 里的示例 `[2,1,0,1,2,3,4,5,4,3]` 也逐项验过。

### 6.2 发现

| 编号 | 现象 | 期望 |
|---|---|---|
| **F1（参数陷阱）** | `poly_order` 默认 **2**，而 `polynomial` 模式要求 `poly_order <= nbsym` ⇒ `mirror(S, mod="polynomial", nbsym=1)` **用默认参数就失败**；`nbsym=0` 更是必然失败（尽管其它模式 `nbsym=0` 合法） | `poly_order` 默认取 `min(2, nbsym)`，或在 `nbsym < 2` 时自动降阶 |
| **F2** | docstring 说非浮点 dtype 抛 `ValueError`，实际**静默提升**到 float64（`(wave*100).astype(int32)` 可以正常跑）；只有 `mirror` 模式保留整型 | 文档改成"整型会被提升为 float64" |
| **F3** | `T` 非单调时**只有 `verbose=True` 才告警**，默认静默重排（`argsort`）。重排本身正确，但"结果与输入顺序不一致"这件事无提示 | 默认 `warn` 一次，或文档强调 |
| **F4** | `BLCSE` 的长度恒为 `L+4`（与 `nbsym` 无关），`nbsym` 只是"平均窗口"；docstring 已写，但极易与其它三种模式的 `L+2·nbsym` 混淆 | 已在 §6.1 表里并列给出 |

### 6.3 正常面

* `nbsym=0` → 只返回中段（`extrema`/`parallel`；`polynomial` 不允许 0）；
* `nbsym > L-1` → 明确 `ValueError`；`extrema`/`polynomial`/`parallel` 都检查；
* 常数/斜坡信号（无极值）→ `ValueError`；`BLCSE` 缺极大或极小 → `ValueError`；
* 显式 `T`（`geomspace` 非均匀）→ 结果可用；递减 `T` → 告警并重排；
* `memmap` 输入可用；2-D/NaN 拒绝；不改输入；五模式确定性；
* 规模轴：600 MiB 上 `mirror` 190 ms、`extrema`(nbsym=4) 317 ms、`BLCSE` 318 ms（都约 0.3× 分配）。

### 6.4 复现

```powershell
.\.venv\Scripts\python.exe -m pytest tests/utils/test_Mirror_deep.py -q
.\.venv\Scripts\python.exe tests/utils/test_Mirror_deep.py --export
.\.venv\Scripts\python.exe tests/size_axis/run_scale.py --tool mirror --export
```
