# `Utils.Slepian` 模块报告

> 由 `.scratch/eemd_new/gen_tool_doc.py` 从**当前代码内省**生成；契约与 `tests/utils/test_Slepian.py` 一一对应。

## 1. 模块定位

Slepian (DPSS) 序列生成的统一入口 —— 三后端分发 (numpy / scipy / C)。

## 2. 公开 API

| 名字 | 签名 | 说明 |
|---|---|---|
| `available_backends` | `() -> dict` | 各后端可用性: ``{"numpy": True, "scipy": True/False, "C": True/False}``。 |
| `cache_info` | `() -> dict` | 缓存统计: ``{version, size, bytes, hits, misses, evictions, skipped}``。 |
| `clear_cache` | `(reset_stats: bool = True) -> None` | 清空缓存 (并可重置统计)。 |
| `resolve_backend` | `(N: int, mod: Literal['numpy', 'scipy', 'C', 'auto'] = 'auto', halfBW: float = None) -> str` | 解析生效后端名。 |
| `slepian` | `(N: int, halfBW: float, nTapers: int = None, mod: Literal['auto', 'numpy', 'scipy', 'C'] = 'auto', return_ratios: bool = False, norm=None, sym: bool = True, cache=None) -> numpy.ndarray` | 生成 Slepian (DPSS) 序列 —— 统一入口。 |

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
.\.venv\Scripts\python.exe -m pytest tests/utils/test_Slepian.py -q
```

`__all__` 共 5 项。

## 5. 深度测试（本轮）

> 测试：`tests/utils/test_Slepian_deep.py`（**35 条**，全绿）；数据：
> `table/utils/slepian/{backend_agreement,n_sweep}.csv`；
> 图：`fig/utils/slepian/slepian_backends.png`。

### 5.1 三后端一致性

`N ∈ {8,16,64,256} × NW ∈ {1,2,3} × K ∈ {1,2,4}`（本机三后端**全部可用**）：

| 对 | 最大绝对差 | 结论 |
|---|---|---|
| numpy vs scipy | ≤ 1e-10 | 符号约定一致（Percival & Walden） |
| numpy vs C | ≤ 1e-10 | 同上 |
| 集中比 vs scipy | ≤ 1e-11 | 同式 |

数学性质：`U Uᵀ = I`（≤1e-10，四个 N）、集中比**严格降序**且 ∈ (0,1]、`norm=None ≡ norm=2`、
`norm="approximate"` 时 0 阶峰值 ≈1（0.9995）、`K=1` 仍返回 2-D（scipy 会给 1-D）。

### 5.2 规模扫描（NW=4, K=4）

| N | numpy | scipy | C |
|---|---|---|---|
| 512 | 26.3 ms | 0.63 ms | 0.76 ms |
| 2048 | 481 ms | 2.7 ms | 3.3 ms |
| 8192 | **13 251 ms** | **7.5 ms** | 12.8 ms |
| 32768 | 拒绝（稠密矩阵超预算） | 32.9 ms | 52.3 ms |

⇒ numpy 后端在 N=8192 上比 scipy **慢 1770×**；`resolve_backend("auto")` 必须把大 N 导向
C/scipy —— 这是默认后端选 `"scipy"` 的实测依据。

### 5.3 发现

| 编号 | 现象 | 期望 |
|---|---|---|
| **F1** | 默认后端的两处 docstring 不一致：模块头写 `SLEPIAN_BACKEND` 默认 **"scipy"**，而 `slepian()` 的 `mod` 参数说明写"（默认 **"C"**）" | 以 `ConstDefine` 的实际值为准，统一文案 |
| **F2** | `resolve_backend(mod="C")` 在 C 未编译时**告警并降级 numpy**，于是 `slepian()` 里"显式指定 C 未编译 → `RealizationError`"的分支**不可达**（docstring 与实现不一致） | 要么删该分支/改文档，要么让 `resolve_backend` 显式抛出 |
| **F3** | numpy 后端的成本是 `O(N²)` 稠密矩阵：N=8192 已 13 s / 882 MiB，N=32768 直接拒绝 | 文档给出"numpy 后端仅适合 N ≲ 2 千"的建议（新增 §5.2 表） |
| **F4** | `norm="approximate"` 只保证 0 阶峰值 ≈1，高阶峰值自然更低（本例 0.874），易被误读为"归一化失败" | docstring 补一句 |
| **F5** | 缓存键用 `float.hex()` 做**按位精确**匹配：`3.0` 与 `nextafter(3.0)` 是两条记录（已按设计验证） | 无需改；这是有意的（避免容差匹配带来的错误复用） |

### 5.4 缓存（逐条验证）

* 命中返回**副本**：改返回值不污染缓存（`hits ≥ 1` 且值未变）；
* `cache=False` 时 `misses/hits` 都不增长；`cache=<int>` 是条数上限（LRU）；
* `return_ratios` 入键（缺 ratios 的条目不会被当成命中）；
* `cache_info()` 键集合 = `{version,size,bytes,hits,misses,evictions,skipped}`；`clear_cache()` 后 size=0；
* 非法 `cache` 值（字符串）报错。

### 5.5 复现

```powershell
.\.venv\Scripts\python.exe -m pytest tests/utils/test_Slepian_deep.py -q
.\.venv\Scripts\python.exe tests/utils/test_Slepian_deep.py --export
```
（Slepian 不进 1 MB–600 MB 规模轴：它的开销由窗长 `N` 决定，与数据量无关，改出 §5.2 的 N 扫描表。）
