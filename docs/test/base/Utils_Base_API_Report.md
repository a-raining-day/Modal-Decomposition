# Utils / Base 工具层索引

工具层与分解方法采用**同一粒度**：每个模块一份报告 + 一个测试文件。
本文件是该层的总索引；各模块细节见其自己的 `*_Engine_Report.md`。

## 1. Base 层（5 个模块）

| 模块 | 报告 | 测试 | 一句话 |
|---|---|---|---|
| `Cache` | [Cache_Engine_Report.md](Cache_Engine_Report.md) | `tests/base/test_Cache.py` | 进程级导入缓存（`Cache` 与 `cache` 是同一个类对象） |
| `CPU` | [CPU_Engine_Report.md](CPU_Engine_Report.md) | `tests/base/test_CPU.py` | `cpu_num` / `allocable_cpu_num` 是模块级 int 常量，`set_allocable` 是函数 |
| `Chunk` | [Chunk_Engine_Report.md](Chunk_Engine_Report.md) | `tests/base/test_Chunk.py` | 分块工具（`chunk` 生成器 + `chunk_size`/`chunked_map`/`chunked_fill`/`fill_memmap`） |
| `Memory` | [Memory_Engine_Report.md](Memory_Engine_Report.md) | `tests/base/test_Memory.py` | 内存预算、`MemoryPolicy`、临时 memmap 生命周期 |
| `Seed` | [Seed_Engine_Report.md](Seed_Engine_Report.md) | `tests/base/test_Seed.py` | 进程级种子（默认 42）；`temporary_seed` 本轮修好 |

## 2. Utils 层（8 个模块）

| 模块 | 报告 | 测试 | 一句话 |
|---|---|---|---|
| `Check` | [Check_Engine_Report.md](../utils/Check_Engine_Report.md) | `tests/utils/test_Check.py` | 入参规范化与 `validate_mod`/`validate_safe` |
| `Envelope` | [Envelope_Engine_Report.md](../utils/Envelope_Engine_Report.md) | `tests/utils/test_Envelope.py` | 四种包络（Hilbert/Lowpass/IQ/PeakInterpolation） |
| `Hilbert` | [Hilbert_Engine_Report.md](../utils/Hilbert_Engine_Report.md) | `tests/utils/test_Hilbert.py` | `FFT` / `FHT` / `scipy` 三后端 |
| `Mirror` | [Mirror_Engine_Report.md](../utils/Mirror_Engine_Report.md) | `tests/utils/test_Mirror.py` | 五种端点延拓（mirror/extrema/polynomial/parallel/BLCSE） |
| `Monotonicity` | [Monotonicity_Engine_Report.md](../utils/Monotonicity_Engine_Report.md) | `tests/utils/test_Monotonicity.py` | `monotonic` + `give_monotony`（`Monotony` 是 3 位编码） |
| `Peaks` | [Peaks_Engine_Report.md](../utils/Peaks_Engine_Report.md) | `tests/utils/test_Peaks.py` | `find_peaks`（`mod="both"` 返 3 元组） |
| `Poly` | [Poly_Engine_Report.md](../utils/Poly_Engine_Report.md) | `tests/utils/test_Poly.py` | `fast_poly` 多项式重拟合 |
| `Slepian` | [Slepian_Engine_Report.md](../utils/Slepian_Engine_Report.md) | `tests/utils/test_Slepian.py` | DPSS 锥（numpy/scipy/C 三后端 + 缓存） |

## 3. 报告怎么来的

13 份报告全部由同一脚本内省生成，可重跑：

```
.\.venv\Scripts\python.exe -u .scratch\eemd_new\gen_tool_doc.py Base Cache CPU Chunk Memory Seed
.\.venv\Scripts\python.exe -u .scratch\eemd_new\gen_tool_doc.py Utils Check Envelope Hilbert Mirror Monotonicity Peaks Poly Slepian
```

生成器规则：有 `__all__` 的模块只列 `__all__` 内的名字；类会展开公开成员（`Class.method`）；同名别名标注"（别名）"；`typing`/`builtins`/`numpy`/`scipy`/`warnings` 等外部来源一律过滤。

## 4. 本轮在工具层修掉的真实缺陷

| 缺陷 | 后果 | 修复 |
|---|---|---|
| `Check.not_numpy_memmap` 判据写反 | `Utils.envelope()` 对任何普通 ndarray 都报错，整个包络 API 不可用 | 改为只拒绝 `np.memmap` |
| `Envelope.envelope` 转 Enum 后仍用字符串 `case` 分派 | 四种包络全部落到 "Unknown envelope method" | 改为 `case ENVELOPE_MODS.*` |
| `Seed.temporary_seed` 写单例**实例**属性 | 类方法 `Seed.get()` 在 `with` 内仍返回旧值，上下文管理器是 no-op | 改写字**类属性** `Seed.GLOBAL_SEED` |
| `Mirror` 的 `output_dtype` 取了类型对象 | `mirror(np.arange(8), mod="extrema")` 抛 `AttributeError`（`np.float64` 没有 `.kind`） | 包一层 `np.dtype(...)` |
| `Slepian.resolve_backend` 的 `match` 分支缺 `return` | `mod="C"` / `mod="scipy"` 返回 `None` → `slepian()` 抛 `ValueError: unsupported backend None`；`mod=None` 抛 `KeyError` | 补 `return`；`None` 与 `"auto"` 同分支；非法值改抛 `ValueError`（与 docstring 一致） |

## 5. 已知行为差异（不影响默认结果，未改代码）

- **`give_monotony` 分块路径的末块**：当 `n % chunk_size == 1` 时该块被判 `Equal`，严格位被合并抹掉。**仅当显式传 `chunk_size` 时才会走到这条路径**；默认 `chunk_size=None` 走整块路径，结果不受影响，故不改代码。测试 `test_chunked_strict_monotonic_mod_degrades_on_singleton_tail` 记录该边界行为。
- **`monotonic`/`give_monotony` 的 docstring** 写 `safe: Literal[0,1,2]`，实现只接受 `"safe"/"suspicious"/"unknown"` 三个字符串。
- **`Peaks.py` 的 docstring** 说 NaN "不会成为峰"，实测默认 `safe="unknown"` 会先抛 `ValueError("NaN or Inf found")`。

## 6. 运行

```
.\.venv\Scripts\python.exe -m pytest tests/base tests/utils -q
```
