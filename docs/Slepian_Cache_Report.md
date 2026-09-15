# Slepian Tier-1 缓存报告(进程内记忆化)

- 日期: 2026-09-15
- 范围: `Utils/Slepian.py` 新增的 `cache` 选项 + `cache_info()` / `clear_cache()`
- 复现: `python tests/comparison/bench_slepian.py`(§4 缓存延迟 / §5 命中率负载); 测试 `pytest tests/test_slepian.py -q`(286 项, 其中缓存 15 项)
- 关联: 后端选型与三后端正确性/性能见 `docs/Slepian_Backend_Report.md`

---

## 0. 结论 (TL;DR)

1. **命中比重新计算快 2~3 个数量级**: N=16384 时 26.8 ms → **0.012 ms**(2291×); N=65536 时 79.5 ms → 0.43 ms(184×, 大头是"返回副本"的拷贝)。
2. **真实批量负载收益**: 等长信号调用 20 次 → 命中率 95%、总耗时 **484 ms → 23.7 ms(20.4×)**; N=65536 调 10 次 → 10.8×。
3. **长度各不相同时没有收益**(命中率 0%, 仅 +4% 开销) —— 这是纯函数缓存的固有边界, 已在文档与报告中写明。
4. 语义安全: **命中与未命中逐位一致**; 命中返回**副本**(调用方可写, 不会污染缓存); 缓存内主副本**只读**; 键为**浮点按位精确**(不存在容差误配); 后端/是否要集中比/`sym`/`norm`/条数/长度/半带宽全部入键; 版本戳随算法改动强制失效。

---

## 1. 设计与规则

| 维度 | 做法 | 理由 |
|---|---|---|
| 定位 | **Tier-1 进程内 LRU**, 不落盘 | 覆盖绝大多数收益(同进程批量/循环/网格), 零 IO、零失效风险 |
| 归属 | 逻辑放在 `Utils/Slepian.py` 内部, **不进 `Base.Cache`** | `docs/CacheConventions.md` §1 明确 `Base.Cache` 是"唯一实例注册表", **不是 memoization**(参数不参与键) |
| 键 | `(缓存版本戳, 后端, N, halfBW 的 `float.hex()`, nTapers, sym, norm, return_ratios)` | 浮点**按位精确**: `NW=3.0` 与 `3.0000000000000004` 结果不同, 不允许容差匹配 |
| 版本戳 | `_CACHE_VERSION = "slepian-cache-v1"`, 同时进键与 `cache_info()` | 任何改动数值的修改(算法/符号约定/归一化/后端实现)必须提升版本, 否则旧条目被错误复用 |
| 容量 | 条数 `SLEPIAN_CACHE_SIZE=8` **且** 总字节 `SLEPIAN_CACHE_MAX_BYTES=256MB` | 只按条数限制无法约束"N=1M 的 5 条 taper"这类大条目; 超出按 LRU 淘汰 |
| 超大条目 | 单条 > 字节上限 → **只算不存**, 计入 `skipped` | 宁可少缓存, 不可撑爆内存 |
| 命中返回 | **副本**(可写) | 保住"返回数组归调用方所有"的既有契约, 避免就地改写污染缓存 |
| 缓存主副本 | `setflags(write=False)` | 防止内部/外部误写 |
| 并发 | `threading.Lock` 保护 get/put/clear/stats | 与库内其它全局状态的约定一致 |
| 开关 | `cache=None` 用 `ConstDefine.SLEPIAN_CACHE`(默认开); `True/False` 显式; 正整数 = 本次条数上限 | 默认开是因为它是纯函数记忆化且命中返回副本, 对调用方语义透明 |

API:

```python
slepian(N, halfBW, nTapers=None, mod=None, return_ratios=False,
        norm=None, sym=True, cache=None)     # cache=None/True/False/<int maxsize>

from Modal_Decomposition.Utils.Slepian import cache_info, clear_cache
cache_info()   # {version, size, bytes, hits, misses, evictions, skipped}
clear_cache()  # 清空并重置统计
```

---

## 2. 命中延迟 (NW=4.0, K=5, 关集中比; 命中取 best-of-5)

| N | 后端 | 计算(miss) ms | 命中(hit) ms | 加速比 | 单条缓存字节 |
|---:|---|---:|---:|---:|---:|
| 1 024 | scipy | 1.25 | 0.004 | **321×** | 40 960 |
| 4 096 | scipy | 4.50 | 0.006 | **776×** | 163 840 |
| 16 384 | scipy | 26.81 | 0.012 | **2 291×** | 655 360 |
| 65 536 | scipy | 79.48 | 0.433 | **184×** | 2 621 440 |
| 1 024 | C | 1.83 | 0.004 | **495×** | 40 960 |
| 4 096 | C | 7.55 | 0.006 | **1 259×** | 163 840 |
| 16 384 | C | 30.73 | 0.025 | **1 254×** | 655 360 |
| 65 536 | C | 117.78 | 0.414 | **284×** | 2 621 440 |

要点: 小/中 N 的命中几乎是纯字典查找(µs 级); N 增大后命中耗时由**返回副本的拷贝**主导(2.6 MB → 0.43 ms), 此时"缓存 vs 拷贝"的取舍开始显现 —— 若某调用方需要零拷贝(只读使用), 可在后续加 `cache_copy=False` 选项返回只读视图(当前未提供, 保持数组归属契约简单)。

---

## 3. 批量负载 (等长 vs 变长)

| 场景 | 调用次数 | 命中 | 未命中 | 命中率 | 总耗时 ms(开缓存) | 总耗时 ms(关缓存) | 加速 |
|---|---:|---:|---:|---:|---:|---:|---:|
| EWT 典型 (N=16384) | 20 | 19 | 1 | **95%** | 23.7 | 484.4 | **20.4×** |
| 中规模 (N=65536) | 10 | 9 | 1 | **90%** | 23.5 | 252.4 | **10.8×** |
| 长度各不同 | 10 | 0 | 10 | 0% | 301.7 | 288.9 | 1.0×(略慢 4%) |

**结论**: 缓存的收益完全取决于"同参数是否重复调用"。等长信号的批处理/循环/网格扫描收益 10–20×; 长度各异的调用没有收益(开销约 4%, 来自键构造与统计), 这也是把默认容量设成小值(8 条)而非无界的原因。

---

## 4. 与小 N 后端切换的配合

`scipy` 无法处理 `NW >= N/2`(默认参数下即 `N <= 6`); 该情形下 `resolve_backend` 会发 `UserWarning` 并改用 `SLEPIAN_SMALL_N_ORDER` 的首个可用后端(默认 `"C"`, 实测小 N 下比 numpy 快 7–10×、内存小 2.4×)。切换后的结果与 numpy 后端逐元素一致(测试 `test_small_n_result_matches_other_backends`), 且**缓存键包含后端名**, 因此"scipy 结果"与"回退后端结果"不会互相污染。

| N (NW=3, K=5) | 小 N 单次开销 | 后端 |
|---|---:|---|
| 2 | C 4.5 µs / 2.3 KB | C (scipy 必错) |
| 4 | C 5.4 µs / 2.8 KB | C |
| 6 | C 6.7 µs / 3.2 KB | C |
| 7 | scipy 可用 | scipy |
| 对比: numpy | 38–67 µs / 7.1–7.6 KB | — |

---

## 5. 测试覆盖(新增 15 项, 全部通过)

| 测试 | 验证内容 |
|---|---|
| `test_cache_hit_miss_and_equality` | 命中/未命中计数 + 结果相同 |
| `test_cache_key_is_bit_exact` | `NW` 位级差异 → 两个条目 |
| `test_cache_key_includes_all_arguments` | 后端 / 集中比 / `sym` / `norm` / 条数 / 长度 / 半带宽 各自入键 |
| `test_cache_disabled_by_flag` | `cache=False` 不写缓存 |
| `test_cache_maxsize_eviction_is_lru` | 条数上限 + LRU 淘汰顺序(最近使用者仍命中) |
| `test_cache_returns_independent_writable_copy` | 命中副本可写、改写不污染 |
| `test_cache_master_is_readonly` | 主副本只读(改写抛 `ValueError`) |
| `test_cache_byte_budget_skips_oversized_entry` | 超大条目只算不存 + `skipped` 计数 |
| `test_cache_consistency_without_cache` | 开/关缓存逐位一致(三后端) |
| `test_cache_invalid_argument` | `0 / -1 / "yes" / 1.5` → `ValueError` |
| `test_cache_clear_resets_stats` | 清空并重置统计 |
| `test_small_n_*` (5 项) | 小 N 换后端 + warning + 数值一致 + 顺序首位为 C |

---

## 6. 已知边界与后续可选增强

1. **不跨进程**: 本层只做进程内。若确需跨脚本/跨进程复用, 再按此前讨论的 Tier-2 方案(哈希文件名 + 头部参数与版本校验 + 原子写 + `mmap` 读 + 容量上限 + 只读文件系统静默降级)单独实现, 且**默认关闭**。
2. **命中拷贝成本**: N ≥ 262144 时命中拷贝约 1–2 ms(仍远小于计算的 290 ms+), 若将来出现"只读消费"的热路径, 可加 `cache_copy=False` 返回只读视图。
3. **版本戳纪律**: 修改任何影响数值的代码路径时必须提升 `_CACHE_VERSION`; 这条已写在该常量上方, 建议进入代码评审清单。
4. **未识别参数**: `cache` 之外的非法值已在 `_resolve_cache` 报错; 后续如新增影响结果的参数, 需同步加入 `_cache_key`。
