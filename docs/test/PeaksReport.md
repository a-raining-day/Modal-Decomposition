# Peaks 效用报告 —— 统一峰检测

- 日期: 2026-09-07 · Python 3.10.11 · numpy 2.2.6 · scipy 1.15.3 · numba 可用
- 模块: `src/Modal_Decomposition/Utils/Peaks.py` · 测试: `tests/test_peaks.py` (23 项)
- 缓存注册: 键 `Modal_Decomposition.Utils.Peaks` (经 `Utils.get_peaks()` 惰性注册)

## 1. 本次优化内容

| 问题 | 旧实现 | 现实现 |
|---|---|---|
| numpy 后端逻辑错误 | `logical_xor(相邻差, 相邻差)` 恒为 False, 且返回 n-1 掩码而非峰索引 | 向量化检测 `S[i-1] < S[i] >= S[i+1]`, 返回 int64 索引 |
| 返回契约不统一 | scipy 返回 `(idx, props)`, numpy 返回掩码 | 全部 `(indices:int64, properties:dict)` |
| 过滤参数 | scipy 手列 9 个 kw 透传; numpy 无过滤 | 统一 `height/threshold/distance` (scipy 语义), 不支持项显式 `NotImplementedError` |
| numba 分支不可运行 | 引用未定义变量 `module` | 按需 `njit` 编译检测内核 (`lru_cache` 每进程一次), 过滤与 numpy 共用 |
| 依赖获取 | 模块级旧式 `Cache` 键 `"ss"/"numba"` | `cache.import_module("scipy.signal"/"numba")` 惰性获取 |
| 无输入校验 | 无 | 1-D、长度 >= 3、实数 dtype; 短输入返回空 |
| distance 过滤复杂度 | — (未实现) | 有序列表 + bisect, O(P log P) |

## 2. 后端语义对照

| 能力 | scipy | numpy | numba |
|---|---|---|---|
| 检测规则 | 完整 (含 plateau 平台规则) | 基础规则 `prev < peak >= next` | 同 numpy (编译内核) |
| `height` / `threshold` / `distance` | ✅ | ✅ (scipy 语义) | ✅ (与 numpy 共用) |
| `prominence` / `width` / `wlen` / `rel_height` / `plateau_size` | ✅ 全透传 | ❌ `NotImplementedError` | ❌ 同 numpy |
| NaN | 不会成为峰 | 不会成为峰 (比较为 False) | 同 numpy |
| 返回 | `(idx, scipy properties)` | `(idx, {"peak_heights"})` | 同 numpy |
| 依赖 | scipy | 无 (纯 numpy) | numba (可选) |

无平台的严格尖峰信号上, numpy/numba 与 scipy 索引**逐位一致** (测试断言,
含 height/distance/threshold 组合)。平台信号请用 scipy 后端。

## 3. 实测性能 (中位数 5 次, 正弦 + 谐波 + 噪声, 无过滤)

| n | scipy | numpy | numba | 峰数 |
|---:|---:|---:|---:|---:|
| 10k | 0.03 ms | 0.02 ms | 0.01 ms | 3 330 |
| 100k | 0.29 ms | 0.09 ms | 0.26 ms | 33 501 |
| 1M | 4.21 ms | 2.95 ms | 3.20 ms | 333 213 |

纯检测时 numpy 向量化后端在 1M 点反而比 scipy 快 (~3 vs ~4.2 ms);
numba 编译内核与 numpy 同级 (此规模下无优势, 优势出现在更大的逐点循环
场景)。带 `height=0.5, distance=20` 过滤 (1M, 保留 13 630 峰):
scipy ~4.4 ms, numpy ~59 ms —— 差距来自 numpy 后端的 Python 贪心距离
过滤; 峰数多且需要 distance 时优先 scipy 后端。

## 4. 分支耗时 (三后端, 中位数 5 次, 同信号)

单位 ms; `detect` = 无过滤; `height` = height=0.5; `dist` = distance=20;
`prom` = prominence=0.3 (仅 scipy 支持)。

| n | scipy detect | scipy height | scipy dist | scipy prom | numpy detect | numpy height | numpy dist | numba detect | numba dist |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 10k | 0.02 | 0.03 | 0.08 | 0.54 | 0.02 | 0.02 | 0.81 | 0.02 | 0.85 |
| 100k | 0.30 | 0.33 | 1.04 | 7.23 | 0.08 | 0.13 | 11.21 | 0.22 | 11.75 |
| 1M | 4.27 | 5.89 | 13.23 | 76.52 | 3.03 | 4.56 | 190.09 | 3.52 | 193.50 |

补充:

* numba **首次调用含 JIT 编译**约 **483 ms** (一次性; 之后每调 ~0.2-3.5 ms,
  经 `lru_cache` 每进程只编译一次);
* 纯检测 (detect): numpy 在各规模都是最快 (1M: 3.0 vs scipy 4.3 ms);
* `distance` 过滤: scipy 走 C 实现 (1M 仅 13 ms), numpy/numba 为 Python
  O(P log P) 贪心 (1M/33 万峰 ~190 ms) —— 峰多 + 距离过滤时**必须选 scipy**;
* `prominence` (仅 scipy): 1M 约 77 ms, 属 scipy 独有能力的合理代价。

## 5. 结论与使用建议

1. 默认 `mod="scipy"`: 需要 plateau 规则或 `prominence/width` 等全参数;
2. `mod="numpy"`: 纯 numpy 环境 / 行为可控的快速检测, 无平台信号下最快;
3. `mod="numba"`: numba 已装的超大数组逐点场景 (编译一次, 之后 ~ms 级);
4. 大量峰的 `distance` 过滤用 scipy 后端 (C 实现), numpy 后端为
   O(P log P) 贪心;
5. 统一取用: `Utils.get_peaks()` / `cache.get("Modal_Decomposition.Utils.Peaks")`。

完整用例 (短输入/常量/NaN/2-D/复数/未知 mod/不支持参数/distance<1/
int 输入/numba 缺失与一致性/三后端过滤一致性/性能烟测) 见
`tests/test_peaks.py`。
