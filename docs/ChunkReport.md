# Chunk 效用报告 —— 统一分块工具

- 日期: 2026-09-07 · Python 3.10.11 · numpy 2.2.6 · psutil 可用
- 模块: `src/Modal_Decomposition/Utils/Chunk.py` · 测试: `tests/test_chunk.py` (37 项)
- 缓存注册: 键 `Modal_Decomposition.Utils.Chunk` (经 `Utils.get_chunk()` 惰性注册)

## 1. 它解决什么

分块逻辑原本散落在 `Check.py` (流式填充 memmap) 与 `Monotonicity.py`
(分块单调检测 + 自适应块大小) 两处, 各自内联一份 `range(0, n, chunk)`
循环。本模块把这些样板收敛为 6 个公开 API, Check / Monotonicity 已改为
复用同一实现 (测试 `test_check_fill_memmap_delegates` /
`test_monotonicity_chunked_path_smoke` 锁定该复用关系)。

| API | 作用 | 峰值内存 |
|---|---|---|
| `iter_chunks(S, c)` / `chunks` | 沿第 0 维零拷贝视图分块 | 0 额外 |
| `chunked_map(fn, S, c, out?, dtype?)` | 逐块映射, 结果拼接 | 一块 + 输出 |
| `chunked_fill(dst, src, c)` | 逐块流式填充 (memmap 目标) | 一块 |
| `exo_chunks(S, c, temp_dir?, dtype?)` | 外存临时 memmap 分块, 迭代结束清理 | 一块 |
| `adapt_chunk_size(c, nbytes, extra=3)` | 按全局内存策略收缩块大小 | — |
| `Chunk(...)` | 类式门面 (兼容旧参数 `mod/memmap_pth/memmap_type`) | 同对应函数 |

## 2. 实测 (中位数; 本机 Windows)

### 2.1 零拷贝迭代开销 —— `iter_chunks` 基本免费

n = 32M float64 (256MB), 对全量求和:

| 方式 | 耗时 | vs 全量 |
|---|---|---|
| `S.sum()` (基线) | 17.3 ms | — |
| `iter_chunks(c=256k)` | 17.5 ms | +1% |
| `iter_chunks(c=1M)` | 18.5 ms | +7% |
| `iter_chunks(c=8M)` | 18.6 ms | +8% |

块是视图 (无拷贝): 测试断言"改块即改原数组"。分块本身只付少量 Python
循环开销, 换来"每块独立处理 / 提前退出 / 内存可控"的结构能力。

### 2.2 逐块映射 —— `chunked_map` 用少量时间换内存上限

n = 8M float64, `np.square`:

| 方式 | 耗时 |
|---|---|
| `np.square(S)` 全量 | 18.8 ms |
| `chunked_map(c=1M)` | 32.3 ms (+72%) |

代价是 Python 逐块循环 + 切片; 收益是工作内存 = 一块 + 输出 (全量平方
需要 输入 + 输出 + 中间数组)。对大数组/受内存策略约束的场景, 该换算是
划算的; 数组小时直接用全量向量化即可。

### 2.3 流式填充 —— `chunked_fill` 把 memmap 填充内存压到一块

目标: 256MB 磁盘 memmap, 源为 244MB 内存数组, c=8M:

| 指标 | 值 |
|---|---|
| 墙钟 | ~256 ms |
| 过程后 RSS | ~581 MB (源仍在 RAM; 填充期间不出现第二份全尺寸拷贝) |

Check 的 memmap 输入层已委托本函数 (`Check._fill_memmap`)。

### 2.4 外存分块 —— `exo_chunks` 的磁盘代价

n = 4M float64, c=1M, 求和:

| 方式 | 耗时 |
|---|---|
| `iter_chunks` (内存视图) | 4.3 ms |
| `exo_chunks` (写临时文件再读回) | 57.9 ms |

磁盘模式约慢一个数量级, 换取"处理超大输入时每块只占 c 元素的内存"。
契约: 块仅在**当前迭代步**内有效 (边消费边拷贝, 勿跨步保留引用);
迭代结束后句柄关闭、临时文件删除 (best-effort, 已被外部视图映射的文件
在 Windows 上可能残留, 交给系统临时目录清理)。

### 2.5 自适应块大小 —— `adapt_chunk_size`

绝对策略 (nbytes=1GB, 调用方 chunk=8M, extra=3 字节/元素):

| 上限 | 预算 (limit - 1GB) | 实际块大小 |
|---|---|---|
| 2.0 GB | 1.0 GB | 8 000 000 (不动) |
| 1.2 GB | 0.2 GB | 8 000 000 (不动) |
| 1.05 GB | 0.05 GB | 8 000 000 (不动) |
| 1.02 GB | 0.02 GB | 7 158 278 (收缩到预算 // 3) |

规则: 输入 < 64MB 不咨询策略; 预算 <= 0 (输入已超限) 保持调用方块大小;
收缩下限 `MIN_CHUNK_ELEMS = 4096`。比率策略同理, 只是上限 =
`ratio × psutil.available` (psutil 缺失时保持原块大小)。

## 3. 结论与使用建议

1. **迭代/逐块** 是零拷贝且近乎免费 —— 任何"逐块扫描 + 提前退出"的算法
   (单调性检测即典型) 都应走 `iter_chunks`;
2. **大数组逐块计算** 用 `chunked_map`, 小数组直接向量化;
3. **写 memmap / 磁盘工作副本** 用 `chunked_fill` (Check 已复用);
4. **需要把大输入逐块放盘再处理** 用 `exo_chunks`, 遵守"当步有效"契约;
5. **块大小调优** 交给 `adapt_chunk_size` 与全局内存策略联动
   (Monotonicity 已复用);
6. 全库统一取用: `Utils.get_chunk()` / `cache.get("Modal_Decomposition.Utils.Chunk")`。

完整边界用例 (空数组、余块、多维、memmap 输入、非法块大小、out 形状
不匹配、dtype 转换、策略各分支、复用锁定、性能烟测) 见
`tests/test_chunk.py`。
