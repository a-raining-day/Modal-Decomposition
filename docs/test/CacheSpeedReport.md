# Cache 速度报告 —— 使用 / 不使用 cache 对比

- 生成日期: 2026-09-07 · Python 3.10.11 · numpy 2.2.6 · scipy 1.15.3
- 生成脚本: `tests/comparison/bench_cache.py` (仓库根目录运行, 每次重跑刷新本文件)

## 1. 四种取用方式

| 方式 | 含义 |
|---|---|
| `direct` | 直接 import 后调用 (无 cache, 基线) |
| `cached_once` | `Utils.get_xxx()` 解析一次, 之后纯属性调用 (cache 一次性) |
| `getter_call` | 每次调用都走 `Utils.get_xxx()` (每调含 `cache.check` + `cache.get`) |
| `cache_get` | 每次调用都走 `cache.get(key)` (跳过 `check`) |

## 2. 单次调用耗时 (中位数)

| 操作 | direct | cached_once | getter_call | cache_get |
|---|---|---|---|---|
| is_monotonic (10k) | 6.70 µs (+0%) | 7.01 µs (+5%) | 8.43 µs (+26%) | 7.19 µs (+7%) |
| Check_Time_and_Signal (2k) | 2.00 µs (+0%) | 2.00 µs (-0%) | 3.43 µs (+71%) | 2.32 µs (+16%) |
| spline fit+eval (500) | 0.15 ms (+0%) | 0.16 ms (+2%) | 0.16 ms (+4%) | 0.16 ms (+6%) |
| envelope Hilbert (2k) | 0.03 ms (+0%) | 0.03 ms (-7%) | 0.03 ms (+13%) | 0.03 ms (+1%) |

> 结论: `cached_once` 与 `direct` 逐点等价 (解析一次后零额外开销);
> `getter_call`/`cache_get` 每调额外 ~0.5–2 µs (RLock + dict 查表),
> 对 ms 级运算可忽略, 仅在 µs 级小数组 (如 `is_monotonic`) 的紧循环中才可见。

## 3. 首次访问成本 (干净子进程, 中位 3 次)

| 模块 | import-only | getter (import + 注册) |
|---|---|---|
| `Hilbert` | 92.794 ms | 96.016 ms |
| `Spline` | 94.525 ms | 95.177 ms |
| `Envelope` | 95.593 ms | 90.760 ms |
| `Monotonicity` | 92.701 ms | 90.082 ms |

> 首次访问含一次性的 import (+ register) 开销; 注册本身只多一次
> `cache.check` + `cache.add` (微秒级)。之后所有调用均为纯查表。

## 4. Cache 使用流程 (Cache.py / Utils getter)

1. **注册 add(name, api, description?)** — 校验键 (非空字符串) 与 api (非 None),
   加锁后写入; 已存在则返回 `False` 且**永不覆盖** (保留最先注册的实例)。
2. **查询 get(name) / check(name) / describe(name)** — 校验键后加锁读取;
   未注册的 `get` 抛 `KeyError` 并列出全部已注册键。
3. **集中式惰性导入 import_module(name, description?)** — 抽象
   `if flag is None: try import except ImportError: raise` 样板: 首次 import
   并注册, 之后直接命中缓存; ImportError 原样传播且不缓存 (可重试)。
4. **声明式惰性加载 lazy(name, description?) 装饰器** — 零参加载函数
   进程内恰好执行一次 (持锁), 对外完全透明 (`functools.wraps`),
   带参调用/返回 None 抛 `TypeError`; 异常不缓存。
5. **惰性接入 (Utils 约定)** — 模块自身不接触缓存; `Utils.get_xxx()` 首次访问时
   `import_module` + `add`, 之后返回 `get(key)` 的同一实例; 幂等、线程安全。
6. **替换** — 唯一合法途径是 `remove(name)` 后再 `add`; 直接重复 `add` 被拒绝。
7. **枚举** — `names()/keys()/values()/items()/descriptions()` 排序快照;
   `in` / `len()` / `[]` / `repr` 等便捷协议。
8. **清空 clear()** — 删除全部条目 (主要供测试隔离使用)。
9. **并发** — 所有读写由单个 `RLock` 串行化: 并发注册同名键恰好成功一次,
   读取者永不看到半写入状态。

## 5. 使用建议

* 纯第三方 import: 用 `cache.import_module("scipy.signal")` (一处 try/except);
* 非 import 的昂贵构造: 用 `@cache.lazy(key)` 装饰零参加载函数;
* 循环外解析一次 (`mod = Utils.get_xxx()`), 循环内直接调 `mod.fn(...)`;
* 不要在 µs 级热循环里每调都走 `get_xxx()`/`cache.get`;
* 跨模块取用统一实例: `cache.get("Modal_Decomposition.Utils.Monotonicity")`
  与 `Utils.get_monotonicity()` 等价且同对象。
