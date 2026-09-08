# Cache 全局使用约定

> 本文是 `Modal_Decomposition.Base.Cache` 的**唯一权威使用规范**。
> 新代码接入缓存前必须阅读; 与本文冲突的写法视为反模式。
> 速度数据见 `docs/CacheSpeedReport.md`; 实现位于 `src/Modal_Decomposition/Base/Cache.py`。

## 1. 定位与边界

`Cache` 是**进程级、线程安全的"唯一实例注册表"**, 职责只有三件:

1. 每个键对应**恰好一个对象**, 且**永不覆盖** (先注册者胜);
2. 让昂贵的 import / 构造**每进程只发生一次**, 并把 try/except 收敛到一处;
3. 提供可观测性 (`names/describe/items`) 与可治理性 (`remove/clear`)。

它**不是**:

* 不是 memoization (不缓存函数调用结果, 参数不参与键);
* 不是 sys.modules 的替代 (Python 自己已有 import 缓存, Cache 在其上补充
  "唯一实例 + 注册表" 语义);
* 不是配置中心 (可变数据一律不放入)。

## 2. 全局约定 (铁律)

| # | 约定 | 说明 |
|---|---|---|
| 1 | **模块自身不接触缓存** | 工具模块 (Check/Chunk/Peaks/...) 内部不写任何 cache 代码; 缓存只由 `Utils` 包的 getter 层引入 |
| 2 | **键 = sys.modules 风格全限定名** | 本库模块: `"Modal_Decomposition.Utils.Peaks"`; 第三方: 直接用模块名 `"scipy.signal"`。键大小写敏感、非空字符串, 禁用 `"ss"` 之类魔数 |
| 3 | **永不覆盖** | `add` 对已存在键返回 `False` 并保留原对象; 替换唯一途径 `remove` → `add` |
| 4 | **异常不缓存、可重试** | `import_module` / `lazy` 中工厂抛异常 (如 ImportError) 时原样传播且**不写入**, 下次调用自动重试 |
| 5 | **恰好执行一次** | 惰性加载在 `RLock` 内完成 (check → 工厂 → add 原子化), 并发调用同一键只执行一次工厂 |
| 6 | **禁止缓存 None** | `add` 拒绝 None api; `lazy` 工厂返回 None 抛 TypeError |
| 7 | **惰性注册时机** | getter 首次访问时才 import + 注册 (首次 ~90ms 主要是一次性 import, 注册本身 µs 级) |
| 8 | **热路径不每调查缓存** | 循环外解析一次 (`mod = Utils.get_xxx()`), 循环内直接 `mod.fn(...)`; 每调 `cache.get` 约 +0.5–2 µs |
| 9 | **双路径唯一实例** | `Modal_Decomposition.*` 与 `src.Modal_Decomposition.*` 两条 import 路径共享同一键; 先注册者胜, 全进程拿到同一对象 |
| 10 | **测试隔离用 clear()** | 涉及全局状态的测试用 autouse fixture `cache.clear()` 前后清理 |

## 3. 原语清单

| 原语 | 用途 |
|---|---|
| `add(name, api, description?, verbose?) -> bool` | 注册 (幂等, 不覆盖) |
| `get(name)` / `check(name)` / `describe(name)` | 查询 (get 未命中抛 KeyError 并列出已注册键) |
| `remove(name) -> bool` | 删除 (替换的唯一前置步骤) |
| `clear()` | 清空 (测试隔离) |
| `names()/keys()/values()/items()/descriptions()` | 排序快照 / 枚举 |
| `len(cache())` / `"x" in cache()` / `cache()["x"]` / `repr` | 便捷协议 |
| `import_module(name, description?, verbose?)` | **集中式惰性导入**: 抽象 `if flag is None: try import except ImportError: raise` 样板 |
| `lazy(name, description?, verbose?)` | **装饰器**: 零参加载函数进程内恰好执行一次, 对外透明 |

## 4. 三种标准用法

### 4.1 纯第三方 import (推荐 `import_module`)

```python
from Modal_Decomposition.Base.Cache import cache

def _signal():
    """唯一的 try/except 收敛于此; 调用方零样板。"""
    return cache.import_module(
        "scipy.signal",
        description="scipy.signal 信号处理子模块, 供 XXX 使用",
    )
```

### 4.2 非 import 的昂贵构造 (装饰器 `lazy`)

```python
@cache.lazy("Modal_Decomposition.Utils.Foo", description="...")
def _load_foo():
    return Foo()          # 恰好构造一次; 异常不缓存可重试
```

### 4.3 新增 Utils 工具模块 (标准三步)

```python
# ① Utils/__init__.py 目录登记
_UTILS_MODULES = {
    ...
    "Foo": ("Modal_Decomposition.Utils.Foo", "Utils.Foo: 一句话说明"),
}

# ② 一个 getter
def get_foo():
    """Return the Foo module via the import cache (lazy import + register)."""
    key, desc = _UTILS_MODULES["Foo"]
    return _get_cached_module("Foo", key, desc)

# ③ 加入 __all__
```

## 5. 取用方式 (使用者视角)

```python
# A. 普通 import: 最快、IDE 提示最全 (sys.modules 层缓存, 不注册 Cache)
from Modal_Decomposition.Utils.Monotonicity import is_monotonic

# B. getter: 惰性注册 + 返回 Cache 中的唯一实例
from Modal_Decomposition import Utils
mono = Utils.get_monotonicity()      # 解析一次
mono.is_monotonic(arr)               # 之后零额外开销

# C. 直接查缓存 (须已注册)
from Modal_Decomposition.Base.Cache import cache
cache.get("Modal_Decomposition.Utils.Monotonicity")
```

A 与 B 拿到的是**同一模块实例** (getter 内部即 import); 差别只在是否
触发 Cache 注册与跨路径唯一性保证。

## 6. 反模式 (禁止)

* ❌ 工具模块内部自注册 (`register_to_cache` 之类的模块级缓存代码);
* ❌ 键用 `"ss"` / `"np"` 等魔数或不一致命名;
* ❌ 在 µs 级热循环里每调 `Utils.get_xxx()` / `cache.get`;
* ❌ 把会被修改的数据 (数组、可变配置) 放进去当共享状态;
* ❌ 直接改 `cache._entries` / `cache._descriptions` (必须走原语);
* ❌ 期待 `add` 覆盖旧值 (会被静默拒绝);
* ❌ 捕获了 KeyError 还继续跑 —— 未注册说明接入流程没走对, 应查 getter。

## 7. 现注册条目一览

| 键 | 对象 |
|---|---|
| `Modal_Decomposition.Utils.Check` | 校验/输入层 |
| `Modal_Decomposition.Utils.Memory` | 内存策略 |
| `Modal_Decomposition.Utils.Monotonicity` | 单调性检测 |
| `Modal_Decomposition.Utils.Seed` | 两级种子 |
| `Modal_Decomposition.Utils.Hilbert` | Hilbert/FHT 门面 |
| `Modal_Decomposition.Utils.Spline` | 样条封装 |
| `Modal_Decomposition.Utils.Envelope` | 包络提取 |
| `Modal_Decomposition.Utils.Chunk` | 分块工具 |
| `Modal_Decomposition.Utils.Peaks` | 峰检测 |
| `scipy.signal` / `scipy.interpolate` / `numba` | 第三方子模块 (按需) |
