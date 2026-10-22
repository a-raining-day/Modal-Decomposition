import time
from functools import wraps
from src.Modal_Decomposition import cache


def timer(func):
    """高精度函数计时装饰器"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start

        # 自动选择合适的单位显示
        if elapsed < 1e-6:
            print(f"{func.__name__} 耗时: {elapsed * 1e9:.2f} ns")
        elif elapsed < 1e-3:
            print(f"{func.__name__} 耗时: {elapsed * 1e6:.2f} µs")
        elif elapsed < 1:
            print(f"{func.__name__} 耗时: {elapsed * 1e3:.2f} ms")
        else:
            print(f"{func.__name__} 耗时: {elapsed:.4f} s")

        return result

    return wrapper

@timer
def direct_try_import():
    try:
        import none
        return none
    except ImportError:
        return False

@timer
def cache_import(api):
    cache.import_module(api)

flag = None

@timer
def if_import():
    global flag

    if flag:
        return flag
    else:
        try:
            import none
            flag = none
            return none
        except ImportError:
            flag = False


# ==================== 基准：三种写法 × 成功/失败 × 首次/后续 ====================
#
# 为什么不能用上面那三个函数直接比: 它们各自只跑一次、没有复位、而且
# ``if_import`` 用 ``if flag:`` 判断, 失败时 ``flag = False`` 会让下一次继续重试
# (等于没有负缓存)。下面用"None 哨兵"的正确形式, 并把状态复位也量化进去。

import statistics
import sys

SUCCESS_MODULE = "colorsys"     # 轻量 stdlib, 默认未被导入 -> 能测"首次真正 import"
MISSING_MODULE = "none"         # 不存在的模块 -> 必然 ImportError


# ---------- A. 裸 try-except: 每次调用都执行 import 语句 ----------
def a_success():
    try:
        import colorsys
        return colorsys
    except ImportError:
        return False


def a_failure():
    try:
        import none  # noqa: F401
        return True
    except ImportError:
        return False


# ---------- B. if flag + try-except: 模块级 flag, None = "还没试过" ----------
_flag_success = None
_flag_failure = None


def b_success():
    global _flag_success
    if _flag_success is None:
        try:
            import colorsys
            _flag_success = colorsys
        except ImportError:
            _flag_success = False
    return _flag_success


def b_failure():
    global _flag_failure
    if _flag_failure is None:
        try:
            import none  # noqa: F401
            _flag_failure = True
        except ImportError:
            _flag_failure = False
    return _flag_failure


# ---------- B'. 反例: 用 if flag: 判断真值 (失败会被反复重试) ----------
_flag_truthy = None


def b_failure_truthy():
    global _flag_truthy
    if _flag_truthy:
        return _flag_truthy
    try:
        import none  # noqa: F401
        _flag_truthy = True
    except ImportError:
        _flag_truthy = False
    return _flag_truthy


# ---------- C. cache.import_module: 统一缓存 + 负缓存 ----------
def c_success():
    return cache.import_module(SUCCESS_MODULE)


def c_failure():
    return cache.import_module(MISSING_MODULE)


# ---------- 状态复位 ----------
def reset(key: str) -> None:
    """把某个策略的进程级状态清干净, 以便测"首次"那一发。"""
    global _flag_success, _flag_failure, _flag_truthy

    sys.modules.pop(SUCCESS_MODULE, None)          # 让它下次是真正的 import

    if key == "b":
        _flag_success = None
        _flag_failure = None
        _flag_truthy = None
    elif key == "c":
        for name in (SUCCESS_MODULE, MISSING_MODULE):
            cache._entries.pop(name, None)
            cache._descriptions.pop(name, None)
            cache._exists.pop(name, None)


def _timeit(fn, n: int) -> float:
    """连续调用 n 次, 返回**单次**平均耗时 (比单次 perf_counter 稳定得多)。"""
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) / n


def measure_first(fn, key: str, rounds: int = 7) -> float:
    """首次耗时: 每轮先复位 + 清空 finder 缓存(真冷), 取中位数。"""
    samples = []
    for _ in range(rounds):
        reset(key)
        sys.path_importer_cache.clear()
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples)


def measure_next(fn, key: str, n: int = 2000) -> float:
    """后续耗时: 先热身一次(把缓存建起来), 再连续 n 次取单次均值。"""
    reset(key)
    fn()
    return _timeit(fn, n)


def fmt(sec: float) -> str:
    if sec < 1e-6:
        return f"{sec * 1e9:8.2f} ns"
    if sec < 1e-3:
        return f"{sec * 1e6:8.2f} µs"
    if sec < 1:
        return f"{sec * 1e3:8.2f} ms"
    return f"{sec:8.4f} s "


def main() -> None:
    # 先跑一遍用户原来的三个演示函数(只为展示用法, 不参与统计)
    print("--- 原有单次演示 (仅展示, 不参与统计) ---")
    direct_try_import()
    cache_import("none")
    if_import()

    rows = [
        ("A. try-except", "a", a_success, a_failure),
        ("B. if flag + try-except", "b", b_success, b_failure),
        ("C. Cache.import_module", "c", c_success, c_failure),
    ]

    print(f"\n--- 三种写法: 成功 import '{SUCCESS_MODULE}' / 失败 import '{MISSING_MODULE}' ---")
    print(f"{'策略':<26} {'成功·首次':>12} {'成功·后续':>12} {'失败·首次':>12} {'失败·后续':>12}")
    print("-" * 78)
    for label, key, f_ok, f_bad in rows:
        ok_first = measure_first(f_ok, key)
        ok_next = measure_next(f_ok, key)
        bad_first = measure_first(f_bad, key)
        bad_next = measure_next(f_bad, key)
        print(f"{label:<26} {fmt(ok_first)} {fmt(ok_next)} {fmt(bad_first)} {fmt(bad_next)}")

    # 反例: if flag: 判断真值, 失败时没有负缓存
    bad_next_truthy = measure_next(b_failure_truthy, "b", n=500)
    truthy_label = "B'. if flag: (反例, 失败会重试)"
    print("-" * 78)
    print(f"{truthy_label:<26} {'-':>12} {'-':>12} {'-':>12} {fmt(bad_next_truthy)}")

    print("\n说明:")
    print(f"  * 「首次」= 每轮先复位(sys.modules / flag / cache)并清空 sys.path_importer_cache, 取 {7} 轮中位数")
    print(f"  * 「后续」= 热身一次后连续 2000 次, 取单次均值 (含 perf_counter 与循环开销, 约几十 ns)")
    print("  * B 的 flag 必须用 None 作「未尝试」哨兵; 用 if flag: 时失败(false)会被反复重试,")
    print("    实测退化成和 A 一样的每次 ImportError 开销, 负缓存完全失效。")
    print("  * C 每次命中都要取 RLock + 两次参数校验, 所以后续耗时明显高于 A/B")


if __name__ == "__main__":
    main()