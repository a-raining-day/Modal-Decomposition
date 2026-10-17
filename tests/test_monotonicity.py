"""
``Utils/Monotonicity`` 的回归测试。

运行::

    .venv\\Scripts\\python.exe tests\\test_monotonicity.py

说明: 复用 ``tests/test_mirror_new.py`` 的加载器 (包 ``__init__`` 尚不可用时注册
最小桩); 被测试的 ``Monotonicity.py`` 始终是磁盘上那一份。
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_mirror_new as _harness  # noqa: E402  (复用最小桩加载器)

_M = _harness._import("Modal_Decomposition.Utils.Monotonicity")

monotonic = _M.monotonic
give_monotony = _M.give_monotony
Monotony = _M.Monotony
MONOTONY_MODS = _M.MONOTONY_MODS


def test_give_monotony_classification():
    """分类 API (不带方向) 本身应当正确。"""
    inc = np.arange(10.0)
    dec = inc[::-1].copy()
    con = np.zeros(10)
    mixed = np.array([1.0, 2.0, 2.0, 3.0])       # 非严格递增
    bad = np.array([1.0, 3.0, 2.0])              # 有升有降

    assert give_monotony(inc) is Monotony.StrictIncreasing
    assert give_monotony(dec) is Monotony.StrictDecreasing
    assert give_monotony(con) is Monotony.Equal
    assert give_monotony(mixed) is Monotony.MixedIncreasing
    assert give_monotony(bad) is Monotony.No


def test_monotonic_default_direction():
    """默认 mod="monotonic": 任一方向单调即 True (递增数组也必须为 True)。"""
    inc = np.arange(10.0)
    dec = inc[::-1].copy()
    con = np.zeros(10)
    bad = np.array([1.0, 3.0, 2.0])

    assert monotonic(inc) is True, "递增数组被判为不单调"
    assert monotonic(dec) is True
    assert monotonic(con) is True
    assert monotonic(bad) is False
    assert monotonic(np.array([1.0])) is True, "n < 2 视为单调"


def test_monotonic_direction_mods():
    """mod="increasing"/"decreasing" 的方向语义 (回归: 曾与字符串比较而恒 false)。"""
    inc = np.arange(10.0)
    dec = inc[::-1].copy()
    con = np.zeros(10)
    bad = np.array([1.0, 3.0, 2.0])

    for name, arr, want_inc, want_dec in (
        ("inc", inc, True, False),
        ("dec", dec, False, True),
        ("con", con, True, True),                # 常数: 非严格下两个方向都算单调
        ("bad", bad, False, False),
    ):
        assert monotonic(arr, mod="increasing") is want_inc, f"{name}: increasing"
        assert monotonic(arr, mod="decreasing") is want_dec, f"{name}: decreasing"

    for member in (MONOTONY_MODS.Increasing, MONOTONY_MODS.Decreasing, MONOTONY_MODS.Monotonic):
        assert monotonic(inc, mod=member) == monotonic(inc, mod=member.value), \
            f"枚举入参与字符串不一致: {member}"


def test_monotonic_strict():
    inc = np.arange(10.0)
    dec = inc[::-1].copy()
    con = np.zeros(10)
    mixed = np.array([1.0, 2.0, 2.0, 3.0])

    assert monotonic(inc, strict=True) is True
    assert monotonic(dec, strict=True) is True
    assert monotonic(con, strict=True) is False, "常数不是严格单调"
    assert monotonic(mixed, strict=True) is False
    assert monotonic(inc, mod="increasing", strict=True) is True
    assert monotonic(mixed, mod="increasing", strict=True) is False
    assert monotonic(mixed, mod="increasing") is True, "非严格: 允许相等"


def test_monotonic_chunked_matches_single_chunk():
    """分块路径 (含跨块边界) 必须与单块结果一致。"""
    rng = np.random.default_rng(0)
    cases = [
        np.arange(500.0),
        np.arange(500.0)[::-1].copy(),
        np.zeros(500),
        np.sort(rng.standard_normal(500)),               # 非严格递增
        np.concatenate([np.arange(200.0), np.arange(200.0)[::-1]]),   # 有升有降
    ]
    for i, arr in enumerate(cases):
        for mod in ("monotonic", "increasing", "decreasing"):
            for strict in (False, True):
                single = monotonic(arr, mod=mod, strict=strict, chunk_size=10_000)
                chunked = monotonic(arr, mod=mod, strict=strict, chunk_size=37)
                assert single == chunked, f"case {i}: mod={mod} strict={strict} ({single} vs {chunked})"


def test_monotonic_list_input():
    """list 输入走 Check.to_array 的非 ndarray 路径 (NumPy 2 下容易炸)。"""
    assert monotonic([1.0, 2.0, 3.0]) is True
    assert monotonic([3.0, 2.0, 1.0], mod="decreasing") is True
    assert monotonic([1.0, 3.0, 2.0]) is False


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed, failed = 0, []
    for fn in tests:
        try:
            fn()
            print(f"[ok]   {fn.__name__}")
            passed += 1
        except Exception as exc:
            print(f"[FAIL] {fn.__name__}: {type(exc).__name__}: {exc}")
            failed.append(fn.__name__)
    print(f"\n{passed} passed, {len(failed)} failed" + (f" -> {failed}" if failed else ""))
    sys.exit(1 if failed else 0)
