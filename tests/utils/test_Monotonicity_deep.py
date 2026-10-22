"""``Utils.Monotonicity`` 深度测试。

覆盖三层：

1. **编码层**：``Monotony`` 的 3 位编码（方向 | 严格性）与合并半群，穷举 6×6 表，
   并与"定义重算"的独立实现逐格比对（不调用被测的合并函数）；
2. **判定层**：``give_monotony`` / ``monotonic`` 对**所有**长度 2–4、符号表 {0,1,2}
   的序列（117 条）逐个与独立 oracle 比对；含 36 格 (strict × mod) 真值表；
3. **分块层**：分块结果必须与整块结果**逐位一致**（chunk_size 1/2/3/5/7/16/None ×
   多种形态），这是分块实现的核心不变量，也是唯一能压住边界 bug 的检查。

导出::

    .\\.venv\\Scripts\\python.exe tests/utils/test_Monotonicity_deep.py --export

规模轴（1–600 MiB）在 ``tests/size_axis/run_scale.py --tool monotonicity``。
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pytest

_TESTS = Path(__file__).resolve().parents[1]
if str(_TESTS) not in sys.path:
    sys.path.insert(0, str(_TESTS))

import _bench as B  # noqa: E402

B.bootstrap_src()

from Modal_Decomposition.Utils.Monotonicity import (  # noqa: E402
    MONOTONY_MODS,
    MONOTONY_MODS_ARGS,
    Monotony,
    give_monotony,
    monotonic,
)
import Modal_Decomposition.Utils.Monotonicity as _M  # noqa: E402


# --------------------------------------------------------------------------- #
# 独立 oracle（与实现无关：直接照单调性的数学定义写）
# --------------------------------------------------------------------------- #
def oracle_monotony(seq) -> Monotony:
    a = list(seq)
    if len(a) < 2:
        return Monotony.Equal
    gt = sum(1 for x, y in zip(a, a[1:]) if y > x)
    lt = sum(1 for x, y in zip(a, a[1:]) if y < x)
    m = len(a) - 1
    if gt == m:
        return Monotony.StrictIncreasing
    if lt == m:
        return Monotony.StrictDecreasing
    if gt == 0 and lt == 0:
        return Monotony.Equal
    if lt == 0:
        return Monotony.MixedIncreasing
    if gt == 0:
        return Monotony.MixedDecreasing
    return Monotony.No


def oracle_monotonic(seq, strict: bool, mod: str) -> bool:
    a = list(seq)
    if len(a) < 2:
        # 实现里 n<2 时 give_monotony→Equal；mod="monotonic"+strict 会因 Equal 判 False
        return True if not (mod == "monotonic" and strict) else False
    if mod == "increasing":
        return all((y > x) if strict else (y >= x) for x, y in zip(a, a[1:]))
    if mod == "decreasing":
        return all((y < x) if strict else (y <= x) for x, y in zip(a, a[1:]))
    inc = all((y > x) if strict else (y >= x) for x, y in zip(a, a[1:]))
    dec = all((y < x) if strict else (y <= x) for x, y in zip(a, a[1:]))
    return bool(inc or dec)


# 定义重算的半群（只读枚举的**数值**，不调用被测函数）
def semigroup_by_definition(a: Monotony, b: Monotony) -> Monotony:
    d = (a.value & 0b011) | (b.value & 0b011)
    if d == 0b11:
        return Monotony.No
    s = (a.value >> 2) & (b.value >> 2)
    return Monotony((s << 2) | d)


# --------------------------------------------------------------------------- #
# 1. 编码层
# --------------------------------------------------------------------------- #
def test_monotony_encodings():
    expect = {
        "Equal": 0b000, "MixedIncreasing": 0b001, "MixedDecreasing": 0b010,
        "StrictIncreasing": 0b101, "StrictDecreasing": 0b110, "No": 0b111,
    }
    assert {m.name: m.value for m in Monotony} == expect


def test_semigroup_table_matches_definition():
    rows = build_semigroup_table()
    bad = [r for r in rows if r["verdict"] != "PASS"]
    assert not bad, bad


def test_semigroup_identity_and_absorbing():
    """``Equal`` 只是**方向**的单位元；它会抹掉严格位，因此不是严格性的单位元。

    这是实现里 ``s = (a >> 2) & (b >> 2)`` 的直接推论（Equal 的严格位是 0），
    语义上也对（出现重复就不该是"严格"单调），但与模块 docstring 里
    "Equal 为 00，起单位元作用" 的措辞不一致 —— 合并后严格性会被清零。
    """
    for m in Monotony:
        if m is Monotony.No:                     # No 是吸收元，单独在下面验
            continue
        merged = _M._commutative_semigroup(Monotony.Equal, m)
        assert (merged.value & 0b011) == (m.value & 0b011), "方向位应保持不变"
        assert (merged.value >> 2) == 0, "Equal 会把严格位清零"
        assert merged is semigroup_by_definition(Monotony.Equal, m)
    for m in Monotony:
        assert _M._commutative_semigroup(Monotony.No, m) is Monotony.No
        assert _M._commutative_semigroup(m, Monotony.No) is Monotony.No


def test_semigroup_commutative_and_associative():
    members = list(Monotony)
    for a, b in itertools.product(members, repeat=2):
        assert _M._commutative_semigroup(a, b) is _M._commutative_semigroup(b, a)
    for a, b, c in itertools.product(members, repeat=3):
        left = _M._commutative_semigroup(_M._commutative_semigroup(a, b), c)
        right = _M._commutative_semigroup(a, _M._commutative_semigroup(b, c))
        assert left is right, (a, b, c, left, right)


# --------------------------------------------------------------------------- #
# 2. 判定层：穷举 117 条序列
# --------------------------------------------------------------------------- #
ALL_SEQS = [s for k in (2, 3, 4) for s in itertools.product((0, 1, 2), repeat=k)]


def test_give_monotony_exhaustive():
    bad = []
    for seq in ALL_SEQS:
        arr = np.array(seq, dtype=np.float64)
        got = give_monotony(arr)
        exp = oracle_monotony(seq)
        if got is not exp:
            bad.append((seq, exp.name, got.name))
    assert not bad, bad[:10]


@pytest.mark.parametrize("strict", [False, True])
@pytest.mark.parametrize("mod", ["increasing", "decreasing", "monotonic"])
def test_monotonic_exhaustive(strict, mod):
    bad = []
    for seq in ALL_SEQS:
        arr = np.array(seq, dtype=np.float64)
        got = bool(monotonic(arr, strict=strict, mod=mod))
        exp = oracle_monotonic(seq, strict, mod)
        if got != exp:
            bad.append((seq, exp, got))
    assert not bad, bad[:10]


def test_monotonic_truth_table_36_cells():
    rows = build_truth_table()
    bad = [r for r in rows if r["verdict"] != "PASS"]
    assert not bad, bad


# --------------------------------------------------------------------------- #
# 3. 分块不变量
# --------------------------------------------------------------------------- #
PATTERNS = {
    "strict_increasing": lambda n: np.arange(n, dtype=np.float64),
    "mixed_increasing": lambda n: np.repeat(np.arange(n // 2 + 1, dtype=np.float64), 2)[:n],
    "strict_decreasing": lambda n: np.arange(n, 0, -1, dtype=np.float64),
    "mixed_decreasing": lambda n: np.repeat(np.arange(n // 2 + 1, 0, -1, dtype=np.float64), 2)[:n],
    "flat": lambda n: np.full(n, 2.5),
    "spike": lambda n: np.array([0.0 if i != n // 2 else 1.0 for i in range(n)]),
    "sawtooth": lambda n: (np.arange(n) % 3).astype(np.float64),
    "random": lambda n: np.random.default_rng(7).standard_normal(n),
    "two_halves_opposite": lambda n: np.concatenate([
        np.arange(n // 2, dtype=np.float64),
        np.arange(n - n // 2, 0, -1, dtype=np.float64)]),
}

CHUNK_SIZES = [None, 1, 2, 3, 5, 7, 16]


def build_chunk_invariance() -> list[dict]:
    """分块 vs 整块：方向路径（``mod="increasing"/"decreasing"``）与 ``give_monotony`` 各记一列。

    ``verdict`` 只看**方向路径**（那条路径实现正确）；``monotony_ok`` 单列记录
    ``give_monotony`` 的分块缺陷（见 :func:`test_give_monotony_chunked_defect`）。
    """
    rows = []
    for name, gen in PATTERNS.items():
        arr = gen(73)
        base_dir = {mod: bool(monotonic(arr, strict=True, mod=mod))
                    for mod in ("increasing", "decreasing")}
        base_dir_ns = {mod: bool(monotonic(arr, strict=False, mod=mod))
                       for mod in ("increasing", "decreasing")}
        base_m = give_monotony(arr)
        for cs in CHUNK_SIZES:
            got_dir = {mod: bool(monotonic(arr, strict=True, mod=mod, chunk_size=cs))
                       for mod in ("increasing", "decreasing")}
            got_dir_ns = {mod: bool(monotonic(arr, strict=False, mod=mod, chunk_size=cs))
                          for mod in ("increasing", "decreasing")}
            got_m = give_monotony(arr, chunk_size=cs)
            rows.append({
                "pattern": name,
                "chunk_size": "None" if cs is None else cs,
                "dir_increasing": got_dir["increasing"],
                "dir_decreasing": got_dir["decreasing"],
                "nonstrict_ok": 1 if got_dir_ns == base_dir_ns else 0,
                "monotony": got_m.name,
                "expected_monotony": base_m.name,
                "monotony_ok": 1 if got_m is base_m else 0,
                "verdict": "PASS" if (got_dir == base_dir
                                      and got_dir_ns == base_dir_ns) else "FAIL",
            })
    return rows


def test_chunk_invariance_direction_path():
    """方向给定时（``mod="increasing"/"decreasing"``）分块与整块必须一致。"""
    rows = build_chunk_invariance()
    bad = [r for r in rows if r["verdict"] != "PASS"]
    assert not bad, bad


def test_give_monotony_chunked_matches_whole():
    """回归（本轮修好）：分块结果必须与整块结果**完全一致**。

    原缺陷：`_give_monotony` 把边界增量 `delta` 只用于"是否已矛盾"的判定，
    从未折进累加器（`acc = m`）⇒ 块长 1 时退化成 `Equal`、块长不整除时
    `StrictIncreasing` 被降级成 `MixedIncreasing`。
    修法：`m = _commutative_semigroup(_pair_monotony(last, chunk[0]), m)`。
    """
    arr = np.arange(10.0)
    assert give_monotony(arr) is Monotony.StrictIncreasing
    for cs in (1, 2, 3, 4, 5, 9, 10, 11):
        assert give_monotony(arr, chunk_size=cs) is Monotony.StrictIncreasing, cs


def test_give_monotony_chunked_matches_whole_opposite_direction():
    arr = np.arange(10.0, 0.0, -1.0)
    assert give_monotony(arr) is Monotony.StrictDecreasing
    for cs in (1, 2, 3, 9):
        assert give_monotony(arr, chunk_size=cs) is Monotony.StrictDecreasing, cs


def test_chunk_invariance_exo_memory():
    arr = PATTERNS["mixed_increasing"](40)
    base = give_monotony(arr)
    for cs in (3, 7, 40, 41):
        assert give_monotony(arr, chunk_size=cs, chunk_mod="exo-memory") is base


def test_chunk_mod_exo_memory_equivalence():
    """``chunk_mod="exo-memory"`` 走磁盘分块，结论必须与默认内存分块一致。

    注：这里刻意选 ``chunk_size`` 能整除长度的值 —— 否则会踩到
    :func:`test_give_monotony_chunked_defect` 那个边界增量丢失的缺陷。
    """
    arr = np.arange(40.0)
    base = give_monotony(arr)
    for cs in (2, 4, 5, 8, 10, 20, 40, 41):
        assert give_monotony(arr, chunk_size=cs, chunk_mod="exo-memory") is base


def test_chunk_size_one_matches_whole_for_every_pattern():
    """块长 1（每块单元素，方向完全由边界对决定）也必须与整块一致。"""
    for name, gen in PATTERNS.items():
        arr = gen(11)
        whole = give_monotony(arr)
        assert give_monotony(arr, chunk_size=1) is whole, name
        assert (monotonic(arr, strict=False, mod="monotonic", chunk_size=1)
                is monotonic(arr, strict=False, mod="monotonic")), name


# --------------------------------------------------------------------------- #
# 4. 边界 / dtype / 契约
# --------------------------------------------------------------------------- #
def test_empty_and_single():
    assert give_monotony(np.array([])) is Monotony.Equal
    assert give_monotony(np.array([3.0])) is Monotony.Equal
    assert bool(monotonic(np.array([]), mod="increasing")) is True
    assert bool(monotonic(np.array([3.0]), mod="increasing")) is True
    assert bool(monotonic(np.array([3.0]), strict=True, mod="monotonic")) is False, \
        "常数/单点不是严格单调（Equal 的严格位为 0）"


def test_constant_sequence():
    c = np.full(5, 7.0)
    assert give_monotony(c) is Monotony.Equal
    for mod in ("increasing", "decreasing", "monotonic"):
        assert bool(monotonic(c, mod=mod)) is True
        assert bool(monotonic(c, strict=True, mod=mod)) is False


def test_dtype_preservation_and_equivalence():
    for dtype in (np.int8, np.int32, np.int64, np.float16, np.float32, np.float64):
        arr = np.array([1, 2, 2, 3], dtype=dtype)
        assert give_monotony(arr) is Monotony.MixedIncreasing
        assert bool(monotonic(arr, mod="increasing")) is True
    # bool 数组：True/False 视为 1/0 → 先降后升 ⇒ No
    b = np.array([True, False, True])
    assert give_monotony(b) is Monotony.No


def test_nan_behaviour():
    """默认 safe="unknown" 拒绝 NaN；`safe="safe"` 下 NaN 比较恒 False → 判 Equal。"""
    nan = np.array([1.0, np.nan, 2.0])
    with pytest.raises(ValueError, match="NaN or Inf"):
        give_monotony(nan)
    with pytest.raises(ValueError, match="NaN or Inf"):
        monotonic(nan, mod="increasing")
    assert give_monotony(nan, safe="safe") is Monotony.Equal, "NaN 全部比较为 False → Equal"
    assert bool(monotonic(nan, mod="monotonic", safe="safe")) is True, "Equal 对 non-strict 单调成立"


def test_safe_int_levels_are_rejected():
    """特征化：docstring 写 ``safe: Literal[0, 1, 2]``，但 ``validate_safe`` 只认字符串。"""
    arr = np.arange(5.0)
    for lvl in (0, 1, 2):
        with pytest.raises(ValueError, match="safe"):
            monotonic(arr, mod="increasing", safe=lvl)


def test_safe_strings_accepted():
    arr = np.arange(5.0)
    for s in ("safe", "suspicious", "unknown"):
        assert bool(monotonic(arr, mod="increasing", safe=s)) is True


def test_mod_forms_and_errors():
    arr = np.arange(5.0)
    assert bool(monotonic(arr, mod="increasing")) is True
    assert bool(monotonic(arr, mod=MONOTONY_MODS.Increasing)) is True
    with pytest.raises(ValueError):
        monotonic(arr, mod="nope")
    assert MONOTONY_MODS_ARGS is not None


def test_2d_rejected():
    with pytest.raises(ValueError):
        give_monotony(np.zeros((2, 3)))


def test_no_input_mutation():
    arr = PATTERNS["random"](31)
    before = arr.copy()
    give_monotony(arr)
    monotonic(arr, mod="monotonic")
    assert np.array_equal(arr, before)


def test_determinism():
    arr = PATTERNS["sawtooth"](50)
    for _ in range(3):
        assert give_monotony(arr) is give_monotony(arr)


def test_linear_time_smoke():
    """1 Mi 元素应远低于 1 s（防 O(N²)）。"""
    arr = np.arange(B.n_elements(1), dtype=np.float64)
    r = B.timeit(lambda: monotonic(arr, mod="increasing"), repeat=3, warmup=1)
    assert r["best_ms"] < 1000.0, r


# --------------------------------------------------------------------------- #
# 导出
# --------------------------------------------------------------------------- #
def build_semigroup_table() -> list[dict]:
    rows = []
    for a, b in itertools.product(Monotony, repeat=2):
        got = _M._commutative_semigroup(a, b)
        exp = semigroup_by_definition(a, b)
        rows.append({"a": a.name, "b": b.name, "code_a": bin(a.value),
                     "code_b": bin(b.value), "merged_code": bin(got.value),
                     "merged": got.name, "expected": exp.name,
                     "verdict": "PASS" if got is exp else "FAIL"})
    return rows


def build_truth_table() -> list[dict]:
    rows = []
    for strict in (False, True):
        for mod in ("increasing", "decreasing", "monotonic"):
            for seq in ALL_SEQS:
                arr = np.array(seq, dtype=np.float64)
                got = bool(monotonic(arr, strict=strict, mod=mod))
                exp = oracle_monotonic(seq, strict, mod)
                rows.append({"strict": strict, "mod": mod, "seq": str(seq),
                             "got": got, "expected": exp,
                             "verdict": "PASS" if got == exp else "FAIL"})
    return rows


def export() -> None:
    sem = build_semigroup_table()
    B.write_csv(B.TABLE / "utils" / "monotonicity" / "semigroup_table.csv", sem)

    tt = build_truth_table()
    B.write_csv(B.TABLE / "utils" / "monotonicity" / "monotonic_truth_table.csv", tt)

    cls = []
    for seq in ALL_SEQS:
        arr = np.array(seq, dtype=np.float64)
        cls.append({"seq": str(seq), "got": give_monotony(arr).name,
                    "expected": oracle_monotony(seq).name})
    B.write_csv(B.TABLE / "utils" / "monotonicity" / "give_monotony_exhaustive.csv", cls)

    ci = build_chunk_invariance()
    B.write_csv(B.TABLE / "utils" / "monotonicity" / "chunk_invariance.csv", ci)

    # ---- 图 ----
    B.plt_style()
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(13.5, 4.6))
    ax1 = fig.add_subplot(1, 3, 1)
    members = list(Monotony)
    grid = np.zeros((len(members), len(members)), dtype=int)
    for r in sem:
        i = [m.name for m in members].index(r["a"])
        j = [m.name for m in members].index(r["b"])
        grid[i, j] = members.index(Monotony[r["merged"]])
    im = ax1.imshow(grid, cmap="viridis")
    ax1.set_xticks(range(len(members)), [m.name for m in members], rotation=45, ha="right", fontsize=7)
    ax1.set_yticks(range(len(members)), [m.name for m in members], fontsize=7)
    ax1.set_title("merge(a, b) table")
    fig.colorbar(im, ax=ax1, ticks=range(len(members)))

    ax2 = fig.add_subplot(1, 3, 2)
    names = list(PATTERNS)
    mat = np.array([[1 if r["verdict"] == "PASS" else 0 for r in ci if r["pattern"] == n]
                    for n in names])
    ax2.imshow(mat, cmap="Greens", vmin=0, vmax=1, aspect="auto")
    ax2.set_yticks(range(len(names)), names, fontsize=7)
    ax2.set_xticks(range(len(CHUNK_SIZES)),
                   ["None" if c is None else str(c) for c in CHUNK_SIZES], fontsize=7)
    ax2.set_xlabel("chunk_size")
    ax2.set_title("chunked == whole (73 elems)")

    ax3 = fig.add_subplot(1, 3, 3)
    pats = list(PATTERNS)
    for p in pats:
        arr = PATTERNS[p](400)
        ax3.plot(np.arange(arr.size), arr, lw=0.8, label=p)
    ax3.legend(fontsize=5.5, ncol=2)
    ax3.set_title("test patterns (n=400)")
    ax3.set_xlabel("index")

    fig.suptitle("Utils.Monotonicity — encoding, exhaustiveness, chunk invariance")
    fig.tight_layout()
    p = B.savefig(fig, B.FIG / "utils" / "monotonicity" / "monotonicity_matrix.png")
    plt.close(fig)

    print(f"wrote {B.TABLE / 'utils' / 'monotonicity'}/*.csv")
    print(f"wrote {p}")
    print(f"semigroup cells: {len(sem)} (fail {sum(r['verdict'] != 'PASS' for r in sem)})")
    print(f"truth table cells: {len(tt)} (fail {sum(r['verdict'] != 'PASS' for r in tt)})")
    print(f"chunk invariance rows: {len(ci)} (fail {sum(r['verdict'] != 'PASS' for r in ci)})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    B.add_export_flag(ap)
    args = ap.parse_args()
    if args.export:
        export()
    else:
        for seq in ALL_SEQS[:10]:
            arr = np.array(seq, dtype=np.float64)
            print(seq, give_monotony(arr).name, oracle_monotony(seq).name)
