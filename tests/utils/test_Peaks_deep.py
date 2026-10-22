"""``Utils.Peaks`` 深度测试：三后端 × 三峰型 × 过滤器语义。

重点是把"三后端应当一致"这件事**证出来**，并把它不一致的边界（平台 plateau、
未知 kwargs、短输入）固化成 characterization：

1. **后端一致性**：numpy vs numba 逐位相同；scipy 在非平台信号上相同，
   平台信号上按各自规则不同（docstring 已声明，这里量化差多少）；
2. **返回契约**：``int64`` 下标、``values is S[idx]``、``both`` 归并后严格升序、
   掩码与下标等长、``mod="min"`` 的取值仍是原信号；
3. **过滤器**：``height``（标量=下界 / 区间）、``threshold``（双侧）、
   ``distance``（贪心去重，``>=1``）；三个后端在同一参数下的等价性；
4. **短输入/退化**：size 0/1/2 的捷径、常量、平台、NaN、Inf；
5. **回退**：scipy/numba 缺失时 warn + 回退 numpy；numba 首次编译耗时。

导出::

    .\\.venv\\Scripts\\python.exe tests/utils/test_Peaks_deep.py --export

规模轴在 ``tests/size_axis/run_scale.py --tool peaks``。
"""

from __future__ import annotations

import argparse
import importlib
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

_TESTS = Path(__file__).resolve().parents[1]
if str(_TESTS) not in sys.path:
    sys.path.insert(0, str(_TESTS))

import _bench as B  # noqa: E402

B.bootstrap_src()

from Modal_Decomposition.Utils import PEAKS_BACKEND, PEAKS_MODS, find_peaks  # noqa: E402

_PeaksMod = importlib.import_module("Modal_Decomposition.Utils.Peaks")

BACKENDS = ("numpy", "scipy", "numba")
MODS = ("max", "min", "both")


def signal(kind: str, n: int = 400) -> np.ndarray:
    t = np.linspace(0, 1, n, endpoint=False)
    if kind == "smooth":
        return np.sin(2 * np.pi * 5 * t) + 0.3 * np.sin(2 * np.pi * 13 * t)
    if kind == "random":
        return np.random.default_rng(11).standard_normal(n)
    if kind == "plateau":
        s = np.sin(2 * np.pi * 4 * t)
        s[100:106] = 1.0                       # 平台
        s[200:203] = -1.0
        return s
    if kind == "constant":
        return np.full(n, 2.0)
    if kind == "ramp":
        return np.arange(n, dtype=float)
    raise ValueError(kind)  # pragma: no cover


# --------------------------------------------------------------------------- #
# 1. 后端一致性
# --------------------------------------------------------------------------- #
def build_backend_agreement() -> list[dict]:
    rows = []
    for name in ("smooth", "random", "plateau", "constant", "ramp"):
        S = signal(name)
        for mod in MODS:
            base = find_peaks(S, mod=mod, backend="numpy")
            for be in BACKENDS:
                got = find_peaks(S, mod=mod, backend=be)
                same_idx = np.array_equal(base[0], got[0])
                same_val = np.array_equal(base[1], got[1])
                same_mask = (len(base) == len(got)) if mod == "both" else True
                if mod == "both" and same_mask:
                    same_mask = np.array_equal(base[2], got[2])
                rows.append({
                    "signal": name, "mod": mod, "backend": be,
                    "n_peaks_numpy": int(base[0].size),
                    "n_peaks_backend": int(got[0].size),
                    "same_indices": int(same_idx),
                    "same_values": int(same_val),
                    "same_mask": int(same_mask),
                })
    return rows


def test_numpy_vs_numba_identical():
    """numpy 与 numba 共用同一份过滤代码 + 同一内核语义 ⇒ 必须逐位相同。"""
    bad = [r for r in build_backend_agreement()
           if r["backend"] == "numba"
           and not (r["same_indices"] and r["same_values"] and r["same_mask"])]
    assert not bad, bad


def test_scipy_agrees_off_plateau():
    bad = [r for r in build_backend_agreement()
           if r["backend"] == "scipy" and r["signal"] != "plateau"
           and not (r["same_indices"] and r["same_values"] and r["same_mask"])]
    assert not bad, bad


def test_plateau_differs_on_purpose():
    """平台信号：numpy 取平台**左沿**，scipy 取平台**中点**（docstring 已声明）。"""
    S = signal("plateau")
    numpy_idx = find_peaks(S, mod="max", backend="numpy")[0]
    scipy_idx = find_peaks(S, mod="max", backend="scipy")[0]
    assert 100 in numpy_idx and 102 in scipy_idx, (numpy_idx[:5], scipy_idx[:5])
    assert not np.array_equal(numpy_idx, scipy_idx)


# --------------------------------------------------------------------------- #
# 2. 返回契约
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("backend", BACKENDS)
def test_return_contract(backend):
    S = signal("smooth")
    for mod in ("max", "min"):
        idx, vals = find_peaks(S, mod=mod, backend=backend)
        assert idx.dtype == np.int64
        assert vals.dtype == S.dtype
        assert np.array_equal(vals, S[idx]), "values 必须是原信号幅值"
        assert np.all(np.diff(idx) > 0), "下标必须升序"
    idx, vals, is_max = find_peaks(S, mod="both", backend=backend)
    assert idx.dtype == np.int64 and is_max.dtype == bool
    assert idx.size == vals.size == is_max.size
    assert np.all(np.diff(idx) > 0), "both 归并后必须严格升序"
    assert np.array_equal(vals, S[idx])
    assert is_max.any() and (~is_max).any()
    # 极大子集与单独 mod="max" 一致
    assert np.array_equal(idx[is_max], find_peaks(S, mod="max", backend=backend)[0])


def test_both_merge_order_is_sample_order():
    """交替峰谷：``both`` 应把极大/极小按样本序交错产出（端点不看）。"""
    S = np.array([0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0])
    idx, vals, is_max = find_peaks(S, mod="both")
    assert np.array_equal(idx, np.array([1, 2, 3, 4, 5])), "端点 0/6 不参与检测"
    assert np.array_equal(is_max, np.array([True, False, True, False, True]))
    assert np.array_equal(vals, S[idx])


def test_min_values_not_negated():
    S = np.array([0.0, -5.0, 0.0, -2.0, 0.0])
    idx, vals = find_peaks(S, mod="min")
    assert np.array_equal(vals, np.array([-5.0, -2.0]))


# --------------------------------------------------------------------------- #
# 3. 过滤器
# --------------------------------------------------------------------------- #
def test_height_scalar_is_lower_bound():
    S = signal("smooth")
    idx_all = find_peaks(S, mod="max")[0]
    idx = find_peaks(S, mod="max", height=0.5)[0]
    assert set(idx) <= set(idx_all)
    assert np.all(S[idx] >= 0.5)


def test_height_interval():
    S = signal("smooth")
    idx = find_peaks(S, mod="max", height=(0.2, 0.8))[0]
    assert np.all((S[idx] >= 0.2) & (S[idx] <= 0.8))


def test_height_on_min_is_mirrored():
    S = signal("smooth")
    idx = find_peaks(S, mod="min", height=-0.5)[0]
    assert np.all(S[idx] <= -0.5), "极小方向 height 是上界（-S >= -0.5）"


def test_threshold():
    S = signal("smooth")
    idx_all = find_peaks(S, mod="max")[0]
    idx = find_peaks(S, mod="max", threshold=0.1)[0]
    assert set(idx) <= set(idx_all)
    for p in idx:
        assert S[p] - S[p - 1] >= 0.1 and S[p] - S[p + 1] >= 0.1


def test_distance_greedy():
    S = signal("smooth")
    idx18 = find_peaks(S, mod="max", distance=1)[0]
    idx = find_peaks(S, mod="max", distance=18)[0]
    assert set(idx) <= set(idx18)
    assert np.all(np.diff(idx) >= 18)


def test_distance_invalid():
    S = signal("smooth")
    for be in ("numpy", "numba"):
        with pytest.raises(ValueError, match="distance must be >= 1"):
            find_peaks(S, mod="max", backend=be, distance=0)


def test_filters_agree_across_backends():
    S = signal("smooth")
    for mod in ("max", "min"):
        for kw in ({"height": 0.3}, {"height": (0.1, 0.9)},
                   {"threshold": 0.05}, {"distance": 15},
                   {"height": 0.2, "distance": 12}):
            ref = find_peaks(S, mod=mod, backend="numpy", **kw)[0]
            for be in ("scipy", "numba"):
                got = find_peaks(S, mod=mod, backend=be, **kw)[0]
                assert np.array_equal(ref, got), (mod, be, kw, ref.size, got.size)


def test_both_filters_are_per_type():
    """``both`` 的 distance 按类型分别施加（极大只和极大比）。"""
    S = signal("smooth")
    idx, _, is_max = find_peaks(S, mod="both", distance=18)
    for flag in (True, False):
        sub = idx[is_max == flag]
        assert np.all(np.diff(sub) >= 18)


def test_unknown_kwargs_diverge_between_backends():
    """特征化（陷阱）：numpy/numba **静默忽略**未知过滤键，scipy 会真的施加。

    于是同一个调用换个后端就得到不同结果 —— ``plateau_size=2`` 时 numpy 返回全部
    9 个峰，scipy 只剩平台峰（本例 0 个）。docstring 只写了"其余键忽略"，
    没有警告这种跨后端口径差异。
    """
    S = signal("smooth")
    a = find_peaks(S, mod="max", backend="numpy", plateau_size=2)
    b = find_peaks(S, mod="max", backend="scipy", plateau_size=2)
    ref = find_peaks(S, mod="max", backend="numpy")
    assert np.array_equal(a[0], ref[0]), "numpy 忽略 plateau_size"
    assert b[0].size == 0, "scipy 施加 plateau_size（该信号没有长度 2 的平台）"
    # 完全无意义的键：numpy/numba 静默通过，scipy 抛 TypeError
    with pytest.raises(TypeError):
        find_peaks(S, mod="max", backend="scipy", no_such_filter=1)
    assert find_peaks(S, mod="max", backend="numpy", no_such_filter=1)[0].size == ref[0].size


# --------------------------------------------------------------------------- #
# 4. 短输入 / 退化
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n", [0, 1, 2])
def test_short_input_single_mod_returns_all(n):
    S = np.arange(n, dtype=float)
    for mod in ("max", "min"):
        idx, vals = find_peaks(S, mod=mod)
        assert np.array_equal(idx, np.arange(n))
        assert np.array_equal(vals, S)


@pytest.mark.parametrize("n", [0, 1, 2])
def test_short_input_both_returns_empty(n):
    S = np.arange(n, dtype=float)
    idx, vals, is_max = find_peaks(S, mod="both")
    assert idx.size == vals.size == is_max.size == 0


def test_constant_signal():
    S = signal("constant")
    assert find_peaks(S, mod="max")[0].size == 0
    assert find_peaks(S, mod="min")[0].size == 0
    assert find_peaks(S, mod="both")[0].size == 0


def test_ramp_signal():
    S = signal("ramp")
    assert find_peaks(S, mod="max")[0].size == 0
    assert find_peaks(S, mod="both")[0].size == 0


def test_noise_nan_policy():
    S = np.array([0.0, 1.0, 0.0, np.nan, 0.0, 1.0, 0.0])
    with pytest.raises(ValueError, match="NaN or Inf"):
        find_peaks(S, mod="max")
    with pytest.warns(None) if False else warnings.catch_warnings():
        warnings.simplefilter("ignore")
        idx, _ = find_peaks(S, mod="max", safe="safe")
    assert np.array_equal(idx, np.array([1, 5])), "NaN 参与比较恒 False，不会成为峰"


def test_inf_rejected():
    with pytest.raises(ValueError):
        find_peaks(np.array([0.0, np.inf, 0.0]), mod="max")


def test_suspicious_rejected():
    with pytest.raises(ValueError, match="suspicious"):
        find_peaks(np.arange(10.0), safe="suspicious")


def test_2d_rejected():
    with pytest.raises(ValueError):
        find_peaks(np.zeros((2, 5)), mod="max")


def test_list_input_accepted():
    idx, vals = find_peaks([0.0, 1.0, 0.0, 2.0, 0.0], mod="max")
    assert np.array_equal(idx, np.array([1, 3]))


def test_invalid_mod_and_backend():
    S = np.arange(10.0)
    with pytest.raises(ValueError):
        find_peaks(S, mod="nope")
    with pytest.raises(ValueError):
        find_peaks(S, backend="nope")


def test_enum_arguments():
    S = signal("smooth")
    a = find_peaks(S, mod=PEAKS_MODS.Max, backend=PEAKS_BACKEND.Numpy)
    b = find_peaks(S, mod="max", backend="numpy")
    assert np.array_equal(a[0], b[0])


# --------------------------------------------------------------------------- #
# 5. 回退
# --------------------------------------------------------------------------- #
def test_scipy_missing_falls_back(monkeypatch):
    class _NoneCache:
        @staticmethod
        def import_module(name, **kw):
            return None

    monkeypatch.setattr(_PeaksMod, "cache", _NoneCache)
    S = signal("smooth")
    with pytest.warns(UserWarning, match="scipy.signal"):
        idx, _ = find_peaks(S, mod="max", backend="scipy")
    assert np.array_equal(idx, find_peaks(S, mod="max", backend="numpy")[0])


def test_numba_missing_falls_back(monkeypatch):
    class _NoneCache:
        @staticmethod
        def import_module(name, **kw):
            return None

    monkeypatch.setattr(_PeaksMod, "cache", _NoneCache)
    S = signal("smooth")
    with pytest.warns(UserWarning, match="numba"):
        idx, _ = find_peaks(S, mod="max", backend="numba")
    assert np.array_equal(idx, find_peaks(S, mod="max", backend="numpy")[0])


def test_numba_compiled_once():
    if _PeaksMod.cache.import_module("numba", verbose=False) is None:
        pytest.skip("numba unavailable; backend falls back to numpy")
    S = signal("smooth")
    find_peaks(S, mod="max", backend="numba")
    c1 = _PeaksMod._get_compiled.cache_info()
    find_peaks(S, mod="max", backend="numba")
    c2 = _PeaksMod._get_compiled.cache_info()
    assert c2.currsize == 1 and c2.misses == c1.misses, (c1, c2)


# --------------------------------------------------------------------------- #
# 6. 契约面 + 复杂度
# --------------------------------------------------------------------------- #
def test_public_surface():
    import Modal_Decomposition.Utils as U

    for name in ("find_peaks", "PEAKS_MODS", "PEAKS_BACKEND", "PEAKS_SUPPORTED_FILTERS"):
        assert getattr(U, name) is getattr(_PeaksMod, name)
    assert set(_PeaksMod.__all__) == {
        "find_peaks", "PEAKS_MODS", "PEAKS_MODS_ARGS", "PEAKS_BACKEND",
        "PEAKS_BACKEND_ARGS", "PEAKS_SUPPORTED_FILTERS", "PEAKS_SUPPORTED_FILTERS_ARGS"}


def test_no_input_mutation():
    S = signal("random")
    before = S.copy()
    for be in BACKENDS:
        for mod in MODS:
            find_peaks(S, mod=mod, backend=be)
    assert np.array_equal(S, before)


def test_deterministic():
    S = signal("random")
    a = find_peaks(S, mod="both")
    b = find_peaks(S, mod="both")
    assert all(np.array_equal(x, y) for x, y in zip(a, b))


def test_linear_time():
    S = B.make_signal(1)
    r = B.timeit(lambda: find_peaks(S, mod="both", backend="numpy"), repeat=3, warmup=1)
    assert r["best_ms"] < 1000.0, r


# --------------------------------------------------------------------------- #
# 导出
# --------------------------------------------------------------------------- #
def export() -> None:
    rows = build_backend_agreement()
    B.write_csv(B.TABLE / "utils" / "peaks" / "backend_agreement.csv", rows)

    surf = []
    for name in ("smooth", "random", "plateau", "constant"):
        S = signal(name)
        for mod in MODS:
            for be in BACKENDS:
                idx = find_peaks(S, mod=mod, backend=be)[0]
                surf.append({"signal": name, "mod": mod, "backend": be,
                             "n_peaks": int(idx.size),
                             "first_peaks": str(idx[:4].tolist())})
    B.write_csv(B.TABLE / "utils" / "peaks" / "peak_counts.csv", surf)

    import time as _t

    find_peaks(signal("smooth"), backend="numba")      # 预热编译
    S = signal("random", 200_000)
    tim = []
    for be in BACKENDS:
        for mod in MODS:
            r = B.timeit(lambda: find_peaks(S, mod=mod, backend=be), repeat=3, warmup=1)
            tim.append({"backend": be, "mod": mod, "n": S.size,
                        "best_ms": round(r["best_ms"], 3),
                        "n_peaks": int(find_peaks(S, mod=mod, backend=be)[0].size)})
    B.write_csv(B.TABLE / "utils" / "peaks" / "backend_timing.csv", tim)

    B.plt_style()
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    ax = axes[0]
    labels = [f"{r['mod']}\n{r['backend']}" for r in tim]
    ax.bar(range(len(tim)), [r["best_ms"] for r in tim])
    ax.set_xticks(range(len(tim)), labels, fontsize=6, rotation=90)
    ax.set_ylabel("best ms (n=200k)")
    ax.set_title("find_peaks backend timing")

    ax = axes[1]
    names = sorted({r["signal"] for r in surf})
    width = 0.25
    for k, be in enumerate(BACKENDS):
        vals = [np.mean([r["n_peaks"] for r in surf if r["signal"] == nm and r["backend"] == be])
                for nm in names]
        ax.bar(np.arange(len(names)) + (k - 1) * width, vals, width, label=be)
    ax.set_xticks(range(len(names)), names, fontsize=8)
    ax.set_ylabel("mean #peaks over mods")
    ax.set_title("peak counts per backend")
    ax.legend(fontsize=7)
    fig.tight_layout()
    p = B.savefig(fig, B.FIG / "utils" / "peaks" / "peaks_backends.png")
    plt.close(fig)
    print(f"wrote {B.TABLE / 'utils' / 'peaks'}/*.csv")
    print(f"wrote {p}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    B.add_export_flag(ap)
    args = ap.parse_args()
    if args.export:
        export()
    else:
        B.print_table(build_backend_agreement())
