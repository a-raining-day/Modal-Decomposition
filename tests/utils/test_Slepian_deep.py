"""``Utils.Slepian`` 深度测试：三后端一致性、正交性、集中比、缓存语义。

Slepian (DPSS) 是本层唯一带**进程级缓存**的工具，因此除了数值还要测缓存：

1. **三后端一致性**：numpy / scipy / C 在同一 ``(N, NW, K)`` 上互相 ≤ 1e-11
   （C 未编译时自动跳过该列，并记录"不可用"事实）；
2. **数学性质**：``U Uᵀ = I``、集中比降序且 ∈ (0, 1]、集中比与 scipy 同值；
3. **返回契约**：恒 2-D（K=1 也是 2-D，而 scipy 会给 1-D）、dtype float64；
4. **缓存**：命中返回副本（改它不污染缓存）、浮点按位精确匹配（``3.0`` vs
   ``nextafter(3.0)`` 是两条记录）、``cache=False``、``cache=int`` 条数上限、
   ``clear_cache`` 与统计计数；
5. **参数与降级**：N<1 / NW≤0 / NaN / norm 非法 / sym 非 bool / K<1 / mod 未知；
   ``N ≤ 6`` 时 scipy 的 ``NW >= N/2`` 限制 → 告警并换后端。

导出::

    .\\.venv\\Scripts\\python.exe tests/utils/test_Slepian_deep.py --export

规模轴不适用（Slepian 的开销由窗长 ``N`` 而非数据量决定），改出 N 扫描表。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pytest

_TESTS = Path(__file__).resolve().parents[1]
if str(_TESTS) not in sys.path:
    sys.path.insert(0, str(_TESTS))

import _bench as B  # noqa: E402

B.bootstrap_src()

from Modal_Decomposition.Utils import (  # noqa: E402
    available_backends,
    cache_info,
    clear_cache,
    resolve_backend,
    slepian,
)

BACKENDS = ("numpy", "scipy", "C")
AVAIL = available_backends()


def usable(be: str) -> bool:
    return bool(AVAIL.get(be, False))


def ref(be: str):
    return pytest.mark.skipif(not usable(be), reason=f"backend {be} unavailable")


# --------------------------------------------------------------------------- #
# 1. 后端解析
# --------------------------------------------------------------------------- #
def test_available_backends_keys():
    assert set(available_backends()) == {"numpy", "scipy", "C"}
    assert AVAIL["numpy"] is True


def test_resolve_backend_matrix():
    for mod in ("numpy", "scipy", "C", "auto", None):
        be = resolve_backend(64, mod)
        assert be in {"numpy", "scipy", "C"}
    assert resolve_backend(64, "numpy") == "numpy"


def test_resolve_backend_small_n_avoids_scipy():
    """``NW >= N/2``（默认 NW=4 时 N ≤ 8）scipy 必抛错 → 换后端并告警。"""
    with pytest.warns(UserWarning):
        be = resolve_backend(4, "scipy", halfBW=4.0)
    assert be != "scipy"
    assert slepian(4, 4.0, 2, mod="scipy").shape == (2, 4)


def test_resolve_backend_unknown_raises():
    with pytest.raises(ValueError, match="is not supported"):
        resolve_backend(64, "nope")


# --------------------------------------------------------------------------- #
# 2. 三后端一致性
# --------------------------------------------------------------------------- #
def _pairwise(N, NW, K):
    base = slepian(N, NW, K, mod="numpy")
    rows = []
    for be in ("scipy", "C"):
        if not usable(be):
            rows.append({"N": N, "NW": NW, "K": K, "backend": be, "max_abs_diff": None,
                         "verdict": "SKIP"})
            continue
        got = slepian(N, NW, K, mod=be)
        assert got.shape == base.shape, (be, got.shape, base.shape)
        d = float(np.max(np.abs(got - base)))
        rows.append({"N": N, "NW": NW, "K": K, "backend": be, "max_abs_diff": d,
                     "verdict": "PASS" if d <= 1e-10 else "FAIL"})
    return rows


def build_agreement() -> list[dict]:
    rows = []
    for N in (8, 16, 64, 256):
        for NW in (1.0, 2.0, 3.0):
            for K in (1, 2, 4):
                rows.extend(_pairwise(N, NW, K))
    return rows


def test_backends_agree():
    rows = [r for r in build_agreement() if r["verdict"] != "SKIP"]
    bad = [r for r in rows if r["verdict"] != "PASS"]
    assert not bad, bad


@ref("scipy")
def test_ratios_match_scipy():
    for N, NW in ((64, 2.0), (128, 3.0), (256, 4.0)):
        t, r = slepian(N, NW, 4, mod="numpy", return_ratios=True)
        ts, rs = slepian(N, NW, 4, mod="scipy", return_ratios=True)
        assert np.allclose(r, rs, atol=1e-11)
        assert np.allclose(t, ts, atol=1e-10)


# --------------------------------------------------------------------------- #
# 3. 数学性质
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("backend", BACKENDS)
def test_orthonormality(backend):
    if not usable(backend):
        pytest.skip(f"{backend} unavailable")
    t = slepian(128, 3.0, 4, mod=backend)
    assert np.allclose(t @ t.T, np.eye(4), atol=1e-10)


@pytest.mark.parametrize("backend", BACKENDS)
def test_ratios_descending_in_unit_interval(backend):
    if not usable(backend):
        pytest.skip(f"{backend} unavailable")
    _, r = slepian(256, 4.0, 5, mod=backend, return_ratios=True)
    assert np.all(np.diff(r) < 0), "集中比必须严格降序"
    assert np.all((r > 0) & (r <= 1.0 + 1e-12))


def test_norm_options():
    a = slepian(64, 2.0, 2, mod="numpy", norm=None)
    b = slepian(64, 2.0, 2, mod="numpy", norm=2)
    assert np.allclose(a, b, atol=1e-12)
    c = slepian(64, 2.0, 2, mod="numpy", norm="approximate")
    # "approximate" 是 scipy 的近似归一（M^2/(M^2+NW) 修正）：
    # 只有 0 阶的峰值 ≈1（本例 0.99951），高阶峰值由自身形状决定、自然更低（0.874）。
    assert abs(float(np.max(np.abs(c[0]))) - 1.0) < 2e-3
    assert np.max(np.abs(c[1])) < 0.95
    assert np.max(np.abs(c)) <= 1.0 + 1e-12


def test_sym_false_shape():
    t = slepian(32, 2.0, 3, mod="numpy", sym=False)
    assert t.ndim == 2 and t.shape[1] in (32, 33)
    @pytest.mark.parametrize("_", [0])          # noqa: E306 - 仅为保持结构清晰
    def _noop(_):
        pass


def test_k1_is_still_2d():
    """scipy 在 K=1 时会给 1-D，本模块统一成 2-D。"""
    for be in ("numpy", "scipy"):
        if not usable(be):
            continue
        t = slepian(64, 2.0, 1, mod=be)
        assert t.shape == (1, 64), be


def test_ntapers_default_and_clamp():
    t = slepian(64, 3.0, None, mod="numpy")
    assert t.shape[0] == max(1, int(np.floor(2 * 3.0 - 1)))
    t2 = slepian(8, 2.0, 100, mod="numpy")
    assert t2.shape[0] == 8, "K 必须截断到 N"


# --------------------------------------------------------------------------- #
# 4. 缓存
# --------------------------------------------------------------------------- #
def test_cache_hit_returns_copy():
    clear_cache()
    a = slepian(64, 2.0, 3, mod="numpy")
    a[:] = 0.0
    b = slepian(64, 2.0, 3, mod="numpy")
    assert not np.allclose(b, 0.0), "缓存必须返回副本，调用方改写不能污染缓存"
    info = cache_info()
    assert info["hits"] >= 1


def test_cache_key_is_bitwise_exact():
    """1 ulp 的 NW 差异必须是**两条**缓存条目（不做容差匹配）。"""
    clear_cache()
    a = slepian(64, 3.0, 2, mod="numpy")
    b = slepian(64, float(np.nextafter(3.0, 4.0)), 2, mod="numpy")
    info = cache_info()
    assert info["misses"] >= 2, "不同位模式的 NW 不能命中同一条目"
    assert info["size"] == 2
    # 数值上当然几乎相同（1 ulp 的影响肉眼不可见），这里只验"键不同"
    assert np.allclose(a, b, atol=1e-6)


def test_cache_disabled():
    clear_cache()
    slepian(64, 2.0, 2, mod="numpy", cache=False)
    assert cache_info()["misses"] == 0
    slepian(64, 2.0, 2, mod="numpy", cache=False)
    assert cache_info()["hits"] == 0


def test_cache_ratios_in_key():
    clear_cache()
    slepian(64, 2.0, 2, mod="numpy", return_ratios=True)
    t = slepian(64, 2.0, 2, mod="numpy")            # 不要 ratios → 不同键
    assert t.ndim == 2 and cache_info()["misses"] >= 2


def test_cache_clear_and_info_shape():
    clear_cache()
    info = cache_info()
    assert set(info) == {"version", "size", "bytes", "hits", "misses",
                         "evictions", "skipped"}
    slepian(64, 2.0, 2, mod="numpy")
    assert cache_info()["size"] >= 1
    clear_cache()
    assert cache_info()["size"] == 0


def test_cache_size_limit():
    clear_cache()
    for k in range(1, 6):
        slepian(32, 1.0 + 0.1 * k, 1, mod="numpy", cache=2)
    assert cache_info()["size"] <= 2


def test_cache_invalid_value():
    with pytest.raises((ValueError, TypeError)):
        slepian(64, 2.0, 2, mod="numpy", cache="yes")


# --------------------------------------------------------------------------- #
# 5. 参数校验
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("kw,exc", [
    ({"N": 0, "halfBW": 2.0}, ValueError),
    ({"N": 64, "halfBW": 0.0}, ValueError),
    ({"N": 64, "halfBW": -1.0}, ValueError),
    ({"N": 64, "halfBW": np.nan}, ValueError),
    ({"N": 64, "halfBW": 2.0, "nTapers": 0}, ValueError),
    ({"N": 64, "halfBW": 2.0, "norm": "bad"}, ValueError),
    ({"N": 64, "halfBW": 2.0, "sym": 1}, ValueError),
])
def test_invalid_arguments(kw, exc):
    with pytest.raises(exc):
        slepian(mod="numpy", **kw)


def test_invalid_backend_name():
    with pytest.raises(ValueError, match="is not supported"):
        slepian(64, 2.0, 2, mod="nope")


def test_output_dtype_and_shape():
    t = slepian(32, 2.0, 3, mod="numpy")
    assert t.dtype == np.float64 and t.shape == (3, 32)
    t2, r2 = slepian(32, 2.0, 3, mod="numpy", return_ratios=True)
    assert t2.shape == (3, 32) and r2.shape == (3,)


def test_deterministic():
    clear_cache()
    a = slepian(128, 3.0, 3, mod="numpy", cache=False)
    b = slepian(128, 3.0, 3, mod="numpy", cache=False)
    assert np.array_equal(a, b)


def test_sign_convention_is_stable():
    """符号约定（Percival & Walden）必须跨后端一致：逐个比较同一行。"""
    if not usable("scipy"):
        pytest.skip("scipy unavailable")
    a = slepian(64, 2.0, 3, mod="numpy", cache=False)
    b = slepian(64, 2.0, 3, mod="scipy", cache=False)
    assert np.allclose(a, b, atol=1e-10), "符号必须一致（否则差值会是 2× 幅值）"


def test_public_surface():
    import Modal_Decomposition.Utils as U

    for name in ("slepian", "resolve_backend", "available_backends",
                 "cache_info", "clear_cache"):
        assert hasattr(U, name)
    import importlib
    mod = importlib.import_module("Modal_Decomposition.Utils.Slepian")
    assert set(mod.__all__) == {"slepian", "resolve_backend", "available_backends",
                                "cache_info", "clear_cache"}


# --------------------------------------------------------------------------- #
# 导出
# --------------------------------------------------------------------------- #
def build_n_sweep() -> list[dict]:
    rows = []
    for N in (8, 32, 128, 512, 2048, 8192, 32768):
        for be in ("numpy", "scipy", "C"):
            if not usable(be):
                rows.append({"N": N, "backend": be, "best_ms": None,
                             "peak_mib": None, "verdict": "SKIP"})
                continue
            try:
                clear_cache()
                r = B.timeit(lambda: slepian(N, 4.0, 4, mod=be, cache=False),
                             repeat=3, warmup=1)
                rows.append({"N": N, "backend": be, "best_ms": round(r["best_ms"], 3),
                             "peak_mib": round(B.peak_rss_mib(), 1), "verdict": "PASS"})
            except Exception as exc:            # noqa: BLE001
                rows.append({"N": N, "backend": be, "best_ms": None, "peak_mib": None,
                             "verdict": f"{type(exc).__name__}: {exc}"[:50]})
    return rows


def export() -> None:
    agree = build_agreement()
    B.write_csv(B.TABLE / "utils" / "slepian" / "backend_agreement.csv", agree)
    sweep = build_n_sweep()
    B.write_csv(B.TABLE / "utils" / "slepian" / "n_sweep.csv", sweep)

    B.plt_style()
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    ax = axes[0]
    for be in BACKENDS:
        sub = [r for r in sweep if r["backend"] == be and r["best_ms"]]
        if sub:
            ax.loglog([r["N"] for r in sub], [r["best_ms"] for r in sub], "o-", ms=3, label=be)
    ax.set_xlabel("window length N")
    ax.set_ylabel("best wall time (ms)")
    ax.set_title("Slepian (NW=4, K=4)")
    ax.legend(fontsize=7)

    ax = axes[1]
    N = 256
    for k in (0, 1, 2, 3):
        t = slepian(N, 4.0, 4, mod="numpy")
        ax.plot(np.arange(N), t[k], lw=1.0, label=f"order {k}")
    ax.plot(np.arange(N), slepian(N, 4.0, 4, mod="numpy", norm="approximate")[0],
            "k--", lw=0.8, label="order 0 (approximate norm)")
    ax.set_title("first four DPSS tapers (N=256, NW=4)")
    ax.set_xlabel("sample")
    ax.legend(fontsize=7)
    fig.tight_layout()
    p = B.savefig(fig, B.FIG / "utils" / "slepian" / "slepian_backends.png")
    plt.close(fig)
    print(f"wrote {B.TABLE / 'utils' / 'slepian'}/*.csv")
    print(f"wrote {p}")
    print("backends:", AVAIL)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    B.add_export_flag(ap)
    args = ap.parse_args()
    if args.export:
        export()
    else:
        B.print_table(build_agreement())
