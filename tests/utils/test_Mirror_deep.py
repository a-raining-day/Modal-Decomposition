"""``Utils.Mirror`` 深度测试：五种端点延拓的几何不变量。

五种模式里只有 ``"mirror"`` 是纯样本操作（可与 ``np.pad(mode="reflect")`` 逐位对照），
其余四种都是"检出极值 → 外延极值表 → 返回 ``(T, A, is_max)``"。因此测试重心是
**几何不变量**而不是数值近似：

* ``T`` 严格递增、无重复时间戳（样条可用性的前提）；
* 端部点确实落在中段之外；
* 端部型别（极大/极小）与中段边界极值相反、且向外交替；
* ``is_max`` 与 ``find_peaks(..., mod="both")`` 的中段一致；
* 长度恒等式：``extrema/polynomial/parallel`` → ``L + 2*nbsym``，``BLCSE`` → ``L + 4``。

导出::

    .\\.venv\\Scripts\\python.exe tests/utils/test_Mirror_deep.py --export

规模轴在 ``tests/size_axis/run_scale.py --tool mirror``。
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

from Modal_Decomposition.Utils import MIRROR_MODS, find_peaks, mirror  # noqa: E402

_Mirror = importlib.import_module("Modal_Decomposition.Utils.Mirror")

EXTRAPOLATING = ("extrema", "polynomial", "parallel", "BLCSE")


def wave(n: int = 601) -> np.ndarray:
    """6 个周期 → 12 个极值（足够 nbsym 取到 5）。"""
    return np.sin(np.linspace(0.0, 12.0 * np.pi, n))


# --------------------------------------------------------------------------- #
# 1. mod="mirror"：与 np.pad(reflect) 逐位一致
# --------------------------------------------------------------------------- #
def test_mirror_matches_numpy_reflect():
    S = np.arange(10.0)
    for nbsym in range(1, 10):
        got = mirror(S, mod="mirror", nbsym=nbsym)
        ref = np.pad(S, (nbsym, nbsym), mode="reflect")
        assert np.array_equal(got, ref), nbsym
        assert got.size == S.size + 2 * nbsym


def test_mirror_docstring_example():
    assert mirror(np.arange(6.0), nbsym=2).tolist() == [2.0, 1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0]


def test_mirror_nbsym_zero_returns_copy():
    S = np.arange(6.0)
    out = mirror(S, mod="mirror", nbsym=0)
    assert np.array_equal(out, S) and out is not S


def test_mirror_single_sample_and_large_nbsym():
    assert np.array_equal(mirror(np.array([5.0]), nbsym=3), np.full(7, 5.0))
    S = np.arange(4.0)
    out = mirror(S, mod="mirror", nbsym=6)      # nbsym > size-1 → np.pad(reflect)
    assert np.array_equal(out, np.pad(S, (6, 6), mode="reflect"))


def test_mirror_empty_rejected():
    with pytest.raises(ValueError, match="must not be empty"):
        mirror(np.array([]), mod="mirror")


def test_mirror_keeps_dtype():
    assert mirror(np.arange(6, dtype=np.int32), nbsym=2).dtype == np.int32


def test_mirror_ignores_T():
    S = np.arange(6.0)
    t = np.linspace(0, 1, 6)
    assert np.array_equal(mirror(S, T=t, nbsym=2), mirror(S, nbsym=2))


# --------------------------------------------------------------------------- #
# 2. 极值域模式的几何不变量
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("mod", EXTRAPOLATING)
@pytest.mark.parametrize("nbsym", [1, 2, 3, 4])
def test_geometry_invariants(mod, nbsym):
    S = wave()
    extra = {"poly_order": 1, "time_poly_order": 1} if mod == "polynomial" else {}
    T, A, is_max = mirror(S, mod=mod, nbsym=nbsym, **extra)
    assert T.ndim == 1 and A.shape == T.shape and is_max.shape == T.shape
    assert np.all(np.diff(T) > 0), "时间轴必须严格递增（否则样条不可用）"
    assert np.all(np.isfinite(T)) and np.all(np.isfinite(A))

    core_T, core_A, core_is_max = find_peaks(S, mod="both")
    L = core_T.size
    if mod == "BLCSE":
        assert T.size == L + 4
        assert np.array_equal(A[2:2 + L], core_A)
        assert np.array_equal(is_max[2:2 + L], core_is_max)
        assert np.all(T[:2] < core_T[0]) and np.all(T[2 + L:] > core_T[-1])
        left_types, right_types = is_max[:2], is_max[2 + L:]
    else:
        assert T.size == L + 2 * nbsym
        assert np.array_equal(A[nbsym:nbsym + L], core_A)
        assert np.array_equal(is_max[nbsym:nbsym + L], core_is_max)
        if nbsym:
            assert np.all(T[:nbsym] < core_T[0])
            assert np.all(T[nbsym + L:] > core_T[-1])
        left_types, right_types = is_max[:nbsym], is_max[nbsym + L:]

    if mod == "extrema" and nbsym:
        # 镜像型别：与镜像源一致（左端取 core 的第 nbsym..1 个）
        assert np.array_equal(left_types, core_is_max[nbsym:0:-1])
        assert np.array_equal(right_types, core_is_max[-2:-nbsym - 2:-1])
    if mod in ("polynomial", "parallel") and nbsym:
        # 合成点的型别：从边界极值出发向外交替（最近的一个与边界相反）
        assert left_types[-1] != core_is_max[0]
        assert right_types[0] != core_is_max[-1]
        for k in range(1, nbsym):
            assert left_types[-1 - k] != left_types[-k]
            assert right_types[k] != right_types[k - 1]


def test_extrema_length_identity():
    S = wave()
    L = find_peaks(S, mod="both")[0].size
    for nbsym in (0, 1, 3, 5):
        T, A, _ = mirror(S, mod="extrema", nbsym=nbsym)
        assert T.size == L + 2 * nbsym
        Tq, Aq, _ = mirror(S, mod="parallel", nbsym=nbsym)
        assert Tq.size == L + 2 * nbsym
    for nbsym in (2, 3, 5):
        Tp, Ap, _ = mirror(S, mod="polynomial", nbsym=nbsym)   # 默认 poly_order=2 需 nbsym>=2
        assert Tp.size == L + 2 * nbsym
    Tb, Ab, _ = mirror(S, mod="BLCSE", nbsym=2)
    assert Tb.size == L + 4


def test_polynomial_needs_nbsym_ge_order():
    """特征化（参数陷阱）：``poly_order`` 默认 2，所以 ``nbsym=1`` 时直接失败。"""
    S = wave()
    with pytest.raises(ValueError, match="<= nbsym"):
        mirror(S, mod="polynomial", nbsym=1)
    assert mirror(S, mod="polynomial", nbsym=2)[0].size > 0


def test_nbsym_zero_returns_core_only():
    S = wave()
    core_T, core_A, core_is_max = find_peaks(S, mod="both")
    for mod in ("extrema", "parallel"):          # polynomial 的 nbsym=0 不合法
        T, A, m = mirror(S, mod=mod, nbsym=0)
        assert np.array_equal(T, core_T) and np.array_equal(A, core_A)
        assert np.array_equal(m, core_is_max)


def test_extrema_actual_values():
    """``extrema`` 是纯镜像：左端 T = 2*T0 - T[nbsym..1]，幅值与型别照抄。"""
    S = wave()
    nbsym = 3
    T, A, is_max = mirror(S, mod="extrema", nbsym=nbsym)
    core_T, core_A, core_is_max = find_peaks(S, mod="both")
    assert np.allclose(T[:nbsym], 2 * core_T[0] - core_T[nbsym:0:-1])
    assert np.allclose(A[:nbsym], core_A[nbsym:0:-1])
    assert np.array_equal(is_max[:nbsym], core_is_max[nbsym:0:-1])


def test_blcse_endpoints_are_same_class_means():
    S = wave()
    T, A, is_max = mirror(S, mod="BLCSE", nbsym=2)
    core_T, core_A, core_is_max = find_peaks(S, mod="both")
    L = core_T.size
    max_a, min_a = core_A[core_is_max], core_A[~core_is_max]
    left_max = A[:2][is_max[:2]]
    left_min = A[:2][~is_max[:2]]
    assert np.allclose(left_max, max_a[:min(2, max_a.size)].mean())
    assert np.allclose(left_min, min_a[:min(2, min_a.size)].mean())


# --------------------------------------------------------------------------- #
# 3. 显式 T
# --------------------------------------------------------------------------- #
def test_explicit_nonuniform_T():
    S = wave()
    t = np.geomspace(1e-3, 1.0, S.size)
    T, A, _ = mirror(S, mod="extrema", nbsym=2, T=t)
    core_idx = find_peaks(S, mod="both")[0]
    assert np.allclose(A[2:-2], S[core_idx])
    assert np.all(np.diff(T) > 0)


def test_decreasing_T_rearranged_with_warning():
    S = wave()
    t = np.linspace(1.0, 0.0, S.size)          # 递减
    with pytest.warns(UserWarning, match="not increasing"):
        T, A, is_max = mirror(S, mod="extrema", nbsym=2, T=t, verbose=True)
    assert np.all(np.diff(T) > 0), "重排后必须递增"


def test_nonuniform_T_warns_only_when_verbose():
    S = wave()
    t = np.geomspace(1e-3, 1.0, S.size)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        mirror(S, mod="extrema", nbsym=2, T=t)          # 不该告警
    with pytest.warns(UserWarning, match="not uniform"):
        mirror(S, mod="extrema", nbsym=2, T=t, verbose=True)


def test_T_length_mismatch():
    with pytest.raises(ValueError, match="same size"):
        mirror(wave(), mod="extrema", nbsym=2, T=np.linspace(0, 1, 10))


def test_T_2d_rejected():
    with pytest.raises(ValueError):
        mirror(wave(), mod="extrema", nbsym=2, T=np.zeros((2, 200)))


# --------------------------------------------------------------------------- #
# 4. 参数与退化
# --------------------------------------------------------------------------- #
def test_short_signal_rejected():
    for n in (0, 1, 2):
        with pytest.raises(ValueError, match="at least 3"):
            mirror(np.arange(n, dtype=float), mod="extrema")


def test_constant_signal_has_no_extrema():
    with pytest.raises(ValueError, match="no extrema"):
        mirror(np.full(50, 2.0), mod="extrema", nbsym=1)


def test_ramp_signal_has_no_extrema():
    with pytest.raises(ValueError, match="no extrema"):
        mirror(np.arange(50.0), mod="extrema", nbsym=1)


def test_nbsym_type_and_range():
    S = wave()
    with pytest.raises(TypeError):
        mirror(S, mod="extrema", nbsym=1.0)
    with pytest.raises(TypeError):
        mirror(S, mod="extrema", nbsym=True)
    with pytest.raises(ValueError):
        mirror(S, mod="extrema", nbsym=-1)


def test_nbsym_too_large():
    S = np.array([0.0, 1.0, 0.0, -1.0, 0.0])      # 3 个极值
    with pytest.raises(ValueError, match="nbsym must be <= "):
        mirror(S, mod="extrema", nbsym=5)


def test_int_dtype_is_promoted_not_rejected():
    """特征化：docstring 说非浮点 dtype 抛 ValueError，实际是**静默提升**到 float64。"""
    S = (wave() * 100).astype(np.int32)
    T, A, _ = mirror(S, mod="extrema", nbsym=1)
    assert A.dtype == np.float64 and T.dtype == np.float64
    # mirror 模式则原样保留整型
    assert mirror(np.arange(6, dtype=np.int32), nbsym=2).dtype == np.int32


def test_dtype_override():
    S = wave().astype(np.float32)
    T, A, _ = mirror(S, mod="extrema", nbsym=1, dtype=np.float64)
    assert A.dtype == np.float64 and T.dtype == np.float64


def test_bad_kwargs():
    S = wave()
    with pytest.raises(TypeError):
        mirror(S, verbose=1)
    with pytest.raises(TypeError):
        mirror(S, mod="extrema", squeeze=1)
    with pytest.raises(TypeError):
        mirror(S, dtype="not-a-dtype")
    with pytest.raises(ValueError):
        mirror(S, mod="nope")
    with pytest.raises(ValueError):
        mirror(S, mod="extrema", peaks_backend="nope")


def test_polynomial_order_checks():
    S = wave()
    with pytest.raises(TypeError):
        mirror(S, mod="polynomial", nbsym=3, poly_order=1.0)
    with pytest.raises(ValueError):
        mirror(S, mod="polynomial", nbsym=3, poly_order=0)
    with pytest.raises(ValueError, match="<= nbsym"):
        mirror(S, mod="polynomial", nbsym=1, poly_order=3)
    with pytest.raises(ValueError, match="<= nbsym"):
        mirror(S, mod="polynomial", nbsym=1, time_poly_order=3)


def test_blcse_needs_both_types():
    with pytest.raises(ValueError, match="both the maxima and the minima"):
        mirror(np.arange(20.0), mod="BLCSE", nbsym=1)


def test_blcse_nbsym_zero_rejected():
    with pytest.raises(ValueError, match="nbsym must be >= 1"):
        mirror(wave(), mod="BLCSE", nbsym=0)


def test_parallel_needs_two_extrema():
    S = np.array([0.0, 1.0, 1.0, 1.0, 1.0])       # 只有 1 个极大，无极小
    assert find_peaks(S, mod="both")[0].size == 1
    with pytest.raises(ValueError, match="at least 2 extrema"):
        mirror(S, mod="parallel", nbsym=1)


def test_2d_rejected():
    with pytest.raises(ValueError):
        mirror(np.zeros((2, 50)), mod="mirror")


def test_nan_rejected():
    S = wave()
    S[10] = np.nan
    with pytest.raises(ValueError, match="NaN or Inf"):
        mirror(S, mod="mirror")


# --------------------------------------------------------------------------- #
# 5. 契约面
# --------------------------------------------------------------------------- #
def test_public_surface():
    import Modal_Decomposition.Utils as U

    for name in ("mirror", "MIRROR_MODS"):
        assert getattr(U, name) is getattr(_Mirror, name)
    assert _Mirror.__all__ == ["mirror", "MIRROR_MODS", "MIRROR_MODS_ARGS"]
    assert [m.value for m in MIRROR_MODS] == ["mirror", "extrema", "polynomial", "parallel", "BLCSE"]


def test_no_input_mutation():
    S = wave()
    T_in = np.linspace(0, 1, S.size)
    s0, t0 = S.copy(), T_in.copy()
    mirror(S, mod="mirror", nbsym=3)
    for mod in EXTRAPOLATING:
        mirror(S, mod=mod, nbsym=2, T=T_in)
    assert np.array_equal(S, s0) and np.array_equal(T_in, t0)


def test_deterministic():
    S = wave()
    for mod in EXTRAPOLATING:
        a = mirror(S, mod=mod, nbsym=2)
        b = mirror(S, mod=mod, nbsym=2)
        assert all(np.array_equal(x, y) for x, y in zip(a, b))


def test_memmap_input(tmp_path):
    S = wave(64)
    path = tmp_path / "s.dat"
    mm = np.memmap(path, dtype=np.float64, mode="w+", shape=S.shape)
    mm[:] = S
    mm.flush()
    try:
        out = mirror(mm, mod="mirror", nbsym=2)
        assert np.array_equal(out, np.pad(S, (2, 2), mode="reflect"))
        T, A, _ = mirror(mm, mod="extrema", nbsym=1)
        assert T.size == find_peaks(S, mod="both")[0].size + 2
    finally:
        mm._mmap.close()
        del mm


# --------------------------------------------------------------------------- #
# 导出
# --------------------------------------------------------------------------- #
def build_table() -> list[dict]:
    S = wave()
    core = find_peaks(S, mod="both")[0]
    rows = []
    for mod in ("mirror",) + EXTRAPOLATING:
        for nbsym in (1, 2, 3, 4, 5):
            try:
                out = mirror(S, mod=mod, nbsym=nbsym)
                if mod == "mirror":
                    rows.append({"mod": mod, "nbsym": nbsym, "out_len": int(out.size),
                                 "L_core": None, "expected_len": int(S.size + 2 * nbsym),
                                 "strictly_increasing_T": None, "error": ""})
                else:
                    T, A, _ = out
                    exp = core.size + (4 if mod == "BLCSE" else 2 * nbsym)
                    rows.append({"mod": mod, "nbsym": nbsym, "out_len": int(T.size),
                                 "L_core": int(core.size), "expected_len": int(exp),
                                 "strictly_increasing_T": int(np.all(np.diff(T) > 0)),
                                 "error": ""})
            except Exception as exc:            # noqa: BLE001
                rows.append({"mod": mod, "nbsym": nbsym, "out_len": None,
                             "L_core": int(core.size), "expected_len": None,
                             "strictly_increasing_T": None,
                             "error": f"{type(exc).__name__}: {exc}"[:60]})
    return rows


def export() -> None:
    rows = build_table()
    B.write_csv(B.TABLE / "utils" / "mirror" / "mirror_modes.csv", rows)

    B.plt_style()
    import matplotlib.pyplot as plt

    S = wave()
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 6.4))
    n = np.arange(S.size)

    ax = axes[0, 0]
    ax.plot(n, S, "k-", lw=1.0, label="signal")
    ax.plot(np.arange(-3, 0), mirror(S, mod="mirror", nbsym=3)[:3], "ro", ms=3)
    ax.plot(np.arange(S.size, S.size + 3),
            mirror(S, mod="mirror", nbsym=3)[-3:], "ro", ms=3)
    ax.set_title("mod='mirror' (sample reflection)")
    ax.legend(fontsize=7)

    ax = axes[0, 1]
    for mod in EXTRAPOLATING:
        T, A, m = mirror(S, mod=mod, nbsym=3)
        ax.plot(T, A, "o-", ms=2.5, lw=0.8, label=mod)
    ax.set_title("extended extrema tables (nbsym=3)")
    ax.legend(fontsize=7)

    ax = axes[1, 0]
    T, A, m = mirror(S, mod="BLCSE", nbsym=2)
    ax.plot(T[m], A[m], "r^", ms=3.5, label="max")
    ax.plot(T[~m], A[~m], "bv", ms=3.5, label="min")
    ax.set_title("BLCSE (one pair per end)")
    ax.legend(fontsize=7)

    ax = axes[1, 1]
    for mod in ("extrema", "polynomial", "parallel"):
        sub = [r for r in rows if r["mod"] == mod and r["out_len"]]
        ax.plot([r["nbsym"] for r in sub], [r["out_len"] for r in sub], "o-", ms=3, label=mod)
    sub = [r for r in rows if r["mod"] == "BLCSE" and r["out_len"]]
    ax.plot([r["nbsym"] for r in sub], [r["out_len"] for r in sub], "s--", ms=3, label="BLCSE")
    ax.set_xlabel("nbsym")
    ax.set_ylabel("extended table length")
    ax.set_title("length identity: L + 2*nbsym  (BLCSE: L + 4)")
    ax.legend(fontsize=7)

    fig.suptitle("Utils.Mirror — five endpoint-extension geometries")
    fig.tight_layout()
    p = B.savefig(fig, B.FIG / "utils" / "mirror" / "mirror_modes.png")
    plt.close(fig)
    print(f"wrote {B.TABLE / 'utils' / 'mirror' / 'mirror_modes.csv'}")
    print(f"wrote {p}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    B.add_export_flag(ap)
    args = ap.parse_args()
    if args.export:
        export()
    else:
        B.print_table(build_table())
