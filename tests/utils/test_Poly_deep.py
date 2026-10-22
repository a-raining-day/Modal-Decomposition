"""``Utils.Poly`` 深度测试。

``fast_poly`` 很小（31 行），但它是包络/极值模块的共用数值核，因此本文件专攻
"数值上到底等价于谁、在什么条件下更稳"：

1. **精确性**：次数 ≤ ``poly_order`` 的多项式必须逐点精确到机器精度；
2. **等价性**：与 ``np.polyfit`` + ``np.polyval`` 在 ``order ∈ [0, 8]`` 上一致，
   并给出逐阶最大偏差（``table/utils/poly/poly_vs_polyfit.csv``）；
3. **病态场景**：自变量是 1.7e9 量级的时间戳时，``np.polyfit`` 出 RankWarning 且
   误差被放大，而 ``fast_poly`` 先中心化再缩放 —— 这是 docstring 的核心主张，必须验；
4. **欠定 / 退化**：``order > n_ref-1``（欠定，lstsq 给最小范数解）、单点、常值 x
   （span=0 → scale 兜底 1.0）、``x_new`` 空、``x_new`` 远超拟合区间（外推）；
5. **契约**：2-D/NaN/非法 safe 的拒绝、dtype、形状、确定性、不改输入。

导出::

    .\\.venv\\Scripts\\python.exe tests/utils/test_Poly_deep.py --export

规模轴在 ``tests/size_axis/run_scale.py --tool poly``（这里关心 ``x_new`` 很大时
Vandermonde 的 ``(order+1)×`` 放大）。
"""

from __future__ import annotations

import argparse
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

from Modal_Decomposition.Utils import fast_poly  # noqa: E402


def ref_polyfit(x, y, order, x_new):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.polyval(np.polyfit(x, y, order), x_new)


def ref_well_conditioned(x, y, order, x_new):
    """独立参考实现：手写"中心化 + 缩放 + lstsq"（与 fast_poly 同数学、不同代码）。"""
    center = x.mean()
    span = x.max() - x.min()
    scale = span if span > 0 else 1.0
    basis = np.vander((x - center) / scale, order + 1, increasing=True)
    coef = np.linalg.lstsq(basis, y, rcond=None)[0]
    target = np.vander((x_new - center) / scale, order + 1, increasing=True)
    return target @ coef


# --------------------------------------------------------------------------- #
# 1. 精确性
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("order", [0, 1, 2, 3, 5, 8])
def test_exact_on_matching_polynomial(order):
    rng = np.random.default_rng(order)
    coef = rng.standard_normal(order + 1)
    x = np.sort(rng.uniform(-3, 3, 40))
    y = np.polynomial.polynomial.polyval(x, coef)
    out = fast_poly(x, y, order, x)
    scale = np.max(np.abs(y)) or 1.0
    assert np.max(np.abs(out - y)) < 1e-9 * scale, f"order={order}"


def test_order_zero_is_mean():
    x = np.linspace(0.0, 1.0, 17)
    y = np.linspace(1.0, 5.0, 17)
    out = fast_poly(x, y, 0, np.array([0.0, 0.5, 1.0]))
    assert np.allclose(out, y.mean(), atol=1e-12)


# --------------------------------------------------------------------------- #
# 2. 与 polyfit 等价
# --------------------------------------------------------------------------- #
def build_vs_polyfit() -> list[dict]:
    rng = np.random.default_rng(42)
    rows = []
    cases = {
        "smooth": (np.linspace(-2, 2, 50),
                   lambda x: np.sin(x) + 0.05 * rng.standard_normal(x.size)),
        "timestamp": (1.7e9 + np.arange(60, dtype=float) * 0.25,
                      lambda x: np.sin((x - x.mean()) * 0.1)),
        "noisy_linear": (np.linspace(0, 10, 40),
                         lambda x: 2.0 * x + 1.0 + rng.standard_normal(x.size)),
    }
    for name, (x, f) in cases.items():
        y = f(x)
        x_new = np.linspace(x.min() + 0.1 * (x.max() - x.min()),
                            x.max() - 0.1 * (x.max() - x.min()), 31)
        for order in range(0, 9):
            got = fast_poly(x, y, order, x_new)
            ref = ref_polyfit(x, y, order, x_new)
            wc = ref_well_conditioned(x, y, order, x_new)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                np.polyval(np.polyfit(x, y, order), x_new)      # 直接调，才能看到 RankWarning
            warned = any(issubclass(w.category, np.exceptions.RankWarning) for w in caught)
            denom = max(np.max(np.abs(wc)), 1e-300)
            rows.append({
                "case": name, "order": order,
                "max_abs_vs_wc": float(np.max(np.abs(got - wc))),
                "max_rel_vs_wc": float(np.max(np.abs(got - wc)) / denom),
                "max_abs_vs_polyfit": float(np.max(np.abs(got - ref))),
                "polyfit_rank_warning": int(warned),
                "finite": int(np.all(np.isfinite(got))),
                "verdict": "PASS" if np.allclose(got, wc, rtol=1e-9, atol=1e-9) else "FAIL",
            })
    return rows


def test_matches_well_conditioned_reference():
    """与"手写中心化 lstsq"逐点一致（这才是 fast_poly 的正确对照）。"""
    rows = build_vs_polyfit()
    bad = [r for r in rows if r["verdict"] != "PASS"]
    assert not bad, bad


def test_polyfit_deviates_on_ill_conditioned_cases():
    """``np.polyfit`` 在 1.7e9 量级自变量上会偏离良态解（并有 RankWarning）；
    ``fast_poly`` 不偏离 —— 这正是 docstring 里"归一化不改变最小二乘解"的意思。"""
    rows = [r for r in build_vs_polyfit() if r["case"] == "timestamp"]
    assert any(r["polyfit_rank_warning"] for r in rows), "该用例应至少一阶触发 RankWarning"
    worst = max(rows, key=lambda r: r["max_abs_vs_polyfit"])
    assert worst["max_abs_vs_polyfit"] > 1e-3, worst
    assert all(r["max_abs_vs_wc"] < 1e-9 for r in rows), rows


# --------------------------------------------------------------------------- #
# 3. 病态：时间戳量级自变量
# --------------------------------------------------------------------------- #
def test_timestamp_exact_cubic_polyfit_blows_up():
    """最有说服力的一例：**精确三次**多项式 + 1.7e9 量级自变量。

    ``np.polyfit`` 给 RankWarning 且最大误差 ~2.5e2；``fast_poly`` 误差 ~1e-13。
    （误差对 ``y`` 而不是对另一份系数，所以不掺近似误差。）
    """
    x = 1.7e9 + np.arange(40, dtype=float) * 0.5
    u = x - 1.7e9
    y = 1.0 + 2.0 * u + 0.5 * u ** 2 + 0.1 * u ** 3
    order = 3

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ref = np.polyval(np.polyfit(x, y, order), x)
    assert any(issubclass(w.category, np.exceptions.RankWarning) for w in caught)

    got = fast_poly(x, y, order, x)
    err_fast = float(np.max(np.abs(got - y)))
    err_ref = float(np.max(np.abs(ref - y)))
    assert err_fast < 1e-8, f"fast_poly 应精确，实测 {err_fast:.3e}"
    assert err_ref > 1.0, f"np.polyfit 在此例应显著失准，实测 {err_ref:.3e}"


def test_extrapolation_beyond_range():
    x = np.linspace(0.0, 1.0, 30)
    y = 1.0 + 2.0 * x + 3.0 * x ** 2
    far = np.array([-5.0, 0.5, 6.0])
    out = fast_poly(x, y, 2, far)
    assert np.allclose(out, 1.0 + 2.0 * far + 3.0 * far ** 2, atol=1e-8)


# --------------------------------------------------------------------------- #
# 4. 欠定 / 退化
# --------------------------------------------------------------------------- #
def test_underdetermined_fits_exactly():
    """``order > n_ref-1`` 时 lstsq 给最小范数解，但在参考点上仍应精确穿过。"""
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([1.0, 4.0, 9.0])
    got = fast_poly(x, y, 5, x)
    assert np.allclose(got, y, atol=1e-8)


def test_single_point_reference():
    got = fast_poly(np.array([2.0]), np.array([7.0]), 3, np.array([2.0]))
    assert np.allclose(got, 7.0, atol=1e-8)


def test_constant_x_reference():
    """span == 0 → scale 兜底 1.0（否则除零）。结果应是 ``y`` 的均值（秩 1）。"""
    x = np.full(9, 3.0)
    y = np.arange(9.0)
    got = fast_poly(x, y, 2, np.array([3.0]))
    assert np.all(np.isfinite(got))
    assert np.allclose(got, y.mean(), atol=1e-8)


def test_empty_x_new():
    x = np.linspace(0.0, 1.0, 10)
    out = fast_poly(x, x, 1, np.array([]))
    assert out.shape == (0,)


def test_x_new_2d_rejected():
    x = np.linspace(0.0, 1.0, 10)
    with pytest.raises(ValueError):
        fast_poly(x, x, 1, np.zeros((2, 3)))


def test_nan_rejected_by_default():
    x = np.linspace(0.0, 1.0, 10)
    y = x.copy()
    y[3] = np.nan
    with pytest.raises(ValueError, match="NaN or Inf"):
        fast_poly(x, y, 1, x)


def test_length_mismatch_behaviour():
    """``x_ref`` 与 ``y_ref`` 长度不一致时：``np.linalg.lstsq`` 抛 LinAlgError/LinAlgError 系。"""
    x = np.linspace(0.0, 1.0, 10)
    y = np.linspace(0.0, 1.0, 9)
    with pytest.raises(Exception):
        fast_poly(x, y, 1, x)


def test_int_inputs_accepted():
    x = np.arange(5)
    y = np.array([0, 2, 4, 6, 8])
    out = fast_poly(x, y, 1, x)
    assert out.dtype == np.float64
    assert np.allclose(out, y, atol=1e-8)


def test_safe_safe_with_list_fails():
    """特征化：``safe="safe"`` 不做 array 化，list 在 ``.mean()`` 处 AttributeError。"""
    with pytest.raises(AttributeError):
        fast_poly([0.0, 1.0, 2.0], [0.0, 1.0, 2.0], 1, [0.0], safe="safe")


def test_safe_suspicious_accepts_2d_but_breaks():
    """特征化：``safe="suspicious"`` 跳过维度检查 → 2-D 输入会在 vander 之后才炸。"""
    with pytest.raises(Exception):
        fast_poly(np.zeros((2, 3)), np.zeros(3), 1, np.zeros(2), safe="suspicious")


# --------------------------------------------------------------------------- #
# 5. 契约面
# --------------------------------------------------------------------------- #
def test_output_dtype_shape():
    x = np.linspace(0.0, 1.0, 7)
    x_new = np.linspace(0.0, 1.0, 13)
    out = fast_poly(x, x, 2, x_new)
    assert out.shape == x_new.shape and out.dtype == np.float64


def test_no_input_mutation():
    x = np.linspace(0.0, 1.0, 20)
    y = x ** 3
    xn = np.linspace(-1.0, 2.0, 11)
    xs, ys, xns = x.copy(), y.copy(), xn.copy()
    fast_poly(x, y, 3, xn)
    assert np.array_equal(x, xs) and np.array_equal(y, ys) and np.array_equal(xn, xns)


def test_deterministic_bitwise():
    rng = np.random.default_rng(3)
    x = np.linspace(0.0, 1.0, 30)
    y = np.sin(5 * x) + rng.standard_normal(30)
    xn = np.linspace(0.0, 1.0, 25)
    assert np.array_equal(fast_poly(x, y, 4, xn), fast_poly(x, y, 4, xn))


def test_public_surface():
    import Modal_Decomposition.Utils as U
    import Modal_Decomposition.Utils.Poly as P

    assert U.fast_poly is fast_poly
    assert P.__all__ == ["fast_poly"]


def test_linear_time_in_x_new():
    """复杂度应随 ``x_new`` 线性、随 order 线性（不是 n_ref 的平方）。"""
    rng = np.random.default_rng(0)
    x = np.linspace(0.0, 1.0, 64)
    y = np.sin(3 * x)
    t = {}
    for n in (10_000, 1_000_000):
        xn = np.linspace(0.0, 1.0, n)
        r = B.timeit(lambda: fast_poly(x, y, 3, xn), repeat=3, warmup=1)
        t[n] = r["best_ms"]
    ratio = t[1_000_000] / t[10_000]
    # 100× 数据；固定开销让比值明显低于 100，但 O(n²) 会给出 ~10000×
    assert 15 < ratio < 400, (t, ratio)


# --------------------------------------------------------------------------- #
# 导出
# --------------------------------------------------------------------------- #
def export() -> None:
    rows = build_vs_polyfit()
    B.write_csv(B.TABLE / "utils" / "poly" / "poly_vs_polyfit.csv", rows)

    B.plt_style()
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    for name in sorted({r["case"] for r in rows}):
        sub = [r for r in rows if r["case"] == name]
        ax.semilogy([r["order"] for r in sub],
                    [max(r["max_abs_vs_polyfit"], 1e-17) for r in sub], "o-", label=name)
    ax.set_xlabel("polynomial order")
    ax.set_ylabel("max abs diff vs np.polyfit")
    ax.set_title("fast_poly vs np.polyfit (smaller = polyfit agrees)")
    ax.legend(fontsize=7)

    ax = axes[1]
    x = np.linspace(-2, 2, 50)
    y = np.sin(x)
    for order in (0, 1, 3, 8):
        ax.plot(x, fast_poly(x, y, order, x), label=f"order {order}")
    ax.plot(x, y, "k--", lw=1.0, label="sin(x)")
    ax.set_title("fit quality vs order (n=50)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    p = B.savefig(fig, B.FIG / "utils" / "poly" / "poly_vs_polyfit.png")
    plt.close(fig)
    print(f"wrote {B.TABLE / 'utils' / 'poly' / 'poly_vs_polyfit.csv'}")
    print(f"wrote {p}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    B.add_export_flag(ap)
    args = ap.parse_args()
    if args.export:
        export()
    else:
        B.print_table(build_vs_polyfit())
