"""``Utils.Envelope`` 深度测试：四种包络策略的定量对比 + 参数契约。

docstring 里那张表（四种模式的 RMSE / 相关系数 / 耗时）是选择默认值的依据，
本文件把它**重算一遍**（``table/utils/envelope/envelope_modes.csv``），并在同一批
信号上验证"什么时候 Lowpass/IQ 比 Hilbert 更好"这个结论。

另有一组参数契约测试：``fs`` / ``fc`` / ``cf`` 的缺失组合、``kwargs`` 透传泄漏、
Peak 模式的样条要求、2-D / NaN / 空输入在各模式下的不同反应。

导出::

    .\\.venv\\Scripts\\python.exe tests/utils/test_Envelope_deep.py --export

规模轴在 ``tests/size_axis/run_scale.py --tool envelope``。
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

from Modal_Decomposition.Utils import ENVELOPE_MODS, envelope  # noqa: E402

_Env = importlib.import_module("Modal_Decomposition.Utils.Envelope")

FS, FC, CF = 10_000.0, 1_000.0, 200.0
N = 2 ** 14


def am_case(name: str) -> tuple[np.ndarray, np.ndarray]:
    """返回 (信号, 真实包络)。"""
    t = np.arange(N) / FS
    if name == "clean":
        amp = 1.0 + 0.4 * np.cos(2 * np.pi * 50 * t)
    elif name == "deep":
        amp = 0.5 + 0.5 * np.cos(2 * np.pi * 10 * t)
    elif name == "two_tone":
        amp = 1.0 + 0.3 * np.cos(2 * np.pi * 30 * t) + 0.2 * np.cos(2 * np.pi * 70 * t)
    else:  # pragma: no cover
        raise ValueError(name)
    s = amp * np.cos(2 * np.pi * FC * t)
    return s, amp


def noisy_case(sigma: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    s, amp = am_case("clean")
    rng = np.random.default_rng(0)
    return s + sigma * rng.standard_normal(s.size), amp


def metrics(env: np.ndarray, amp: np.ndarray, trim: int = 200) -> tuple[float, float]:
    sl = slice(trim, -trim)
    e, a = env[sl], amp[sl]
    rmse = float(np.sqrt(np.mean((e - a) ** 2)))
    corr = float(np.corrcoef(e, a)[0, 1])
    return rmse, corr


METHOD_ARGS = {
    "Hilbert": {},
    "Lowpass": {"fs": FS, "cf": CF},
    "IQ": {"fs": FS, "fc": FC, "cf": CF},
    "PeakInterpolation": {},
}


def run(method: str, s: np.ndarray, **extra) -> np.ndarray:
    kw = dict(METHOD_ARGS[method])
    kw.update(extra)
    return envelope(s, method=method, **kw)


# --------------------------------------------------------------------------- #
# 1. 四种模式的定量对比
# --------------------------------------------------------------------------- #
def build_mode_table() -> list[dict]:
    rows = []
    cases = {"clean": am_case("clean"), "deep": am_case("deep"),
             "two_tone": am_case("two_tone"), "noisy": noisy_case()}
    for cname, (s, amp) in cases.items():
        for method in METHOD_ARGS:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                env = run(method, s)
            rmse, corr = metrics(env, amp)
            r = B.timeit(lambda: run(method, s), repeat=3, warmup=1)
            rows.append({
                "case": cname, "method": method,
                "rmse": round(rmse, 6), "corr": round(corr, 6),
                "best_ms": round(r["best_ms"], 3),
            })
    return rows


def test_clean_am_hilbert_and_peak_are_exact():
    rows = [r for r in build_mode_table() if r["case"] == "clean"]
    by = {r["method"]: r for r in rows}
    assert by["Hilbert"]["rmse"] < 1e-3, by["Hilbert"]
    assert by["Hilbert"]["corr"] > 0.9999
    # Peak 在干净 AM 上也应很好（样条穿过峰值点）
    assert by["PeakInterpolation"]["rmse"] < 0.1, by["PeakInterpolation"]


def test_lowpass_and_iq_need_right_parameters():
    """干净 AM 上的排序：IQ≈Peak（精确）> Hilbert（5e-4）> Lowpass（1.7e-2）。

    Lowpass 最差的物理原因：全波整流 + 4 阶 Butterworth 的残余纹波落在包络里。
    """
    rows = [r for r in build_mode_table() if r["case"] == "clean"]
    by = {r["method"]: r for r in rows}
    assert by["IQ"]["corr"] > 0.999 and by["PeakInterpolation"]["corr"] > 0.999
    assert by["Hilbert"]["rmse"] < 1e-3
    assert by["Lowpass"]["corr"] > 0.99
    assert by["Lowpass"]["rmse"] > by["Hilbert"]["rmse"], "Lowpass 在本例不如 Hilbert"


def test_noisy_case_and_ranking():
    """噪声下排序反转：IQ(0.014) < Lowpass(0.024) < Hilbert(0.050) < Peak(0.075)。

    docstring 只断言了 "Lowpass 优于 Hilbert（0.026 vs 0.051）"，这里顺带记录
    真正最好的是 IQ（窄带滤波抑制噪声），以及 Peak 在噪声下最差（伪峰）。
    """
    rows = [r for r in build_mode_table() if r["case"] == "noisy"]
    by = {r["method"]: r for r in rows}
    assert all(r["corr"] > 0.95 for r in rows), by
    assert by["Lowpass"]["rmse"] < by["Hilbert"]["rmse"], "docstring 的主张应成立"
    assert by["IQ"]["rmse"] < by["Lowpass"]["rmse"], "窄带 IQ 在噪声下最好"
    assert by["PeakInterpolation"]["rmse"] > by["Hilbert"]["rmse"], "伪峰使 Peak 最差"


# --------------------------------------------------------------------------- #
# 2. 参数契约
# --------------------------------------------------------------------------- #
def test_lowpass_requires_parameters():
    s, _ = am_case("clean")
    with pytest.raises(ValueError, match="Either fc, fs or cf"):
        envelope(s, method="Lowpass")
    with pytest.raises(ValueError, match="Either fc, fs or cf"):
        envelope(s, method="IQ")


def test_partial_parameters_fail_later_with_type_error():
    """特征化（缺陷）：校验只在 **三者全 None** 时触发。

    只给 ``cf`` 不给 ``fs`` 时，``_lowpass`` 里 ``0.5 * fs`` → TypeError（不是
    一开始就给出清晰的 ValueError）。
    """
    s, _ = am_case("clean")
    with pytest.raises(TypeError):
        envelope(s, method="Lowpass", cf=CF)
    with pytest.raises(TypeError):
        envelope(s, method="IQ", fs=FS, fc=FC)      # 缺 cf


def test_enum_skips_parameter_validation():
    """特征化：用枚举传 method 时，``method in ["Lowpass","IQ"]`` 恒 False ⇒ 校验被跳过。"""
    s, _ = am_case("clean")
    with pytest.raises(TypeError):
        envelope(s, method=ENVELOPE_MODS.Lowpass)


def test_unknown_method():
    s, _ = am_case("clean")
    with pytest.raises(ValueError, match="Unknown envelope method"):
        envelope(s, method="nope")
    with pytest.raises(ValueError, match="Unknown envelope method"):
        envelope(s, method="hilbert")               # 大小写敏感
    with pytest.raises(ValueError, match="Unknown envelope method"):
        envelope(s, method="Peak")                  # 枚举名是 Peak，但字符串值必须是 PeakInterpolation
    assert ENVELOPE_MODS.Peak.value == "PeakInterpolation"


def test_enum_roundtrip():
    assert ENVELOPE_MODS("PeakInterpolation") is ENVELOPE_MODS.Peak
    assert ENVELOPE_MODS.Peak.value == "PeakInterpolation"
    s, _ = am_case("clean")
    a = envelope(s, method=ENVELOPE_MODS.Hilbert)
    b = envelope(s, method="Hilbert")
    assert np.array_equal(a, b)


def test_peak_mod_kwarg_no_longer_leaks():
    """回归（本轮修好）：`peak_mod` / `hilbert_mod` 会先 `pop` 出 kwargs。

    原缺陷：kwargs 同时喂给 `find_peaks` 与 `CubicSpline`，`peak_mod` 让
    `CubicSpline.__init__()` 直接 TypeError。
    """
    s, _ = am_case("clean")
    base = envelope(s, method="PeakInterpolation")
    for be in ("numpy", "scipy"):
        got = envelope(s, method="PeakInterpolation", peak_mod=be)
        assert got.shape == s.shape
        assert np.allclose(got, base, atol=1e-12), be


def test_custom_spline_factory():
    s, _ = am_case("clean")
    calls = {}

    def factory(x, y, **kw):
        from scipy.interpolate import CubicSpline
        calls["x"] = np.asarray(x).size
        return CubicSpline(x, y)

    env = envelope(s, method="PeakInterpolation", spline=factory)
    assert calls["x"] > 0 and env.shape == s.shape


def test_peak_needs_at_least_two_peaks():
    with pytest.raises(ValueError):
        envelope(np.arange(1000.0), method="PeakInterpolation")
    with pytest.raises(ValueError):
        envelope(np.full(500, 3.0), method="PeakInterpolation")


def test_hilbert_backend_switch_works():
    """回归（本轮修好）：`hilbert_mod` 不再被透传进 `hilbert(**kwargs)`。

    三个后端都应可用，且互相在 1e-9 内一致（FFT/FHT 与 scipy 的解析信号差 ~1e-15）。
    """
    s, amp = am_case("clean")
    ref = envelope(s, method="Hilbert")                      # 默认 scipy
    for mod in ("scipy", "FFT", "FHT"):
        got = envelope(s, method="Hilbert", hilbert_mod=mod)
        assert got.shape == s.shape
        assert np.allclose(got, ref, atol=1e-9), mod
        rmse, _ = metrics(got, amp)
        assert rmse < 1e-3, (mod, rmse)
    # 未知后端名仍应报错（ValueError 来自 HILBERT_MODS 校验）
    with pytest.raises(ValueError):
        envelope(s, method="Hilbert", hilbert_mod="nope")


def test_list_input_rejected():
    with pytest.raises(TypeError):
        envelope([1.0, 2.0, 3.0], method="Hilbert")


def test_memmap_input_rejected(tmp_path):
    path = tmp_path / "m.dat"
    mm = np.memmap(path, dtype=np.float64, mode="w+", shape=(64,))
    mm[:] = np.sin(np.linspace(0, 6.28, 64))
    mm.flush()
    try:
        with pytest.raises(TypeError):
            envelope(mm, method="Hilbert")
    finally:
        mm._mmap.close()
        del mm


def test_2d_accepted_because_backend_is_scipy():
    """特征化：默认（scipy）后端接受 2-D 并沿末轴做解析信号。

    库内 ``hilbert(..., "FFT")`` 会拒绝 2-D，但如前所述 ``hilbert_mod`` 传不进去，
    所以 Envelope 的 2-D 行为恒等于 scipy 后端的行为。
    """
    s2 = np.sin(np.linspace(0, 6.28, 64))[None, :].repeat(3, axis=0)
    env = envelope(s2, method="Hilbert")
    assert env.shape == (3, 64)
    assert np.allclose(env, np.abs(env))


def test_nan_policy_depends_on_mode():
    """特征化：Hilbert 模式不校验（NaN 透传），Peak 模式走 find_peaks 的默认校验 → 抛错。"""
    s = np.sin(np.linspace(0, 20 * np.pi, 400))
    s[100] = np.nan
    env = envelope(s, method="Hilbert")
    assert np.isnan(env[100])
    with pytest.raises(ValueError, match="NaN or Inf"):
        envelope(s, method="PeakInterpolation")


def test_empty_input():
    """两个模式都拒绝空输入，但错误来源不同：scipy 说 N must be positive，
    Peak 模式在 find_peaks/CubicSpline 上失败。"""
    with pytest.raises(ValueError, match="N must be positive"):
        envelope(np.array([]), method="Hilbert")
    with pytest.raises(ValueError):
        envelope(np.array([]), method="PeakInterpolation")


def test_output_dtype_shape():
    s, _ = am_case("clean")
    for method in METHOD_ARGS:
        env = run(method, s)
        assert env.shape == s.shape
        assert env.dtype == np.float64


def test_no_input_mutation():
    s, _ = am_case("clean")
    before = s.copy()
    for method in METHOD_ARGS:
        run(method, s)
    assert np.array_equal(s, before)


def test_deterministic():
    s, _ = am_case("clean")
    for method in METHOD_ARGS:
        assert np.array_equal(run(method, s), run(method, s))


# --------------------------------------------------------------------------- #
# 导出
# --------------------------------------------------------------------------- #
def export() -> None:
    rows = build_mode_table()
    B.write_csv(B.TABLE / "utils" / "envelope" / "envelope_modes.csv", rows)

    B.plt_style()
    import matplotlib.pyplot as plt

    s, amp = am_case("clean")
    t = np.arange(0, 0.02, 1 / FS)
    sl = slice(0, t.size)

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))
    ax = axes[0]
    ax.plot(t, s[sl], lw=0.5, color="gray", label="AM signal")
    for method in METHOD_ARGS:
        ax.plot(t, run(method, s)[sl], lw=1.1, label=method)
    ax.plot(t, amp[sl], "k--", lw=0.9, label="truth")
    ax.set_title("clean AM: 4 envelope modes")
    ax.set_xlabel("time (s)")
    ax.legend(fontsize=6)

    ax = axes[1]
    names = list(METHOD_ARGS)
    width = 0.2
    for k, cname in enumerate(["clean", "deep", "two_tone", "noisy"]):
        vals = [next(r["rmse"] for r in rows if r["case"] == cname and r["method"] == m)
                for m in names]
        ax.bar(np.arange(len(names)) + (k - 1.5) * width, vals, width, label=cname)
    ax.set_xticks(range(len(names)), names, rotation=20, fontsize=7)
    ax.set_yscale("log")
    ax.set_ylabel("RMSE vs true envelope")
    ax.set_title("accuracy by case")
    ax.legend(fontsize=6)

    ax = axes[2]
    s2, _ = noisy_case(0.2)
    for method in METHOD_ARGS:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ax.plot(run(method, s2)[sl], lw=1.0, label=method)
    ax.set_title("noisy AM (sigma=0.2)")
    ax.legend(fontsize=6)

    fig.tight_layout()
    p = B.savefig(fig, B.FIG / "utils" / "envelope" / "envelope_modes.png")
    plt.close(fig)
    print(f"wrote {B.TABLE / 'utils' / 'envelope' / 'envelope_modes.csv'}")
    print(f"wrote {p}")
    B.print_table(rows, ["case", "method", "rmse", "corr", "best_ms"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    B.add_export_flag(ap)
    args = ap.parse_args()
    if args.export:
        export()
    else:
        B.print_table(build_mode_table())
