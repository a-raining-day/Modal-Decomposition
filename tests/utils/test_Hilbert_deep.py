"""``Utils.Hilbert`` 深度测试：三后端 + FHT 原语 + 相位约定。

``hilbert`` 是**唯一不做输入校验**的工具（直接进 numpy FFT），所以"三个后端的行为
差异"比数值精度更值得测。本文件把它固化成矩阵：

========================  ==========  ==========  ==========
输入/参数                 FFT         FHT         scipy
========================  ==========  ==========  ==========
1-D 任意长度              精确        精确        基线
2-D                      ValueError  ValueError   **接受**（沿末轴）
空输入                    返回空      **ValueError**（N must be positive）
未知 kwarg               TypeError   TypeError   **静默忽略**
``matlab_phase``         生效        生效        **静默忽略**
========================  ==========  ==========  ==========

导出::

    .\\.venv\\Scripts\\python.exe tests/utils/test_Hilbert_deep.py --export

规模轴在 ``tests/size_axis/run_scale.py --tool hilbert``。
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

from Modal_Decomposition.Utils import (  # noqa: E402
    HILBERT_MODS,
    fht_forward,
    fht_inverse,
    hilbert,
    hilbert_transform,
)

_fht = importlib.import_module("Modal_Decomposition.Utils._Hilbert._fht")

IMPLEMENTED = ("FFT", "FHT", "scipy")
UNIMPLEMENTED = [m.value for m in HILBERT_MODS if m.value not in IMPLEMENTED]

SIZES = (1, 2, 3, 4, 7, 8, 15, 16, 255, 256, 511, 512, 513, 1000, 1024, 4096, 10000)


def x_of(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal(n)


# --------------------------------------------------------------------------- #
# 1. 分发与未实现模式
# --------------------------------------------------------------------------- #
def test_implemented_mods():
    x = x_of(64)
    for mod in IMPLEMENTED:
        z = hilbert(x, mod=mod)
        assert z.shape == x.shape and z.dtype == np.complex128


@pytest.mark.parametrize("mod", UNIMPLEMENTED)
def test_unimplemented_mods_raise(mod):
    with pytest.raises(ValueError, match="Unknown Hilbert"):
        hilbert(x_of(64), mod=mod)


def test_unknown_mod_raises():
    with pytest.raises(ValueError):
        hilbert(x_of(64), mod="nope")


def test_enum_and_string_equivalent():
    x = x_of(32)
    a = hilbert(x, mod=HILBERT_MODS.FFT)
    b = hilbert(x, mod="FFT")
    assert np.array_equal(a, b)


# --------------------------------------------------------------------------- #
# 2. 与 scipy 基线的一致性
# --------------------------------------------------------------------------- #
def build_agreement() -> list[dict]:
    rows = []
    for n in SIZES:
        x = x_of(n, seed=n)
        ref = hilbert(x, mod="scipy")
        scale = max(float(np.max(np.abs(x))), 1e-300)
        for mod in ("FFT", "FHT"):
            z = hilbert(x, mod=mod)
            err = np.max(np.abs(z - ref)) if n else 0.0
            rows.append({
                "n": n, "backend": mod, "pow2": int(n > 0 and (n & (n - 1)) == 0),
                "max_abs_err": float(err),
                "rel_to_signal": float(err / scale),
                "verdict": "PASS" if err <= 1e-9 * scale else "FAIL",
            })
    return rows


def test_agrees_with_scipy():
    bad = [r for r in build_agreement() if r["verdict"] != "PASS"]
    assert not bad, bad


def test_analytic_signal_properties():
    x = x_of(129)
    for mod in ("FFT", "FHT"):
        z = hilbert(x, mod=mod)
        assert np.array_equal(z.real, x), "实部必须逐位等于输入"
        assert np.max(np.abs(hilbert(x, mod=mod).imag)) > 0


def test_am_envelope_accuracy():
    """``|z|`` 在干净 AM 上应精确复现调制包络（这是 Envelope 默认模式的依据）。"""
    fs, fc, fm = 10_000.0, 1_000.0, 50.0
    n = 2 ** 14
    t = np.arange(n) / fs
    amp = 1.0 + 0.4 * np.cos(2 * np.pi * fm * t)
    s = amp * np.cos(2 * np.pi * fc * t)
    for mod in ("FFT", "FHT", "scipy"):
        env = np.abs(hilbert(s, mod=mod))
        edge = slice(200, -200)
        rmse = float(np.sqrt(np.mean((env[edge] - amp[edge]) ** 2)))
        assert rmse < 1e-3, (mod, rmse)


# --------------------------------------------------------------------------- #
# 3. 相位约定
# --------------------------------------------------------------------------- #
def test_matlab_phase_false_is_conjugate():
    x = x_of(256, seed=5)
    for mod in ("FFT", "FHT"):
        a = hilbert(x, mod=mod, matlab_phase=True)
        b = hilbert(x, mod=mod, matlab_phase=False)
        assert np.allclose(b, np.conj(a), atol=1e-12), mod


def test_scipy_backend_ignores_matlab_phase():
    """特征化：scipy 后端把 kwargs 全吞掉，``matlab_phase`` 静默失效。"""
    x = x_of(128, seed=6)
    a = hilbert(x, mod="scipy", matlab_phase=True)
    b = hilbert(x, mod="scipy", matlab_phase=False)
    assert np.array_equal(a, b), "scipy 后端不认 matlab_phase"
    c = hilbert(x, mod="FFT", matlab_phase=False)
    assert not np.allclose(a, c)


# --------------------------------------------------------------------------- #
# 4. 输入契约差异（矩阵）
# --------------------------------------------------------------------------- #
def build_contract_matrix() -> list[dict]:
    cases = {
        "2d": (np.zeros((3, 32)), {}),
        "empty": (np.array([]), {}),
        "n1": (np.array([1.0]), {}),
        "nan": (np.array([0.0, np.nan]), {}),
        "unknown_kwarg": (np.zeros(16), {"foo": 1}),
        "int_dtype": (np.arange(8), {}),
        "list_input": ([1.0, 2.0, 3.0], {}),
    }
    rows = []
    for name, (arg, kw) in cases.items():
        for mod in IMPLEMENTED:
            try:
                z = hilbert(arg, mod=mod, **kw)
                rows.append({"case": name, "backend": mod, "outcome": "ok",
                             "shape": str(z.shape)})
            except Exception as exc:            # noqa: BLE001
                rows.append({"case": name, "backend": mod,
                             "outcome": type(exc).__name__,
                             "shape": str(exc)[:60]})
    return rows


def test_contract_matrix_documented():
    rows = build_contract_matrix()
    lookup = {(r["case"], r["backend"]): r["outcome"] for r in rows}
    # 1-D 无论长度都行
    for mod in IMPLEMENTED:
        assert lookup[("n1", mod)] == "ok"
        assert lookup[("nan", mod)] == "ok", "无校验：NaN 直接透传"
        assert lookup[("list_input", mod)] == "ok"
    # 2-D：只有 scipy 接受
    assert lookup[("2d", "FFT")] == "ValueError"
    assert lookup[("2d", "FHT")] == "ValueError"
    assert lookup[("2d", "scipy")] == "ok"
    # 空输入：只有 scipy 抛
    assert lookup[("empty", "FFT")] == "ok"
    assert lookup[("empty", "FHT")] == "ok"
    assert lookup[("empty", "scipy")] == "ValueError"
    # 未知 kwarg：只有 scipy 静默
    assert lookup[("unknown_kwarg", "FFT")] == "TypeError"
    assert lookup[("unknown_kwarg", "FHT")] == "TypeError"
    assert lookup[("unknown_kwarg", "scipy")] == "ok"


def test_complex_input_drops_imaginary_with_warning():
    """docstring 兑现：复数输入丢虚部并发 ``ComplexWarning``（不是抛错）。"""
    z = np.array([1.0 + 2j, 3.0 + 4j])
    with pytest.warns(np.exceptions.ComplexWarning):
        out = hilbert(z, mod="FFT")
    assert np.allclose(out.real, z.real), "虚部被丢弃，只保留实部"


def test_no_validation_of_length_or_finiteness():
    """Hilbert 不做 ``Check``：NaN 透传、无长度上限、无 dtype 归一（除 float64）。"""
    z = hilbert(np.array([0.0, np.nan, 1.0]), mod="FFT")
    assert np.isnan(z.imag[1]) or np.isnan(z.real[1])


# --------------------------------------------------------------------------- #
# 5. FHT 原语
# --------------------------------------------------------------------------- #
def test_fht_roundtrip_pow2():
    x = x_of(64, seed=9)
    h = fht_forward(x)
    back = fht_inverse(h)
    assert np.allclose(back, x, atol=1e-12)


def test_fht_zero_pads_non_pow2():
    x = x_of(1000, seed=10)
    h = fht_forward(x)
    assert h.shape == (1024,), "非 2 的幂会零填充到下一个 2 的幂"
    assert np.allclose(fht_inverse(h)[:1000], x, atol=1e-10)


def test_dht_definition():
    """DHT 定义直检：H[k] = Σ x[n]·cas(2πkn/N)（取小尺寸 O(N²) 对照）。"""
    x = np.array([1.0, 2.0, 0.5, -1.0])
    n = x.size
    k = np.arange(n)[:, None]
    nn = np.arange(n)[None, :]
    cas = np.cos(2 * np.pi * k * nn / n) + np.sin(2 * np.pi * k * nn / n)
    assert np.allclose(fht_forward(x), cas @ x, atol=1e-12)


def test_hilbert_transform_matches_pure_reference():
    x = x_of(1000, seed=11)
    native = hilbert_transform(x, matlab_phase=True)
    pure = _fht._hilbert_transform_pure(x, matlab_phase=True)
    assert np.max(np.abs(native - pure)) < 1e-12


def test_hilbert_transform_zeropad_artefact():
    """原语的零填充伪影：非 2 的幂长度上相对 scipy 的误差是 O(1)%。"""
    x = x_of(1000, seed=12)
    zp = hilbert_transform(x, matlab_phase=True)
    ref = hilbert(x, mod="scipy")
    rel = np.linalg.norm(zp - ref) / max(np.linalg.norm(ref), 1e-300)
    assert rel > 1e-4, f"应能看到显著伪影，实测 {rel:.3e}"
    assert rel < 0.1, f"误差不应荒谬，实测 {rel:.3e}"


def test_hilbert_transform_default_phase_is_sao():
    """原语默认 ``matlab_phase=False``（SAO 的 +90°），与 ``hilbert`` 默认相反。"""
    x = x_of(64, seed=13)
    a = hilbert_transform(x)                       # 默认 False
    b = hilbert_transform(x, matlab_phase=True)
    assert np.allclose(a, np.conj(b), atol=1e-12)
    assert np.allclose(hilbert(x, mod="FHT"), b, atol=1e-12)


# --------------------------------------------------------------------------- #
# 6. 契约面 / 确定性
# --------------------------------------------------------------------------- #
def test_dtype_and_shape():
    for n in (1, 7, 64, 65):
        z = hilbert(np.arange(n), mod="FFT")
        assert z.dtype == np.complex128 and z.shape == (n,)


def test_no_input_mutation():
    x = x_of(128, seed=14)
    before = x.copy()
    for mod in IMPLEMENTED:
        hilbert(x, mod=mod)
    assert np.array_equal(x, before)


def test_deterministic():
    x = x_of(200, seed=15)
    for mod in IMPLEMENTED:
        assert np.array_equal(hilbert(x, mod=mod), hilbert(x, mod=mod))


def test_public_surface():
    import Modal_Decomposition.Utils as U

    for name in ("hilbert", "fht_forward", "fht_inverse", "hilbert_transform",
                 "HILBERT_MODS"):
        assert hasattr(U, name)


# --------------------------------------------------------------------------- #
# 导出
# --------------------------------------------------------------------------- #
def export() -> None:
    rows = build_agreement()
    B.write_csv(B.TABLE / "utils" / "hilbert" / "backend_agreement.csv", rows)
    cm = build_contract_matrix()
    B.write_csv(B.TABLE / "utils" / "hilbert" / "contract_matrix.csv", cm)

    timings = []
    for n in (256, 1024, 4096, 16384, 65536, 262144):
        x = x_of(n, seed=n)
        for mod in IMPLEMENTED:
            r = B.timeit(lambda: hilbert(x, mod=mod), repeat=5, warmup=1)
            timings.append({"n": n, "backend": mod, "best_ms": round(r["best_ms"], 5),
                            "us_per_call": round(r["best_ms"] * 1e3, 2)})
    B.write_csv(B.TABLE / "utils" / "hilbert" / "backend_timing.csv", timings)

    zp = []
    for n in (100, 255, 256, 512, 513, 1000, 1024, 4000):
        x = x_of(n, seed=n)
        z = hilbert_transform(x, matlab_phase=True)
        ref = hilbert(x, mod="scipy")
        zp.append({"n": n, "pow2": int((n & (n - 1)) == 0),
                   "rel_l2_vs_scipy": float(np.linalg.norm(z - ref)
                                            / max(np.linalg.norm(ref), 1e-300))})
    B.write_csv(B.TABLE / "utils" / "hilbert" / "zeropad_artefact.csv", zp)

    B.plt_style()
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))
    ax = axes[0]
    ax.loglog([r["n"] for r in rows if r["backend"] == "FFT"],
              [max(r["max_abs_err"], 1e-18) for r in rows if r["backend"] == "FFT"],
              "o-", ms=3, label="FFT")
    ax.loglog([r["n"] for r in rows if r["backend"] == "FHT"],
              [max(r["max_abs_err"], 1e-18) for r in rows if r["backend"] == "FHT"],
              "s-", ms=3, label="FHT")
    ax.set_xlabel("n")
    ax.set_ylabel("max |z - scipy|")
    ax.set_title("agreement with scipy")
    ax.legend(fontsize=7)

    ax = axes[1]
    for mod in IMPLEMENTED:
        sub = [r for r in timings if r["backend"] == mod]
        ax.loglog([r["n"] for r in sub], [r["us_per_call"] for r in sub], "o-", ms=3, label=mod)
    ax.set_xlabel("n")
    ax.set_ylabel("us / call")
    ax.set_title("backend timing")
    ax.legend(fontsize=7)

    ax = axes[2]
    ax.semilogy([r["n"] for r in zp], [max(r["rel_l2_vs_scipy"], 1e-18) for r in zp],
                "o-", ms=3)
    ax.axhline(1e-13, ls="--", lw=0.8, color="gray", label="machine precision")
    ax.set_xlabel("n")
    ax.set_ylabel("relative L2 vs scipy")
    ax.set_title("hilbert_transform zero-pad artefact")
    ax.legend(fontsize=7)
    fig.tight_layout()
    p = B.savefig(fig, B.FIG / "utils" / "hilbert" / "hilbert_backends.png")
    plt.close(fig)
    print(f"wrote {B.TABLE / 'utils' / 'hilbert'}/*.csv")
    print(f"wrote {p}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    B.add_export_flag(ap)
    args = ap.parse_args()
    if args.export:
        export()
    else:
        B.print_table(build_agreement())


# keep `warnings` referenced for the docstring example above
_ = warnings
