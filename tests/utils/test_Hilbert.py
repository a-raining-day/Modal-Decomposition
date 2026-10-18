"""
tests for ``Modal_Decomposition.Utils.Hilbert`` (Hilbert 三个后端).

后端一览 (详见 ``docs/test/Hilbert_Backend_Report.md``):

* ``"FFT"``  — **默认**: rfft/irfft 实半谱法, 任意长度精确, 约为 scipy 的
  0.78–0.94×;
* ``"FHT"``  — **自适应**: 2 的幂且 n ≤ 512 走 SAO 编译内核 (最快),
  其余走实半谱法 —— 因此它也任意长度精确、并在所有尺寸上快于 scipy
  (0.45–0.88×);
* ``"scipy"`` — ``scipy.signal.hilbert``, 作为基线/参照。

覆盖:
1. 分发契约 —— 三个后端 + 枚举成员 + 未知/未实现模式 + 默认值;
2. 正确性 —— 全长度与 scipy 逐点一致; 解析式对照; 实部恒等于输入;
   纯/原生内核一致; ``matlab_phase``;
3. SAO 原语 —— ``hilbert_transform`` / ``fht_forward`` 仍保留"零填充到
   2 的幂"的语义, 本文件用 characterization test 固定该伪影, 并断言**后端
   分发已经绕开它** (非 2 的幂不再有伪影);
4. 边界 —— 2-D / 复数 / float32 / int / list / 空 / 长度 1–3 / NaN;
5. Cache —— scipy 后端惰性注册 ``scipy.signal``;
6. 性能烟测 —— 打印三个后端的耗时并做量级闸门。
"""

import time
import warnings

import numpy as np
import pytest

from src.Modal_Decomposition.Utils.Hilbert import (
    fht_forward,
    fht_inverse,
    hilbert,
    hilbert_transform,
)
from src.Modal_Decomposition.Utils._Hilbert import HILBERT_MODS, _fht

NATIVE = _fht._native is not None
ALL_LENGTHS = (256, 1000, 1023, 1024, 4096)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _tone(n: int, cycles: int) -> tuple[np.ndarray, np.ndarray]:
    """cos 单音及其理想相位 (scipy / MATLAB 约定下 analytic = exp(j·phase))。"""
    t = np.arange(n, dtype=np.float64)
    ph = 2 * np.pi * cycles * t / n
    return np.cos(ph), ph


def _am(n: int, fc: int, fm: int, depth: float = 0.5):
    """调幅信号及其理想包络。"""
    t = np.arange(n, dtype=np.float64)
    env = 1.0 + depth * np.cos(2 * np.pi * fm * t / n)
    return env * np.cos(2 * np.pi * fc * t / n), env


# --------------------------------------------------------------------------- #
# 1. 分发契约
# --------------------------------------------------------------------------- #
def test_scipy_backend_matches_scipy_direct():
    from scipy import signal as ss

    rng = np.random.default_rng(0)
    for n in (64, 256, 1000, 1024):
        S = rng.standard_normal(n)
        assert np.array_equal(hilbert(S, "scipy"), ss.hilbert(S))


@pytest.mark.parametrize("n", ALL_LENGTHS)
@pytest.mark.parametrize("mod", ["FFT", "FHT"])
def test_accelerated_backends_match_scipy_on_any_length(mod, n):
    """两个加速后端都必须与 scipy 逐点一致 —— 含**非 2 的幂**长度。"""
    from scipy import signal as ss

    rng = np.random.default_rng(1)
    S = rng.standard_normal(n)
    z = hilbert(S, mod)
    assert z.shape == S.shape and z.dtype == np.complex128
    assert np.max(np.abs(z - ss.hilbert(S))) < 1e-12


def test_default_backend_is_fft():
    S = np.sin(2 * np.pi * 0.05 * np.arange(256))
    assert hilbert.__defaults__[0] == "FFT"
    assert np.array_equal(hilbert(S), hilbert(S, "FFT"))


def test_enum_members_are_accepted():
    S = np.sin(2 * np.pi * 0.05 * np.arange(256))
    assert np.array_equal(hilbert(S, HILBERT_MODS.FFT), hilbert(S, "FFT"))
    assert np.array_equal(hilbert(S, HILBERT_MODS.FHT), hilbert(S, "FHT"))
    assert np.array_equal(hilbert(S, HILBERT_MODS.Scipy), hilbert(S, "scipy"))
    assert HILBERT_MODS.Scipy.value == "scipy"   # canonical 值是小写, 见报告 §8


def test_unknown_mod_raises_value_error():
    with pytest.raises(ValueError, match="mod"):
        hilbert(np.zeros(8), "bogus")


@pytest.mark.parametrize(
    "mod",
    ["Fourth-order Wave", "SB-Hilbert", "Kramers-Kronig", "FIR", "HST",
     "Optional-order FIR"],
)
def test_declared_but_unimplemented_mods_raise(mod):
    with pytest.raises((ValueError, NotImplementedError)):
        hilbert(np.zeros(8), mod)


@pytest.mark.xfail(
    strict=True,
    reason="已知不一致: 枚举值是 'scipy' (小写), 而 Hilbert.py 文档、LMD.py、"
           "Envelope.py 都传 'Scipy'; 修好后本用例会 XPASS, 请同步更新报告 §8",
)
def test_documented_capital_spelling_accepted():
    S = np.zeros(8)
    assert np.array_equal(hilbert(S, "Scipy"), hilbert(S, "scipy"))


# --------------------------------------------------------------------------- #
# 2. 数值正确性
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("mod", ["FFT", "FHT", "scipy"])
def test_analytic_of_cosine_matches_exponential(mod):
    """x = cos(θ) 的解析信号应为 exp(jθ) (虚部 = +sin)。"""
    n = 1024
    x, ph = _tone(n, n // 32)
    z = hilbert(x, mod)
    assert np.max(np.abs(z - np.exp(1j * ph))) < 1e-10
    assert np.max(np.abs(z.real - x)) < 1e-12


@pytest.mark.parametrize("mod", ["FFT", "FHT", "scipy"])
def test_am_envelope_recovers_modulation(mod):
    n = 2048
    x, env = _am(n, n // 8, n // 128)
    got = np.abs(hilbert(x, mod))
    assert np.max(np.abs(got - env) / env) < 1e-10


def test_sao_kernel_matches_pure_reference():
    """编译内核与纯 NumPy 参考实现必须给出同一结果 (两种相位约定)。"""
    rng = np.random.default_rng(2)
    for n in (256, 1024, 4096):
        S = rng.standard_normal(n)
        for mp in (True, False):
            native = hilbert_transform(S, matlab_phase=mp)
            pure = _fht._hilbert_transform_pure(S, matlab_phase=mp)
            assert np.max(np.abs(native - pure)) < 1e-12


def test_matlab_phase_flag_flips_imaginary_part():
    rng = np.random.default_rng(3)
    S = rng.standard_normal(1000)          # 非 2 的幂也走实半谱, 同样成立
    a = hilbert(S, "FFT", matlab_phase=False)
    b = hilbert(S, "FFT", matlab_phase=True)
    assert np.array_equal(a.real, b.real)
    assert np.max(np.abs(a.imag + b.imag)) < 1e-15
    assert np.max(np.abs(b - hilbert(S, "scipy"))) < 1e-12


def test_dispatch_uses_matlab_phase_by_default():
    """``hilbert(S, "FFT"/"FHT")`` 默认 matlab_phase=True (与 scipy 一致);
    原语 ``hilbert_transform(S)`` 默认 False (SAO +90°, 两者互为共轭)。"""
    from scipy import signal as ss

    S = np.sin(2 * np.pi * 0.05 * np.arange(1024))
    for mod in ("FFT", "FHT"):
        assert np.max(np.abs(hilbert(S, mod) - ss.hilbert(S))) < 1e-12
    assert np.max(np.abs(hilbert_transform(S) - np.conj(ss.hilbert(S)))) < 1e-12


def test_fht_and_fft_agree_everywhere():
    """自适应 FHT 与纯半谱 FFT 在两种路径上一致: 2 的幂(≤512 走 SAO 内核)
    与非 2 的幂(都走半谱) 都应在 1e-12 内相同。"""
    rng = np.random.default_rng(4)
    for n in (64, 256, 512, 1024, 1000, 4096):
        S = rng.standard_normal(n)
        assert np.max(np.abs(hilbert(S, "FHT") - hilbert(S, "FFT"))) < 1e-12


# --------------------------------------------------------------------------- #
# 3. SAO 原语的零填充语义 (characterization) 与"分发已修复"
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n", [1000, 1023, 2000])
def test_raw_sao_kernel_still_has_padding_artifact(n):
    """原语 ``hilbert_transform`` 依旧零填充 ⇒ 非 2 的幂上误差 O(1)。
    这是 SAO 内核的固有语义, 报告 §5.2 记录; 若哪天内核改成不填充, 本用例
    会失败, 请同步更新报告。"""
    from scipy import signal as ss

    rng = np.random.default_rng(5)
    S = rng.standard_normal(n)
    err = np.abs(hilbert_transform(S, matlab_phase=True) - ss.hilbert(S))
    assert err.max() > 1e-2
    assert np.mean(err > 1e-6) > 0.5


@pytest.mark.parametrize("n", [1000, 1023, 2000, 100000])
def test_dispatch_bypasses_padding_for_non_power_of_two(n):
    """后端分发对非 2 的幂走实半谱, 因此**没有**零填充伪影 (2026-10-19 修复)。"""
    from scipy import signal as ss

    rng = np.random.default_rng(6)
    S = rng.standard_normal(n)
    for mod in ("FFT", "FHT"):
        assert np.max(np.abs(hilbert(S, mod) - ss.hilbert(S))) < 1e-12


def test_raw_kernel_artifact_decays_away_from_edges():
    """原语的伪影在两端最大、向中段衰减 (1/距离)。"""
    from scipy import signal as ss

    n = 1025
    rng = np.random.default_rng(7)
    S = rng.standard_normal(n)
    err = np.abs(hilbert_transform(S, matlab_phase=True) - ss.hilbert(S))
    assert err[:32].mean() > err[n // 2 - 16:n // 2 + 16].mean() > 0


# --------------------------------------------------------------------------- #
# 4. FHT 原语
# --------------------------------------------------------------------------- #
def _dht_reference(x: np.ndarray) -> np.ndarray:
    """O(N²) 直算 DHT: H[k] = Σ_n x[n]·(cos + sin)(2πkn/N)。"""
    n = x.size
    k = np.arange(n)[:, None]
    j = np.arange(n)[None, :]
    return (x[None, :] * (np.cos(2 * np.pi * k * j / n)
                          + np.sin(2 * np.pi * k * j / n))).sum(axis=1)


@pytest.mark.parametrize("impl", ["native", "pure"])
def test_fht_forward_matches_dht_definition(impl):
    rng = np.random.default_rng(8)
    S = rng.standard_normal(64)
    ref = _dht_reference(S)
    got = fht_forward(S) if impl == "native" else _fht._fht_forward_pure(S)
    assert np.max(np.abs(got - ref)) < 1e-10


@pytest.mark.parametrize("n", [64, 1024, 4096])
def test_fht_self_inverse(n):
    rng = np.random.default_rng(9)
    S = rng.standard_normal(n)
    assert np.max(np.abs(fht_inverse(fht_forward(S)) - S)) < 1e-12


def test_fht_pads_to_next_power_of_two():
    rng = np.random.default_rng(10)
    S = rng.standard_normal(1000)
    assert fht_forward(S).size == 1024
    assert fht_inverse(S).size == 1024
    assert np.max(np.abs(fht_inverse(fht_forward(S)) - np.r_[S, np.zeros(24)])) < 1e-12


def test_fht_normalize_order_false_is_bit_reversed():
    """native 的 normalize_order=False 保留位反转序; 纯 NumPy 参考实现忽略该
    参数 —— 差异见报告 §8 (#10)。"""
    rng = np.random.default_rng(11)
    S = rng.standard_normal(256)
    normal = fht_forward(S, normalize_order=True)
    bitrev = fht_forward(S, normalize_order=False)
    if NATIVE:
        assert not np.allclose(normal, bitrev)
        assert np.allclose(np.sort(normal), np.sort(bitrev))
    else:
        assert np.allclose(normal, bitrev)


# --------------------------------------------------------------------------- #
# 5. 输入边界
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("mod", ["FFT", "FHT"])
def test_2d_input_rejected(mod):
    with pytest.raises(ValueError, match="1D|1-d|一维"):
        hilbert(np.zeros((3, 8)), mod)


def test_2d_input_accepted_by_scipy_backend():
    S2 = np.zeros((3, 8))
    assert hilbert(S2, "scipy").shape == S2.shape


@pytest.mark.parametrize("mod", ["FFT", "FHT"])
def test_complex_input_casts_with_warning(mod):
    S = np.zeros(8, dtype=np.complex128)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", np.exceptions.ComplexWarning)
        out = hilbert(S, mod)
    assert out.shape == S.shape
    with pytest.raises(ValueError):
        hilbert(S, "scipy")


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int32])
def test_real_input_dtypes(dtype):
    """FFT/FHT 后端一律返回 complex128; scipy 保留精度 (float32→complex64)。"""
    S = np.arange(32).astype(dtype)
    for mod in ("FFT", "FHT"):
        assert hilbert(S, mod).dtype == np.complex128
    expected = np.complex64 if np.dtype(dtype) == np.float32 else np.complex128
    assert hilbert(S, "scipy").dtype == expected


@pytest.mark.parametrize("mod", ["FFT", "FHT", "scipy"])
def test_list_input_accepted(mod):
    assert hilbert([0.0, 1.0, 0.0, -1.0], mod).shape == (4,)


@pytest.mark.parametrize("mod", ["FFT", "FHT"])
def test_empty_input_returns_empty(mod):
    assert hilbert(np.zeros(0), mod).shape == (0,)


def test_empty_input_scipy_raises():
    with pytest.raises(ValueError):
        hilbert(np.zeros(0), "scipy")


@pytest.mark.parametrize("n", [1, 2, 3])
@pytest.mark.parametrize("mod", ["FFT", "FHT"])
def test_tiny_inputs(mod, n):
    S = np.arange(n, dtype=np.float64)
    z = hilbert(S, mod)
    assert z.shape == S.shape
    assert np.allclose(z.real, S)


@pytest.mark.parametrize("mod", ["FFT", "FHT", "scipy"])
def test_nan_propagates(mod):
    S = np.sin(2 * np.pi * 0.05 * np.arange(256))
    S[10] = np.nan
    assert np.isnan(hilbert(S, mod)).any()


# --------------------------------------------------------------------------- #
# 6. Cache
# --------------------------------------------------------------------------- #
def test_scipy_backend_registers_scipy_signal_in_cache():
    from src.Modal_Decomposition.Base.Cache import cache

    hilbert(np.zeros(16), "scipy")
    assert cache.check("scipy.signal")


def test_hilbert_facade_exports():
    import src.Modal_Decomposition.Utils as U

    mod = U.Hilbert
    for name in ("hilbert", "fht_forward", "fht_inverse", "hilbert_transform"):
        assert callable(getattr(mod, name))


# --------------------------------------------------------------------------- #
# 7. 性能烟测 (数字供 docs/test/Hilbert_Backend_Report.md)
# --------------------------------------------------------------------------- #
def _median_seconds(fn, repeats=7):
    fn()  # warm-up
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


@pytest.mark.parametrize("n", [256, 16384])
def test_performance_smoke(n, capsys):
    rng = np.random.default_rng(12)
    S = rng.standard_normal(n)

    t_scipy = _median_seconds(lambda: hilbert(S, "scipy"))
    t_fft = _median_seconds(lambda: hilbert(S, "FFT"))
    t_fht = _median_seconds(lambda: hilbert(S, "FHT"))

    with capsys.disabled():
        print(
            f"\n[hilbert-perf] n={n}: scipy={t_scipy * 1e3:.3f} ms | "
            f"FFT={t_fft * 1e3:.3f} ms ({t_fft / t_scipy:.2f}x) | "
            f"FHT={t_fht * 1e3:.3f} ms ({t_fht / t_scipy:.2f}x) | "
            f"native_kernel={NATIVE}"
        )

    # 量级闸门: 单次调用必须在 50 ms 内
    assert max(t_scipy, t_fft, t_fht) < 0.05
