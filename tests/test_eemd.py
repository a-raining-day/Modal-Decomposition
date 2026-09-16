"""
tests for ``Modal_Decomposition.EEMD`` (native implementation).

背景: `EEMD` 自 0.3.0 起由**原生实现**承担 (原 `EEMD_new`), 不再依赖 PyEMD;
`pyemd=True` 保留一条到第三方 PyEMD 的过渡通道。本文件是该换版的专属回归 ——
此前原生实现只被 ``tests/_cases.py`` 的契约矩阵覆盖, 没有算法级测试。

覆盖:
1. 重构精确性 —— ``ΣIMFs + Res == S`` (对旧 PyEMD 包装"把最后一行 IMF 当残差"
   的语义修正, 旧包装实测误差 0.27);
2. 结果契约、config 快照 (含 ``backend``)、``info`` 诊断;
3. 集成语义 —— 噪声幅度口径、trials 生效、``ensemble_std``、逐轮即时截断;
4. 可复现性 (同 seed 逐位相同 / 全局 seed 覆盖局部 seed);
5. ``pyemd=True`` 过渡分支 —— 可用、backend 标记正确、且**同样满足精确重构**;
6. ``**kwargs`` 透传内层 EMD 引擎 (sd_thr / max_iter / find_peaks_mod / faster)
   并拒绝未知键;
7. 参数校验与边界输入 (常量 / 单调 / 极短 / max_imf 截断 / workers 互斥);
8. facade 等价与性能烟测。
"""

import time

import numpy as np
import pytest

from src.Modal_Decomposition import Class, Function
from src.Modal_Decomposition.Base import (
    Decomposer,
    DecompositionResult,
    Name,
    Reference,
)
from src.Modal_Decomposition.EEMD import EEMD, EEMDConfig
from src.Modal_Decomposition.EMD import EMD

pytestmark = pytest.mark.filterwarnings("ignore:.*:UserWarning")

#: 测试统一用的小规模参数 (默认 trials=100 太慢)。
FAST = dict(trials=10, seed=0)


def _two_tone(n=512, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, n, endpoint=False)
    S = np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 37 * t)
    if noise:
        S = S + noise * rng.standard_normal(n)
    return S, t


# --------------------------------------------------------------------------- #
# 1. 重构精确性 (旧包装的语义修正)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n", [64, 256, 1024])
def test_reconstruction_is_exact(n):
    S, _ = _two_tone(n)
    r = EEMD(**FAST).decompose(S)
    err = float(np.max(np.abs(r.reconstruct() - S)))
    assert r.IMFs.ndim == 2 and r.IMFs.shape[1] == n
    assert r.Res.shape == (n,)
    assert err < 1e-9, f"重构误差 {err:.3e}"


@pytest.mark.parametrize("noise", [0.0, 0.15, 0.6])
def test_reconstruction_is_exact_under_noise(noise):
    S, _ = _two_tone(512, noise=noise)
    r = EEMD(**FAST).decompose(S)
    assert np.allclose(r.reconstruct(), S, atol=1e-9)


def test_reconstruction_exact_with_dc():
    S, _ = _two_tone(512, noise=0.2)
    S = S + 5.0
    r = EEMD(**FAST).decompose(S)
    assert np.max(np.abs(r.reconstruct() - S)) < 1e-9


def test_residual_is_true_remainder_not_an_imf_row():
    """
    核心语义修正: ``Res`` 必须是 ``S − ΣIMFs``, 而**不是** PyEMD 堆栈的末行。
    旧包装把末行当残差, 导致少输出一阶 IMF 且重构不精确。
    """
    S, _ = _two_tone(512, noise=0.15)
    r = EEMD(**FAST).decompose(S)
    expected = np.asarray(S, dtype=np.float64) - r.IMFs.sum(axis=0)
    assert np.allclose(r.Res, expected, atol=1e-12)


# --------------------------------------------------------------------------- #
# 2. 契约 / config / info
# --------------------------------------------------------------------------- #
def test_result_contract_matches_framework():
    S, _ = _two_tone(256, noise=0.2)
    r = EEMD(**FAST).decompose(S)

    assert isinstance(r, DecompositionResult)
    assert isinstance(EEMD(), Decomposer)
    assert EEMD.name == "EEMD"
    assert EEMD().name == "EEMD"
    assert EEMD().full_name == Name["EEMD"]
    assert EEMD().reference == Reference["EEMD"]
    assert isinstance(r.config, EEMDConfig)
    assert r.n_imfs == r.IMFs.shape[0]
    assert r.shape == tuple(r.IMFs.shape)
    assert r.IMFs.dtype == np.float64 and r.Res.dtype == np.float64


def test_call_alias_equals_decompose():
    S, _ = _two_tone(256, noise=0.2)
    a = EEMD(**FAST)(S)
    b = EEMD(**FAST).decompose(S)
    assert np.array_equal(a.IMFs, b.IMFs)
    assert np.array_equal(a.Res, b.Res)


def test_registered_under_official_key():
    assert Class.EEMD is EEMD
    assert hasattr(Function, "EEMD")
    assert not hasattr(Class, "EEMD_new"), "换版后不应再暴露 _new 键"


def test_config_snapshot_fields_and_backend():
    r = EEMD(**FAST).decompose(_two_tone(256, noise=0.2)[0])
    d = r.config.to_dict()
    for key in ("trials", "noise_width", "max_imf", "seed", "parallel",
                "workers", "spline_kind", "nbsym", "backend"):
        assert key in d, f"config 缺少 {key}"
    assert d["backend"] == "native"
    assert d["trials"] == FAST["trials"]
    assert d["seed"] == 0


def test_config_records_effective_seed():
    r = EEMD(trials=5, seed=17).decompose(_two_tone(256)[0])
    assert r.config.to_dict()["seed"] == 17


def test_info_diagnostics_present():
    S, _ = _two_tone(512, noise=0.2)
    r = EEMD(**FAST).decompose(S)
    for key in ("iterations", "n_trials", "workers"):
        assert key in r.info, f"info 缺少 {key}"
    # iterations 按阶给出, 每阶 trials 个计数
    assert len(r.info["iterations"]) == r.IMFs.shape[0]
    for per_order in r.info["iterations"]:
        assert len(per_order) == FAST["trials"]
        assert all(i >= 1 for i in per_order)
    assert r.info["n_trials"] == FAST["trials"]
    assert r.info["workers"] == 1


def test_rich_info_adds_ensemble_std():
    S, _ = _two_tone(512, noise=0.2)
    plain = EEMD(**FAST).decompose(S)
    rich = EEMD(**FAST, rich_info=True).decompose(S)

    assert "ensemble_std" not in plain.info
    assert "ensemble_std" in rich.info
    assert rich.info["ensemble_std"].shape == rich.IMFs.shape
    assert np.all(rich.info["ensemble_std"] >= 0.0)
    # rich_info 只加诊断, 不得改变分解结果
    assert np.array_equal(plain.IMFs, rich.IMFs)


# --------------------------------------------------------------------------- #
# 3. 集成语义
# --------------------------------------------------------------------------- #
def test_noise_width_actually_changes_output():
    """噪声幅度必须是真旋钮 (相对信号峰峰值, 与 PyEMD 同口径)。"""
    S, _ = _two_tone(512, noise=0.15)
    a = EEMD(trials=10, seed=0, noise_width=0.01).decompose(S)
    b = EEMD(trials=10, seed=0, noise_width=0.3).decompose(S)
    changed = a.IMFs.shape != b.IMFs.shape or not np.allclose(a.IMFs, b.IMFs, atol=0)
    assert changed, "noise_width 未改变输出"


def test_trials_actually_changes_output():
    S, _ = _two_tone(512, noise=0.15)
    a = EEMD(trials=3, seed=0).decompose(S)
    b = EEMD(trials=30, seed=0).decompose(S)
    changed = a.IMFs.shape != b.IMFs.shape or not np.allclose(a.IMFs, b.IMFs, atol=0)
    assert changed, "trials 未改变输出"


def test_zero_noise_width_reduces_to_single_emd_average():
    """
    noise_width=0 时每次试验的扰动消失, 全部 trial 同解 ⇒ 结果应等于单次
    ``EMD`` (逐轮即时截断可能少一阶, 故只要求"共同阶"逐点一致)。
    """
    S, _ = _two_tone(512)
    r = EEMD(trials=5, seed=0, noise_width=0.0).decompose(S)
    ref = EMD().decompose(S)
    k = min(r.IMFs.shape[0], ref.IMFs.shape[0])
    assert k > 0
    for i in range(k):
        assert np.allclose(r.IMFs[i], ref.IMFs[i], atol=1e-12), f"阶{i} 不一致"


def test_signal_tone_is_captured_in_a_single_mode():
    """
    37 Hz 真值分量必须被**某一阶**捕获到 (corr > 0.9)。

    注意**不能**断言"首阶跟踪最高频信号分量" (那是裸 ``EMD`` 的测试口径):
    EEMD 每次都往信号里注入白噪声, 注入噪声必然占据最高频, 故首阶是噪声
    (实测主频 ~448 Hz), 而 37 Hz 落在第 2 阶。实测该归属在
    ``trials`` ∈ {10, 50}、``seed`` ∈ {0,1,2} 下稳定 (corr 0.96–0.98)。
    """
    S, t = _two_tone(1024, noise=0.15)
    r = EEMD(trials=20, seed=0).decompose(S)
    assert r.IMFs.shape[0] >= 3

    high = np.sin(2 * np.pi * 37 * t)
    corrs = [abs(float(np.corrcoef(r.IMFs[i], high)[0, 1])) for i in range(r.IMFs.shape[0])]
    best = int(np.argmax(corrs))
    assert corrs[best] > 0.9, f"37Hz 未被任一阶捕获, 最佳 corr={corrs[best]:.4f} (阶{best})"
    # 且该阶的谱峰应落在 37 Hz 附近
    spec = np.abs(np.fft.rfft(r.IMFs[best]))
    freqs = np.fft.rfftfreq(S.size, 1.0 / 1024.0)
    peak_hz = float(freqs[np.argmax(spec[1:]) + 1])
    assert abs(peak_hz - 37.0) < 5.0, f"最佳匹配阶的主频为 {peak_hz:.1f} Hz"


def test_ensemble_std_grows_then_converges_with_trials():
    """
    ``ensemble_std`` 是**实现间离散度**的估计, 不是"平均后的噪声残余" ——
    它对 trials 单调下降的直觉是错的。实测 (3 个 seed 的均值):
    trials 5→10→20→50→100 给出 0.161→0.217→0.230→0.230→0.230, 即
    **先随样本数上升、随后收敛** (样本少时逐点标准差被系统性低估)。
    """
    S, _ = _two_tone(512, noise=0.15)
    means = []
    for tr in (5, 10, 20, 50):
        vals = [
            float(EEMD(trials=tr, seed=sd, rich_info=True).decompose(S).info["ensemble_std"].mean())
            for sd in (0, 1, 2)
        ]
        means.append(float(np.mean(vals)))

    # 前段上升 (样本不足)
    assert means[1] > means[0], f"trials 5→10 未上升: {means}"
    # 后段收敛 (20 -> 50 变化 < 1%)
    assert abs(means[3] - means[2]) / means[2] < 0.01, f"20→50 未收敛: {means}"
    # 全程非负且有限
    assert all(np.isfinite(m) and m >= 0.0 for m in means)


def test_trial_truncation_keeps_all_orders_populated():
    """
    逐轮即时截断: 每阶必须恰好有 ``trials`` 个真实样本 (不做零填充),
    故 ``iterations`` 的每阶长度都等于 trials。
    """
    S, _ = _two_tone(1024, noise=0.3)
    r = EEMD(trials=12, seed=0).decompose(S)
    for per_order in r.info["iterations"]:
        assert len(per_order) == 12


# --------------------------------------------------------------------------- #
# 4. 可复现性
# --------------------------------------------------------------------------- #
def test_same_seed_is_bitwise_identical():
    S, _ = _two_tone(512, noise=0.15)
    a = EEMD(**FAST).decompose(S)
    b = EEMD(**FAST).decompose(S)
    assert np.array_equal(a.IMFs, b.IMFs)
    assert np.array_equal(a.Res, b.Res)
    assert a.info["iterations"] == b.info["iterations"]


def test_different_seed_differs():
    S, _ = _two_tone(512, noise=0.15)
    a = EEMD(trials=10, seed=1).decompose(S)
    b = EEMD(trials=10, seed=2).decompose(S)
    assert not np.allclose(a.IMFs, b.IMFs)


def test_global_seed_overrides_local_seed():
    from src.Modal_Decomposition import set_seed

    S, _ = _two_tone(512, noise=0.15)
    ref = EEMD(trials=10, seed=5).decompose(S)

    set_seed(5)
    try:
        with pytest.warns(UserWarning, match="global seed"):
            r = EEMD(trials=10, seed=99).decompose(S)
        assert r.config.to_dict()["seed"] == 5
        assert np.allclose(r.IMFs, ref.IMFs)
    finally:
        set_seed(None)


# --------------------------------------------------------------------------- #
# 5. pyemd=True 过渡分支
# --------------------------------------------------------------------------- #
def test_pyemd_branch_runs_and_marks_backend():
    pytest.importorskip("PyEMD")
    S, _ = _two_tone(512, noise=0.15)
    r = EEMD(trials=10, seed=0, pyemd=True).decompose(S)
    assert r.config.to_dict()["backend"] == "pyemd"
    assert r.info["backend"] == "pyemd"
    assert r.IMFs.shape[1] == S.size


def test_pyemd_branch_also_reconstructs_exactly():
    """
    PyEMD 的 ``eemd()`` 不返回残差行; 本分支按库内契约由总和反推 ``Res``,
    故即便走 PyEMD 也满足精确重构 (旧包装正是在这里不成立)。
    """
    pytest.importorskip("PyEMD")
    S, _ = _two_tone(512, noise=0.15)
    r = EEMD(trials=10, seed=0, pyemd=True).decompose(S)
    assert np.max(np.abs(r.reconstruct() - S)) < 1e-9


def test_pyemd_branch_exposes_its_own_residue():
    pytest.importorskip("PyEMD")
    S, _ = _two_tone(512, noise=0.15)
    r = EEMD(trials=10, seed=0, pyemd=True).decompose(S)
    assert "pyemd_residue" in r.info
    assert r.info["pyemd_residue"] is not None


def test_pyemd_branch_differs_from_native():
    """两条路径是不同实现, 不应给出逐位相同的结果。"""
    pytest.importorskip("PyEMD")
    S, _ = _two_tone(512, noise=0.15)
    n = EEMD(**FAST).decompose(S)
    p = EEMD(**FAST, pyemd=True).decompose(S)
    same = n.IMFs.shape == p.IMFs.shape and np.allclose(n.IMFs, p.IMFs, atol=0)
    assert not same, "原生与 PyEMD 路径输出完全相同, 可疑"


# --------------------------------------------------------------------------- #
# 6. **kwargs 透传内层 EMD 引擎
# --------------------------------------------------------------------------- #
#: 各 ``**kwargs`` 键改变输出所需的最小可测差异 (相对模式幅度)。
#: ``atol=0`` 的 ``np.allclose`` 在个别信号上会因浮点巧合判定"相同",
#: 故这里用"相对该阶最大幅度"的显式阈值。
_ENGINE_EFFECT_TOL = 1e-9


@pytest.mark.parametrize("kw", [
    {"sd_thr": 0.3},
    {"max_iter": 3},
    {"faster": True},
])
@pytest.mark.parametrize("noise", [0.0, 0.15])
def test_engine_kwargs_change_output(kw, noise):
    """
    ``**kwargs`` 必须真的透传到内层 EMD (否则就是"接受了却忽略"的坏味道)。

    注意 ``faster`` 的方向: 本实现**不显式传**该键, 内层 EMD 走自身默认
    ``faster=False`` (质量档), 故 ``faster=False`` 与默认逐位相同 —— 那是
    "透传正确"的表现, 不是死参数。要证明透传必须传 ``faster=True``。

    (``find_peaks_mod`` 例外: numpy 与 scipy 后端按 ``Utils.Peaks`` 契约在无平台
    信号上结果一致, 见下条。)
    """
    S, _ = _two_tone(512, noise=noise)
    base = EEMD(**FAST).decompose(S)
    got = EEMD(**FAST, **kw).decompose(S)

    changed = got.IMFs.shape != base.IMFs.shape
    if not changed:
        amp = float(np.max(np.abs(base.IMFs))) + 1e-300
        changed = float(np.max(np.abs(got.IMFs - base.IMFs))) > _ENGINE_EFFECT_TOL * amp
    assert changed, f"{kw} 未改变输出, 疑为未透传"


def test_faster_false_equals_default_because_default_is_false():
    """默认内层档位就是 ``faster=False``, 故显式传它应与默认逐位相同。"""
    S, _ = _two_tone(512, noise=0.15)
    a = EEMD(**FAST).decompose(S)
    b = EEMD(**FAST, faster=False).decompose(S)
    assert EEMD(**FAST).engine_kwargs == {}, "默认不应注入任何引擎 kwargs"
    assert np.array_equal(a.IMFs, b.IMFs)


def test_find_peaks_backend_accepted_and_consistent():
    S, _ = _two_tone(512)
    a = EEMD(**FAST, find_peaks_mod="numpy").decompose(S)
    b = EEMD(**FAST, find_peaks_mod="scipy").decompose(S)
    assert a.IMFs.shape == b.IMFs.shape
    assert np.allclose(a.IMFs, b.IMFs, atol=1e-12)


def test_unknown_kwarg_rejected():
    for kw in (dict(bogus=1), dict(alpha=5), dict(max_imf_kw=2)):
        with pytest.raises(ValueError):
            EEMD(**kw)


# --------------------------------------------------------------------------- #
# 7. 参数校验与边界输入
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("kw", [
    dict(trials=0),
    dict(trials=1.5),
    dict(trials=True),
    dict(noise_width=-1),
    dict(noise_width=np.nan),
    dict(max_imf=0),
    dict(max_imf=1.5),
    dict(spline_kind="cubic"),
    dict(nbsym=-1),
    dict(parallel=1),
    dict(rich_info=1),
    dict(pyemd=1),
])
def test_invalid_parameters_raise(kw):
    with pytest.raises(ValueError):
        EEMD(**kw)


def test_workers_and_cpu_ratio_are_mutually_exclusive():
    with pytest.raises(ValueError):
        EEMD(workers=2, cpu_ratio=0.5)


def test_constant_signal_returns_no_imfs():
    S = np.full(256, 3.0)
    r = EEMD(**FAST).decompose(S)
    assert r.IMFs.shape == (0, 256)
    assert np.array_equal(r.reconstruct(), S)
    assert r.info["n_trials"] == 0


def test_zero_signal_returns_no_imfs():
    r = EEMD(**FAST).decompose(np.zeros(256))
    assert r.IMFs.shape == (0, 256)


def test_too_short_signal_handled():
    """极短信号要么给出结果, 要么明确报错 —— 不得静默产生 NaN。"""
    for n in (4, 8, 16):
        S = np.sin(np.linspace(0, 6.0, n))
        try:
            r = EEMD(trials=2, seed=0).decompose(S)
        except ValueError:
            continue
        assert np.all(np.isfinite(r.IMFs))
        assert np.allclose(r.reconstruct(), S, atol=1e-9)


def test_max_imf_truncates():
    S, _ = _two_tone(512, noise=0.15)
    full = EEMD(**FAST).decompose(S)
    assert full.IMFs.shape[0] > 1
    capped = EEMD(**FAST, max_imf=1).decompose(S)
    assert capped.IMFs.shape[0] == 1
    assert np.allclose(capped.reconstruct(), S, atol=1e-9)


def test_multichannel_rejected():
    with pytest.raises(ValueError):
        EEMD(**FAST).decompose(np.zeros((2, 64)))


def test_time_axis_validated():
    S, t = _two_tone(256, noise=0.2)
    a = EEMD(**FAST).decompose(S)
    b = EEMD(**FAST).decompose(S, t)
    assert np.array_equal(a.IMFs, b.IMFs)
    with pytest.raises(ValueError):
        EEMD(**FAST).decompose(S, np.arange(10))


def test_small_workload_falls_back_to_serial():
    """工作量低于阈值时 parallel=True 也应退化为串行 (workers 报 1)。"""
    S, _ = _two_tone(256, noise=0.2)
    r = EEMD(trials=4, seed=0, parallel=True).decompose(S)
    assert r.info["workers"] == 1


# --------------------------------------------------------------------------- #
# 8. facade 与性能烟测
# --------------------------------------------------------------------------- #
def test_facade_matches_class():
    S, _ = _two_tone(512, noise=0.15)
    a = Function.EEMD(S, **FAST)
    b = Class.EEMD(**FAST).decompose(S)
    assert np.array_equal(a.IMFs, b.IMFs)
    assert np.array_equal(a.Res, b.Res)
    assert a.config.to_dict() == b.config.to_dict()


def test_facade_docstring_is_composed():
    doc = Function.EEMD.__doc__
    assert "Ensemble Empirical Mode Decomposition" in doc
    assert "Parameters" in doc
    assert "References" in doc
    assert "10.1142/S1793536909000047" in doc
    assert "Random seed" in doc
    assert "EEMDConfig" in doc


def test_does_not_mutate_input():
    S, _ = _two_tone(512, noise=0.2)
    ref = S.copy()
    EEMD(**FAST).decompose(S)
    assert np.array_equal(S, ref), "decompose 不得改写输入"


def test_performance_smoke():
    S, _ = _two_tone(2048, noise=0.15)
    EEMD(trials=5, seed=0).decompose(S)          # 预热 (惰性 import / 缓存)
    t0 = time.perf_counter()
    EEMD(trials=20, seed=0).decompose(S)
    dt = time.perf_counter() - t0
    assert dt < 10.0, f"trials=20, N=2048 耗时 {dt:.2f}s"
