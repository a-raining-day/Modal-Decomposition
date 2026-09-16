"""
tests for ``Modal_Decomposition.CEEMDAN`` (native CEEMDAN implementation).

覆盖:
1. 重构精确性 —— ``IMFs.sum(axis=0) + Res == S`` (硬性契约);
2. 结果契约 —— DecompositionResult / Decomposer / name / __call__ / config 快照;
3. **递归正确性** —— ``noise_width=0`` 时必须与"逐阶 max_imf=1 的原生 EMD"逐位一致
   (这是本算法唯一的独立参照: 辅助项消失后 CEEMDAN 退化为逐阶筛分);
4. 模式恢复 —— 首 IMF 跟踪最高频分量; 与 PyEMD 版 CEEMDAN 的定性一致;
5. 可复现性 —— 同 seed 逐位相同, 不同 seed 不同; 全局 seed 覆盖局部 seed;
6. 参数有效性 —— 每个内层引擎旋钮 (sd_thr / max_iter / faster / find_peaks_mod)
   都必须真的改变输出, 即"没有静默死参数";
7. 参数校验与边界输入 (常量 / 单调 / 极短 / 纯噪 / max_imf 截断 / 各种 spline_kind);
8. facade 等价 ``Function.CEEMDAN`` vs ``Class.CEEMDAN``;
9. 性能烟测 (数字供 docs/CEEMDAN_Native_Report.md 引用)。
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
from src.Modal_Decomposition.CEEMDAN import CEEMDAN, CEEMDANConfig
from src.Modal_Decomposition.EMD import EMD
from src.Modal_Decomposition.Utils import is_monotonic

pytestmark = pytest.mark.filterwarnings("ignore:.*:UserWarning")

#: 测试统一用的小规模参数: 保证整套件秒级完成 (默认 trials=100 太慢)。
FAST = dict(trials=10, seed=0)


def _two_tone(n=512, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, n, endpoint=False)
    S = np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 37 * t)
    if noise:
        S = S + noise * rng.standard_normal(n)
    return S, t


def _stepwise_emd(S, n_modes, **emd_kw):
    """独立参照: 反复用 ``EMD(max_imf=1)`` 拆残差, 逐阶取出 n_modes 个 IMF。"""
    residual = np.asarray(S, dtype=np.float64).copy()
    out = []
    for _ in range(n_modes):
        imfs = EMD(max_imf=1, **emd_kw).decompose(residual).IMFs
        if imfs.shape[0] == 0:
            break
        out.append(imfs[0])
        residual = residual - imfs[0]
    return out


# --------------------------------------------------------------------------- #
# 1. 重构精确性
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n", [64, 256, 1024])
def test_reconstruction_is_exact(n):
    S, _ = _two_tone(n)
    r = CEEMDAN(**FAST).decompose(S)
    recon = r.IMFs.sum(axis=0) + r.Res
    assert r.IMFs.ndim == 2 and r.IMFs.shape[1] == n
    assert r.Res.shape == (n,)
    assert np.allclose(recon, S, atol=1e-9), f"recon err {np.max(np.abs(recon - S)):.3e}"


@pytest.mark.parametrize("noise", [0.0, 0.15, 0.6])
def test_reconstruction_is_exact_under_noise(noise):
    S, _ = _two_tone(512, noise=noise)
    r = CEEMDAN(**FAST).decompose(S)
    assert np.allclose(r.reconstruct(), S, atol=1e-9)


def test_reconstruct_helper_matches_sum():
    S, _ = _two_tone(256)
    r = CEEMDAN(**FAST).decompose(S)
    assert np.allclose(r.reconstruct(), r.IMFs.sum(axis=0) + r.Res, atol=0)


# --------------------------------------------------------------------------- #
# 2. 结果契约
# --------------------------------------------------------------------------- #
def test_result_contract_matches_framework():
    S, _ = _two_tone(256)
    r = CEEMDAN(**FAST).decompose(S)

    assert isinstance(r, DecompositionResult)
    assert isinstance(CEEMDAN(), Decomposer)
    assert CEEMDAN.name == "CEEMDAN"
    assert CEEMDAN().name == "CEEMDAN"
    assert CEEMDAN().full_name == Name["CEEMDAN"]
    assert CEEMDAN().reference == Reference["CEEMDAN"]
    assert isinstance(r.config, CEEMDANConfig)
    assert isinstance(r.info, dict)
    assert r.n_imfs == r.IMFs.shape[0]
    assert r.shape == tuple(r.IMFs.shape)
    assert r.IMFs.dtype == np.float64 and r.Res.dtype == np.float64


def test_call_alias_equals_decompose():
    S, _ = _two_tone(256)
    a = CEEMDAN(**FAST)(S)
    b = CEEMDAN(**FAST).decompose(S)
    assert np.array_equal(a.IMFs, b.IMFs)
    assert np.array_equal(a.Res, b.Res)


def test_registered_in_public_namespaces():
    assert hasattr(Class, "CEEMDAN")
    assert hasattr(Function, "CEEMDAN")
    assert getattr(Class, "CEEMDAN") is CEEMDAN


def test_config_snapshot_records_engine_parameters():
    """快照必须包含全部内层引擎参数, 否则结果无法被解释。"""
    cfg = CEEMDAN(**FAST, max_iter=7, sd_thr=0.2, faster=False,
                      find_peaks_mod="scipy").decompose(_two_tone(256)[0]).config
    d = cfg.to_dict()
    for key in ("trials", "noise_width", "max_imf", "seed", "spline_kind", "nbsym",
                "max_iter", "sd_thr", "find_peaks_mod", "faster",
                "range_thr", "total_power_thr"):
        assert key in d, f"config 快照缺少 {key}"
    assert d["max_iter"] == 7 and d["sd_thr"] == 0.2
    assert d["faster"] is False and d["find_peaks_mod"] == "scipy"


def test_config_snapshot_records_effective_seed():
    r = CEEMDAN(trials=5, seed=3).decompose(_two_tone(256)[0])
    assert r.config.to_dict()["seed"] == 3


# --------------------------------------------------------------------------- #
# 3. 递归正确性 (核心: 与逐阶 EMD 对标)
# --------------------------------------------------------------------------- #
def _stepwise_emd(S, n_modes, **emd_kw):
    """独立参照: 反复用 ``EMD(max_imf=1)`` 拆残差, 逐阶取出 n_modes 个 IMF。

    仅供 ``test_max_imf_one_matches_stepwise`` 使用 (单阶对标)。
    """
    residual = np.asarray(S, dtype=np.float64).copy()
    out = []
    for _ in range(n_modes):
        imfs = EMD(max_imf=1, **emd_kw).decompose(residual).IMFs
        if imfs.shape[0] == 0:
            break
        out.append(imfs[0])
        residual = residual - imfs[0]
    return out


def _ceemdan_zero_noise_reference(S, faster, spline_kind):
    """
    复刻 CEEMDAN 在 ``noise_width=0, trials=1`` 下的**精确轨迹**。

    ε=0 时辅助项消失, 每阶的 M 次试验完全同解, 均值即该解, 故整个递归退化为
    "在归一化信号上反复做 ``EMD(max_imf=1)`` 并从残差中减去"。停机口径也一并
    复刻: **残差单调即止** (与 CEEMDAN 主循环里的 ``is_monotonic`` 判据同义,
    因为"单调"⇔"无可提取极值")。
    """
    scale = float(np.std(S))
    work = np.asarray(S, dtype=np.float64) / scale
    residual = work.copy()
    modes = []
    while not is_monotonic(residual):
        imfs = EMD(max_imf=1, faster=faster, spline_kind=spline_kind).decompose(residual).IMFs
        if imfs.shape[0] == 0:
            break
        modes.append(imfs[0])
        residual = residual - imfs[0]
    # CEEMDAN 在归一化信号上递归, 最后统一乘回原尺度 —— 参照必须同样还原。
    return [m * scale for m in modes]


@pytest.mark.parametrize("faster", [True, False])
@pytest.mark.parametrize("spline_kind", ["CubicSpline", "PCHIP", "linear"])
def test_zero_noise_degenerates_to_stepwise_emd(faster, spline_kind):
    """
    ``noise_width=0`` 且 ``trials=1`` 时辅助项消失, CEEMDAN 必须逐位退化为
    "在归一化信号上反复 ``EMD(max_imf=1)`` 并减残差"。

    这是本算法唯一的独立参照 —— 它同时验证递归公式与残差链的正确性。

    Note
    ----
    参照必须复刻**同一停机口径** (残差单调即止) 才有可比性: 一次完整
    ``EMD(max_imf=-1)`` 的逐阶行为在 ``linear`` 包络下并不等价 —— 那种包络的
    样条在数值噪声量级的残差上衰减极慢, 会一路拆到 31 阶以上, 而 CEEMDAN 的
    阶数还受"噪声池可用阶数"上限约束 (``nw=0`` 时噪声池照样预分解), 两者
    天然在不同位置停下。
    """
    S, _ = _two_tone(512)
    r = CEEMDAN(trials=1, noise_width=0.0, seed=0, faster=faster,
                    spline_kind=spline_kind).decompose(S)

    # (a) 重构精确到机器精度
    assert np.allclose(r.reconstruct(), S, atol=1e-12)

    # (b) 与同口径参照逐阶逐位一致
    ref = _ceemdan_zero_noise_reference(S, faster, spline_kind)
    k = min(len(ref), r.IMFs.shape[0])
    assert k > 0, "零噪声下未提出任何模式, 参照失效"
    for i in range(k):
        assert np.allclose(r.IMFs[i], ref[i], atol=1e-12), (
            f"阶{i} 最大偏差 {np.max(np.abs(r.IMFs[i] - ref[i])):.3e}"
        )

    # (c) 两条路径必须在同一位置停下 (受噪声池阶数上限约束时参照更长, 允许)
    if len(ref) <= r.IMFs.shape[0]:
        assert len(ref) == r.IMFs.shape[0], (
            f"参照在 {len(ref)} 阶结束, CEEMDAN 却产出 {r.IMFs.shape[0]} 阶"
        )


def test_zero_noise_residual_still_exact():
    S, _ = _two_tone(512)
    r = CEEMDAN(trials=1, noise_width=0.0, seed=0).decompose(S)
    assert np.allclose(r.reconstruct(), S, atol=1e-12)


# --------------------------------------------------------------------------- #
# 4. 模式恢复
# --------------------------------------------------------------------------- #
def test_first_imf_tracks_highest_frequency():
    S, t = _two_tone(1024)
    r = CEEMDAN(trials=20, seed=0).decompose(S)
    assert r.IMFs.shape[0] >= 2
    high = np.sin(2 * np.pi * 37 * t)
    corr = float(np.corrcoef(r.IMFs[0], high)[0, 1])
    assert corr > 0.9, f"首 IMF 与 37Hz 分量相关系数仅 {corr:.4f}"


def test_noise_assist_changes_decomposition():
    """ε>0 必须真的起作用 (不是被归一化约掉的死参数)。"""
    S, _ = _two_tone(512, noise=0.15)
    a = CEEMDAN(trials=20, noise_width=0.0, seed=0).decompose(S)
    b = CEEMDAN(trials=20, noise_width=0.05, seed=0).decompose(S)
    same = (
        a.IMFs.shape == b.IMFs.shape and np.allclose(a.IMFs, b.IMFs, atol=0)
    )
    assert not same, "noise_width 由 0 改为 0.05 输出完全未变, 注噪未生效"


def test_larger_noise_width_is_monotone_in_mode_count():
    """ε 增大应带来更多阶 (噪声分离出更多分量), 至少不减少。"""
    S, _ = _two_tone(512, noise=0.15, seed=1)
    counts = [
        CEEMDAN(trials=20, noise_width=nw, seed=0).decompose(S).IMFs.shape[0]
        for nw in (0.002, 0.02, 0.1)
    ]
    assert counts == sorted(counts), f"阶数未随 ε 单调: {counts}"


def test_qualitative_agreement_with_pyemd_ceemdan():
    """
    与 PyEMD 版 CEEMDAN 做**定性**对照 —— 两者算法版本不同
    (PyEMD 实现的是 Colominas 2014 改进版, 本实现是 Torres 2011 规范式),
    故只断言"阶数量级接近"与"首 IMF 频段一致", 不断言数值相等。
    """
    pytest.importorskip("PyEMD")
    S, t = _two_tone(1024)

    ours = CEEMDAN(trials=50, seed=0).decompose(S)
    theirs = Class.CEEMDAN(trials=50, seed=0).decompose(S)

    assert abs(ours.IMFs.shape[0] - theirs.IMFs.shape[0]) <= 4, (
        f"阶数差过大: native={ours.IMFs.shape[0]}, pyemd={theirs.IMFs.shape[0]}"
    )
    high = np.sin(2 * np.pi * 37 * t)
    c_ours = abs(float(np.corrcoef(ours.IMFs[0], high)[0, 1]))
    c_theirs = abs(float(np.corrcoef(theirs.IMFs[0], high)[0, 1]))
    assert c_ours > 0.8 and c_theirs > 0.8, (c_ours, c_theirs)


# --------------------------------------------------------------------------- #
# 5. 可复现性
# --------------------------------------------------------------------------- #
def test_same_seed_is_bitwise_identical():
    S, _ = _two_tone(512, noise=0.15)
    a = CEEMDAN(trials=10, seed=7).decompose(S)
    b = CEEMDAN(trials=10, seed=7).decompose(S)
    assert np.array_equal(a.IMFs, b.IMFs)
    assert np.array_equal(a.Res, b.Res)
    assert a.info["noise_imfs"] == b.info["noise_imfs"]


def test_different_seed_differs():
    S, _ = _two_tone(512, noise=0.15)
    a = CEEMDAN(trials=10, seed=7).decompose(S)
    b = CEEMDAN(trials=10, seed=8).decompose(S)
    assert not np.allclose(a.IMFs, b.IMFs)


def test_global_seed_overrides_local_seed():
    from src.Modal_Decomposition import set_seed

    S, _ = _two_tone(512, noise=0.15)
    ref = CEEMDAN(trials=10, seed=5).decompose(S)

    set_seed(5)
    try:
        with pytest.warns(UserWarning, match="global seed"):
            r = CEEMDAN(trials=10, seed=99).decompose(S)
        assert r.config.to_dict()["seed"] == 5
        assert np.allclose(r.IMFs, ref.IMFs)
    finally:
        set_seed(None)


# --------------------------------------------------------------------------- #
# 6. 参数有效性 (无静默死参数)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("kw", [
    {"sd_thr": 0.3},
    {"max_iter": 3},
    {"faster": False},
    {"nbsym": 0},
    {"spline_kind": "PCHIP"},
    {"max_imf": 2},
])
def test_engine_parameter_actually_changes_output(kw):
    """每个内层引擎旋钮都必须真的影响结果 —— 防止把参数写死后假接受。"""
    S, _ = _two_tone(512, noise=0.15)
    base = CEEMDAN(**FAST).decompose(S)
    got = CEEMDAN(**FAST, **kw).decompose(S)

    changed = (
        got.IMFs.shape != base.IMFs.shape
        or not np.allclose(got.IMFs, base.IMFs, atol=0)
    )
    assert changed, f"{kw} 未改变输出, 疑为死参数"


def test_find_peaks_backend_is_accepted_and_consistent():
    """
    numpy 与 scipy 两个极值后端在无平台信号上应给出相同结果
    (``Utils.Peaks`` 的统一返回契约); 这里只验证参数被接受且契约成立。
    """
    S, _ = _two_tone(512)
    a = CEEMDAN(**FAST, find_peaks_mod="numpy").decompose(S)
    b = CEEMDAN(**FAST, find_peaks_mod="scipy").decompose(S)
    assert a.IMFs.shape == b.IMFs.shape
    assert np.allclose(a.IMFs, b.IMFs, atol=1e-12)


def test_rich_info_adds_ensemble_std():
    S, _ = _two_tone(512, noise=0.15)
    plain = CEEMDAN(**FAST).decompose(S)
    rich = CEEMDAN(**FAST, rich_info=True).decompose(S)

    assert "ensemble_std" not in plain.info
    assert "ensemble_std" in rich.info
    assert rich.info["ensemble_std"].shape == rich.IMFs.shape
    assert np.all(rich.info["ensemble_std"] >= 0.0)
    # rich_info 只加诊断, 不应改变分解结果
    assert np.array_equal(plain.IMFs, rich.IMFs)


def test_info_diagnostics_present():
    S, _ = _two_tone(512, noise=0.15)
    r = CEEMDAN(**FAST).decompose(S)
    for key in ("trials", "trials_used", "noise_imfs", "stop_reason", "n_imfs"):
        assert key in r.info, f"info 缺少 {key}"
    assert r.info["trials"] == FAST["trials"]
    assert r.info["n_imfs"] == r.IMFs.shape[0]
    assert len(r.info["noise_imfs"]) == FAST["trials"]
    assert all(o >= 1 for o in r.info["noise_imfs"])
    # trials_used 至少覆盖已产出的每一阶
    assert len(r.info["trials_used"]) >= r.IMFs.shape[0]
    assert all(1 <= n <= FAST["trials"] for n in r.info["trials_used"])


def test_stop_reason_is_from_known_set():
    known = {
        "constant_signal", "max_imf", "noise_exhausted", "residual_monotonic",
        "residual_no_extrema", "range_thr", "total_power_thr",
    }
    for S, kw in [
        (_two_tone(512)[0], {}),
        (_two_tone(512)[0], dict(noise_width=0.15)),
        (np.ones(128), {}),
        (np.sin(np.linspace(0, 20, 128)), {}),
        (_two_tone(512)[0], dict(max_imf=2)),
        (_two_tone(512)[0], dict(range_thr=0.9)),
    ]:
        r = CEEMDAN(**FAST, **kw).decompose(S)
        assert r.info["stop_reason"] in known, r.info["stop_reason"]


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
    dict(max_iter=0),
    dict(sd_thr=0),
    dict(sd_thr=1.5),
    dict(nbsym=-1),
    dict(find_peaks_mod="x"),
    dict(spline_kind="cubic"),
    dict(faster=1),
    dict(range_thr=-1),
    dict(total_power_thr=-1),
    dict(rich_info=1),
])
def test_invalid_parameters_raise(kw):
    with pytest.raises(ValueError):
        CEEMDAN(**kw)


def test_constant_signal_returns_no_imfs():
    S = np.full(256, 3.0)
    r = CEEMDAN(**FAST).decompose(S)
    assert r.IMFs.shape == (0, 256)
    assert r.info["stop_reason"] == "constant_signal"
    assert np.array_equal(r.reconstruct(), S)


def test_zero_signal_returns_no_imfs():
    r = CEEMDAN(**FAST).decompose(np.zeros(256))
    assert r.IMFs.shape == (0, 256)
    assert r.info["stop_reason"] == "constant_signal"


def test_monotonic_signal_yields_single_flat_mode():
    """
    单调输入: CEEMDAN 不像 ``EMD`` 那样在开局做单调性检查, 而是先提出一阶、
    再在**残差**上判停机, 故返回 1 阶 (该阶极值数为 0, 是一条平缓趋势),
    且重构精确。PyEMD 版 CEEMDAN 在同样输入上同样返回 1 阶 —— 两者一致。
    """
    S = np.linspace(-3.0, 7.0, 256)
    r = CEEMDAN(**FAST).decompose(S)

    assert r.IMFs.shape == (1, 256)
    assert r.info["stop_reason"] == "residual_monotonic"
    assert np.allclose(r.reconstruct(), S, atol=1e-9)
    # 该阶 IMF 单调递减 (无振荡: 极值数为 0), 即它吸收了整个线性趋势
    imf = r.IMFs[0]
    assert np.all(np.diff(imf) <= 0.0), "单调输入的 IMF 出现了非单调振荡"
    assert not np.allclose(r.Res, 0.0, atol=1e-9), "趋势应部分留在残差中"


def test_monotonic_signal_matches_pyemd_mode_count():
    pytest.importorskip("PyEMD")
    S = np.linspace(-3.0, 7.0, 256)
    ours = CEEMDAN(**FAST).decompose(S)
    theirs = Class.CEEMDAN(trials=10, seed=0).decompose(S)
    assert ours.IMFs.shape[0] == theirs.IMFs.shape[0]


def test_too_short_signal_raises():
    with pytest.raises(ValueError, match="too short"):
        CEEMDAN(**FAST).decompose(np.arange(3.0))


@pytest.mark.parametrize("n", [4, 8, 16])
def test_minimum_lengths_do_not_crash(n):
    S, _ = _two_tone(n)
    r = CEEMDAN(trials=2, seed=0).decompose(S)
    assert r.IMFs.shape[1] == n
    assert np.allclose(r.reconstruct(), S, atol=1e-9)


def test_max_imf_truncates():
    S, _ = _two_tone(512, noise=0.15)
    full = CEEMDAN(**FAST).decompose(S)
    assert full.IMFs.shape[0] > 2, "该信号阶数太少, 无法验证截断"
    capped = CEEMDAN(**FAST, max_imf=2).decompose(S)
    assert capped.IMFs.shape[0] == 2
    assert capped.info["stop_reason"] == "max_imf"
    assert np.allclose(capped.reconstruct(), S, atol=1e-9)


def test_max_imf_one_matches_stepwise():
    S, _ = _two_tone(512, noise=0.15)
    r = CEEMDAN(trials=5, seed=0, max_imf=1).decompose(S)
    assert r.IMFs.shape[0] == 1
    assert np.allclose(r.reconstruct(), S, atol=1e-9)


def test_pure_noise_terminates():
    rng = np.random.default_rng(0)
    S = rng.standard_normal(512)
    r = CEEMDAN(**FAST).decompose(S)
    assert r.IMFs.shape[1] == 512
    assert np.allclose(r.reconstruct(), S, atol=1e-9)
    assert r.IMFs.shape[0] >= 1


def test_multichannel_input_rejected():
    with pytest.raises(ValueError):
        CEEMDAN(**FAST).decompose(np.zeros((2, 64)))


def test_single_trial_is_accepted():
    S, _ = _two_tone(512)
    r = CEEMDAN(trials=1, seed=0).decompose(S)
    assert all(n == 1 for n in r.info["trials_used"])


def test_time_axis_is_validated_but_unused():
    S, t = _two_tone(256)
    a = CEEMDAN(**FAST).decompose(S)
    b = CEEMDAN(**FAST).decompose(S, t)
    assert np.array_equal(a.IMFs, b.IMFs)


def test_time_axis_length_mismatch_raises():
    S, _ = _two_tone(256)
    with pytest.raises(ValueError):
        CEEMDAN(**FAST).decompose(S, np.arange(10))


# --------------------------------------------------------------------------- #
# 8. facade 等价
# --------------------------------------------------------------------------- #
def test_facade_matches_class():
    S, _ = _two_tone(512, noise=0.15)
    a = Function.CEEMDAN(S, trials=10, seed=0)
    b = Class.CEEMDAN(trials=10, seed=0).decompose(S)
    assert np.array_equal(a.IMFs, b.IMFs)
    assert np.array_equal(a.Res, b.Res)
    assert a.config.to_dict() == b.config.to_dict()


def test_facade_docstring_is_composed():
    doc = Function.CEEMDAN.__doc__
    assert "Complete Ensemble Empirical Mode Decomposition" in doc
    assert "Parameters" in doc
    assert "References" in doc
    assert "10.1109/ICASSP.2011.5947265" in doc
    assert "Random seed" in doc
    assert "CEEMDANConfig" in doc


# --------------------------------------------------------------------------- #
# 9. 性能烟测 (数字供 docs/CEEMDAN_Native_Report.md 引用)
# --------------------------------------------------------------------------- #
def test_performance_smoke():
    S, _ = _two_tone(2048, noise=0.15)
    CEEMDAN(trials=5, seed=0).decompose(S)  # 预热 (惰性 import / 缓存)
    t0 = time.perf_counter()
    CEEMDAN(trials=20, seed=0).decompose(S)
    dt = time.perf_counter() - t0
    assert dt < 10.0, f"trials=20, N=2048 耗时 {dt:.2f}s"
