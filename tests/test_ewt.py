"""
``EWT`` (自研实现, 注册键 ``"EWT"``) 的核心契约与各项修复的回归测试。

覆盖:

* ``band_factor="auto"`` —— 按边界间距反推 γ, 滤波器组必须是紧框架
  (``Σ_i H_i² ≡ 1``), 且显式 float 仍走固定宽度 (旧行为);
* 最高带拉平到 Nyquist —— 全轴 (含 Nyquist 频点) 都是紧框架;
* ``envelope`` 边界分支 —— 不再退化为 1 个频带, 且边界落在真实分量附近;
* ``Smooth`` 频域预处理分支 —— 默认关闭, `smooth_width=1` 时退化为恒等;
* ``reconstruct()`` 精确性 (各分支组合下同样成立)。

对应报告: ``docs/EWT_Native_Report.md``。
"""

import numpy as np
import pytest

import src.Modal_Decomposition as MD
from src.Modal_Decomposition.EWT import EWT

FS = 1000.0
BOUNDARY_MODS = ("maximum", "max-min", "scale-space", "envelope")


def tones_signal(N, tones, noise=0.05, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(N) / FS
    x = sum(a * np.sin(2 * np.pi * f * t)
            for a, f in zip([1.0 / (i + 1) for i in range(len(tones))], tones))
    return x + (noise * rng.standard_normal(N) if noise else 0.0)


@pytest.fixture
def multitones():
    return tones_signal(2048, (50, 150, 320))


@pytest.fixture
def two_tones():
    return tones_signal(2048, (50, 70), noise=0.02)


def band_sum(mfb):
    """滤波器组的幅度和 / 能量和 (逐频点)。"""
    mfb = np.asarray(mfb, dtype=np.float64)
    return mfb.sum(axis=0), (mfb ** 2).sum(axis=0)


def axis_freq(mfb, fs=FS):
    """由滤波器组的频率轴长度反推频率坐标 (mirror=True 时轴长为 2N)。"""
    n_bins = np.asarray(mfb).shape[1]
    return np.fft.rfftfreq(2 * (n_bins - 1), 1.0 / fs)


# --------------------------------------------------------------------------- #
# band_factor="auto": 紧框架
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("mod", BOUNDARY_MODS)
def test_auto_band_factor_is_tight_frame(multitones, mod):
    """
    auto γ 下任意边界策略都应给出能量型紧框架 ``Σ H² ≡ 1``:

    * 内部过渡带严格互补;
    * 最高带的滤波上边界被抬到 ``Nyquist/(1−γ)``, 于是 Nyquist 端不再有"没有
      对偶带补偿"的半条过渡带 —— **全轴**(含 Nyquist 频点)都满足 ``ΣH² = 1``。
    """
    r = EWT(num_imfs=5, boundary_mod=mod).decompose(multitones, fs=FS)
    mfb = np.asarray(r.info["mfb"], dtype=np.float64)
    freq = axis_freq(mfb)
    gamma = r.info["band_factor"]
    b_last = float(np.asarray(r.info["boundaries"], dtype=np.float64)[-2])
    s, s2 = band_sum(mfb)

    assert r.info["band_factor_auto"] is True
    assert 0.0 < gamma < 1.0
    assert np.max(np.abs(s2 - 1.0)) < 1e-12, "bank is not a tight frame on the whole axis"

    interior = freq <= (1.0 - gamma) * b_last
    assert interior.sum() > 0.2 * freq.size, "interior should be a substantial part of the axis"

    # 内部过渡带中点: 互补对 => Σ H = √2; Nyquist 端现在是通带 => Σ H = 1
    assert np.isclose(s[interior].max(), np.sqrt(2.0), atol=1e-9)
    assert np.isclose(s[-1], 1.0, atol=1e-9)


def test_top_band_reaches_nyquist(multitones):
    """最高带在 Nyquist 处必须是满增益 (修好前的值是 1/√2)。"""
    r = EWT(num_imfs=5).decompose(multitones, fs=FS)
    mfb = np.asarray(r.info["mfb"], dtype=np.float64)
    assert np.isclose(mfb[-1, -1], 1.0, atol=1e-12)
    # info["boundaries"] 仍是检测到的真实边界 (末项 = Nyquist), 未被滤波副本污染
    assert np.isclose(np.asarray(r.info["boundaries"], dtype=np.float64)[-1], FS / 2.0)


@pytest.mark.parametrize("mod", BOUNDARY_MODS)
def test_explicit_band_factor_unchanged(multitones, mod):
    """显式 γ=0.15 保持旧行为: 不报 auto, 且边界聚拢时 Σ H² 可超过 1。"""
    r = EWT(num_imfs=5, band_factor=0.15, boundary_mod=mod).decompose(multitones, fs=FS)
    assert r.info["band_factor_auto"] is False
    assert r.info["band_factor"] == pytest.approx(0.15)
    _, s2 = band_sum(r.info["mfb"])
    assert s2.max() >= 1.0 - 1e-12


def test_auto_reduces_band_sum_error(multitones):
    """同一组边界: auto γ 的带和误差不大于固定 0.15 (实测 7× 改善)。"""
    def err(bf):
        r = EWT(num_imfs=5, band_factor=bf).decompose(multitones, fs=FS)
        return np.linalg.norm(multitones - r.IMFs.sum(axis=0)) / np.linalg.norm(multitones)

    assert err("auto") < err(0.15)


def test_auto_frame_defect_smaller_than_fixed(multitones):
    """
    紧框架缺陷 (内部 |ΣH²−1| 的总量) 在 auto γ 下远小于固定 0.15 ——
    后者在边界聚拢时让相邻过渡带叠加, 同一频段被重复计入。
    """
    def defect(bf):
        r = EWT(num_imfs=5, band_factor=bf).decompose(multitones, fs=FS)
        _, s2 = band_sum(r.info["mfb"])
        freq = axis_freq(r.info["mfb"])
        gamma = r.info["band_factor"]
        b_last = float(np.asarray(r.info["boundaries"], dtype=np.float64)[-2])
        interior = freq <= (1.0 - gamma) * b_last
        return float(np.abs(s2[interior] - 1.0).sum()), float((s2 - 1.0).max())

    auto_sum, auto_max = defect("auto")
    fixed_sum, fixed_max = defect(0.15)
    assert auto_sum < 1e-12 < fixed_sum
    assert auto_max < fixed_max


def test_auto_band_factor_reproducible_from_config(multitones):
    """config 记录的是**实际使用**的 γ, 用它回灌可复现同一结果。"""
    r = EWT(num_imfs=5).decompose(multitones, fs=FS)
    gamma = r.config.band_factor
    assert gamma == pytest.approx(r.info["band_factor"])
    r2 = EWT(num_imfs=5, band_factor=gamma).decompose(multitones, fs=FS)
    assert np.allclose(r2.IMFs, r.IMFs, rtol=0, atol=1e-12)


def test_auto_band_factor_single_band_falls_back(multitones):
    """num_imfs=1 (无内部边界) 时退回固定 0.15, 不因 Nyquist 约束取到极端值。"""
    r = EWT(num_imfs=1).decompose(multitones, fs=FS)
    assert r.info["band_factor"] == pytest.approx(0.15)


@pytest.mark.parametrize("bad", [None, "nope", 0.0, 1.0, 1.5, -0.2])
def test_band_factor_validation(bad):
    with pytest.raises((ValueError, TypeError)):
        EWT(num_imfs=5, band_factor=bad)


# --------------------------------------------------------------------------- #
# envelope 分支: 不退化
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", ["multitones", "two_tones"])
def test_envelope_not_degenerate(name, request):
    """envelope 分支必须给出 >= 2 个频带 (修复前恒为 1)。"""
    S = request.getfixturevalue(name)
    r = EWT(num_imfs=5, boundary_mod="envelope").decompose(S, fs=FS)
    assert r.info["n_bands"] >= 2
    assert len(r.info["boundaries"]) == r.info["n_bands"] + 1


def test_envelope_separates_two_tones(two_tones):
    """50/70 Hz 双音: envelope 的边界必须把两个音分到不同频带。"""
    r = EWT(num_imfs=5, boundary_mod="envelope").decompose(two_tones, fs=FS)
    bnd = np.asarray(r.info["boundaries"], dtype=np.float64)
    assert np.any((bnd > 50.0) & (bnd < 70.0)), f"tones not separated: {bnd}"


def test_envelope_boundaries_not_stuck_at_low_frequency(multitones):
    """边界不得聚在低频小片 (修复前全部落在 3–13 Hz 的噪声平台里)。"""
    r = EWT(num_imfs=5, boundary_mod="envelope").decompose(multitones, fs=FS)
    bnd = np.asarray(r.info["boundaries"], dtype=np.float64)
    interior = bnd[(bnd > 0.0) & (bnd < FS / 2.0)]
    assert interior.size >= 2
    # 至少有一条内部边界落在 150 Hz 以上 (三个音的中-高频区)
    assert interior.max() > 150.0


def test_envelope_band_sum_error_is_sane(multitones):
    """envelope 的带和误差应在 0.1 以内 (修复前退化到 1 带时也小, 故与带数一起看)。"""
    r = EWT(num_imfs=5, boundary_mod="envelope").decompose(multitones, fs=FS)
    err = np.linalg.norm(multitones - r.IMFs.sum(axis=0)) / np.linalg.norm(multitones)
    assert err < 0.1
    assert r.info["n_bands"] >= 4


# --------------------------------------------------------------------------- #
# 重构契约 (auto γ 下)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("mod", BOUNDARY_MODS)
@pytest.mark.parametrize("mirror", [True, False])
def test_reconstruct_exact(multitones, mod, mirror):
    r = EWT(num_imfs=5, boundary_mod=mod, mirror=mirror).decompose(multitones, fs=FS)
    back = r.reconstruct() if hasattr(r, "reconstruct") else (r.IMFs.sum(axis=0) + r.Res)
    assert np.allclose(np.asarray(back), multitones, rtol=0, atol=1e-12)


def test_facade_forwards_fs(multitones):
    """
    ``Function.EWT(S, fs=...)`` 必须把 ``fs`` 交给 ``decompose`` (而不是被构造
    函数的 ``**kwargs`` 吞掉后静默退回 ``fs=1.0``)。
    """
    a = EWT(num_imfs=5, mirror=False).decompose(multitones, fs=FS)
    b = MD.Function.EWT(multitones, num_imfs=5, mirror=False, fs=FS)
    assert np.allclose(a.IMFs, b.IMFs, rtol=0, atol=0)
    # fs 确实生效: 不传时按 fs=1.0 建频率轴, 边界数值随之相差 1000 倍
    c = MD.Function.EWT(multitones, num_imfs=5, mirror=False)
    assert not np.allclose(a.info["boundaries"], c.info["boundaries"])


def test_facade_and_class_share_the_native_entry():
    """注册键 "EWT" 现在就是自研实现 (Class 与 Function 同源)。"""
    assert MD.Class.EWT is EWT
    assert MD.Class.EWT.__module__.split(".")[-1] == "EWT"      # 不是 EWTpy / 旧包装
    assert MD.Function.EWT.__qualname__ == "Function.EWT"


@pytest.mark.parametrize("mod", BOUNDARY_MODS)
def test_reconstruct_exact_pre_deal(multitones, mod):
    """预处理分支下 Res 吸收全部差异, reconstruct 仍等于原始输入。"""
    for pre in ("no-dc", "no-trend", "window", "Slepian-Optimize", "Smooth"):
        r = EWT(num_imfs=5, boundary_mod=mod, pre_deal=pre).decompose(multitones, fs=FS)
        back = r.reconstruct() if hasattr(r, "reconstruct") else (r.IMFs.sum(axis=0) + r.Res)
        assert np.allclose(np.asarray(back), multitones, rtol=0, atol=1e-12)


# --------------------------------------------------------------------------- #
# Smooth: 频域预处理分支 (只作用于边界检测用的幅度谱, 默认关闭)
# --------------------------------------------------------------------------- #
def test_smooth_off_by_default(multitones):
    """默认不平滑: info["pre_deal"] 为空, 与显式 pre_deal=None 逐位一致。"""
    a = EWT(num_imfs=5).decompose(multitones, fs=FS)
    b = EWT(num_imfs=5, pre_deal=None).decompose(multitones, fs=FS)
    assert a.info["pre_deal"] == ()
    assert b.info["pre_deal"] == ()
    assert np.allclose(a.IMFs, b.IMFs, rtol=0, atol=0)
    assert np.allclose(a.info["boundaries"], b.info["boundaries"], rtol=0, atol=0)


def test_smooth_width_one_is_identity(multitones):
    """smooth_width=1 ⇒ 窗长 1, 平滑退化为恒等: 边界与模态都不变。"""
    a = EWT(num_imfs=5).decompose(multitones, fs=FS)
    b = EWT(num_imfs=5, pre_deal="Smooth", smooth_width=1).decompose(multitones, fs=FS)
    assert b.info["pre_deal"] == ("Smooth",)
    assert np.allclose(a.info["boundaries"], b.info["boundaries"], rtol=0, atol=0)
    assert np.allclose(a.IMFs, b.IMFs, rtol=0, atol=0)


def test_smooth_moves_boundary_towards_ideal(multitones):
    """
    平滑的意义: multitone 的 50/150 Hz 理想边界是 100 Hz, 不平滑时被双峰
    结构拉偏, 平滑后贴到 1 Hz 以内 (实测 102.8 → 99.9 Hz)。
    """
    def boundary_near(bnd, target):
        b = np.asarray(bnd, dtype=np.float64)
        b = b[(b > 0.0) & (b < FS / 2.0)]
        return float(b[int(np.argmin(np.abs(b - target)))])

    raw = EWT(num_imfs=5).decompose(multitones, fs=FS)
    smooth = EWT(num_imfs=5, pre_deal="Smooth").decompose(multitones, fs=FS)
    assert not np.allclose(raw.info["boundaries"], smooth.info["boundaries"])
    d_raw = abs(boundary_near(raw.info["boundaries"], 100.0) - 100.0)
    d_smooth = abs(boundary_near(smooth.info["boundaries"], 100.0) - 100.0)
    assert d_smooth < d_raw
    assert d_smooth < 1.0


def test_smooth_gaussian_and_box_differ(multitones):
    """两种窗应给出不同结果 (参数真的生效)。"""
    box = EWT(num_imfs=5, pre_deal="Smooth", smooth_kind="box", smooth_width=5)
    gau = EWT(num_imfs=5, pre_deal="Smooth", smooth_kind="gaussian", smooth_width=5)
    a = box.decompose(multitones, fs=FS)
    b = gau.decompose(multitones, fs=FS)
    assert not np.allclose(a.info["boundaries"], b.info["boundaries"])


def test_smooth_canonical_order_with_slepian(multitones):
    """频域分支按 canonical 顺序施加: Slepian-Optimize → Smooth (与传入顺序无关)。"""
    a = EWT(num_imfs=5, pre_deal=["Smooth", "Slepian-Optimize"]).decompose(multitones, fs=FS)
    b = EWT(num_imfs=5, pre_deal=["Slepian-Optimize", "Smooth"]).decompose(multitones, fs=FS)
    assert a.info["pre_deal"] == ("Slepian-Optimize", "Smooth")
    assert np.allclose(a.IMFs, b.IMFs, rtol=0, atol=1e-15)


@pytest.mark.parametrize("mod", BOUNDARY_MODS)
def test_smooth_composes_with_boundary_mods(multitones, mod):
    """平滑与 4 套边界策略自由组合, 且不动重构契约。"""
    r = EWT(num_imfs=5, boundary_mod=mod, pre_deal="Smooth").decompose(multitones, fs=FS)
    back = r.reconstruct() if hasattr(r, "reconstruct") else (r.IMFs.sum(axis=0) + r.Res)
    assert np.allclose(np.asarray(back), multitones, rtol=0, atol=1e-12)
    assert r.info["n_bands"] >= 1


def test_smooth_validation():
    with pytest.raises(ValueError):
        EWT(num_imfs=5, pre_deal="Smooth", smooth_kind="median")
    with pytest.raises(ValueError):
        EWT(num_imfs=5, pre_deal="Smooth", smooth_width=0)
    with pytest.raises(ValueError):
        EWT(num_imfs=5, pre_deal="Smooth", smooth_width=0.5)


def test_smooth_unknown_branch_name_still_rejected(multitones):
    """未登记的分支名仍然报错 (新增分支没有放宽名字校验)。"""
    with pytest.raises(ValueError):
        EWT(num_imfs=5, pre_deal="Smoothh")
