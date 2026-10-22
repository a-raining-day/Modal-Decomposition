"""
``Modal_Decomposition.Utils.Envelope`` 模块测试 (一模块一测试)。

被测对象: ``envelope(S, method="Hilbert", fs=None, fc=None, cf=None,
spline=None, **kwargs)`` —— 从信号中提取**幅度包络**的对外工具, 四种策略:

* ``"Hilbert"`` (**默认**): ``|H[s(t)]|``, 无必填参数;
* ``"Lowpass"``: 全波整流 + Butterworth 低通, π/2 增益修正, 需 ``fs`` + ``cf``;
* ``"IQ"``: 正交解调, 需 ``fs`` + ``fc`` + ``cf``;
* ``"PeakInterpolation"``: 峰值点三次样条插值, 无必填参数。

本文件同时是**两个真实缺陷的回归锁**
------------------------------------
1. ``Check.not_numpy_memmap`` 判据曾写反 (``not isinstance(x, np.memmap)``
   就抛错) ⇒ 任何普通 ``ndarray`` 都被拒。现由
   ``test_default_call_accepts_plain_ndarray`` 锁死: 普通 ndarray + 默认参数
   必须成功。
2. ``method`` 转成 ``ENVELOPE_MODS`` 枚举后仍用字符串 ``case`` 分派 ⇒ 四种
   方法全部落到 ``"Unknown envelope method"``。现由
   ``test_all_four_methods_are_dispatchable`` / ``test_method_returns_same_shape_and_finite``
   锁死: 四种方法都必须真的返回包络。

实测行为 (以下断言全部按 ``.venv`` 实跑结果写, 非推测)
------------------------------------------------------
* 单位幅值正弦 (``n=256, t=linspace(0,1,256,endpoint=False), f=5``,
  ``fs=1.0, fc=0.1, cf=0.2``) 上的包络均值: ``Hilbert`` = 1.0 (逐点恒为 1),
  ``Lowpass`` ≈ 0.99995, ``IQ`` ≈ 1.268 (``fc`` 不是真载频, 拍频抬高),
  ``PeakInterpolation`` ≈ 0.99944。
* 四种方法对同一输入都返回 ``shape == S.shape``、全有限的 ``ndarray``;
  字符串与 Enum 成员两种写法**逐位相同**。
* ``Lowpass`` / ``IQ`` 三个参数全缺时抛
  ``ValueError: Either fc, fs or cf must be specified!``; 但该前置检查只比较
  **字符串** (在 Enum 转换之前), 所以传 ``ENVELOPE_MODS.Lowpass`` 且缺参时
  会漏过去、改由后续代码抛 ``TypeError`` —— 见 §6 的 characterization 用例。
* 长度 8 的短信号: ``Hilbert`` / ``PeakInterpolation`` 正常; ``Lowpass`` /
  ``IQ`` 抛 ``ValueError`` (``filtfilt`` 的 ``padlen=15``)。
* 常量信号: ``Hilbert`` = 1, ``Lowpass`` = π/2, ``IQ`` ≈ 2 (边缘振铃),
  ``PeakInterpolation`` 抛 ``ValueError`` (无峰值点)。
* 含 NaN: 前三种把 NaN 传播到整段, ``PeakInterpolation`` 抛 ``ValueError``。
* 白噪声上 ``PeakInterpolation`` 会因三次样条下冲出现**负值** (实测 min≈-1.5)
  —— 因此"非负"只对干净 AM/正弦断言, 不对噪声断言 (见 §7)。
"""

import inspect
import typing

import numpy as np
import pytest

import Modal_Decomposition.Utils as U
from Modal_Decomposition.Utils import ENVELOPE_MODS, ENVELOPE_MODS_ARGS, envelope
from Modal_Decomposition.Utils.Envelope import __all__ as ENVELOPE_ALL

# 与 ``docs/test/utils/Envelope_Engine_Report.md`` 的复现口径一致
FS, FC, CF = 1.0, 0.1, 0.2
PARAMS = dict(fs=FS, fc=FC, cf=CF)

METHODS = ("Hilbert", "Lowpass", "IQ", "PeakInterpolation")

# 字符串写法 -> Enum 成员 (注意 Peak 的**成员名**与**值**不同, 见 §1)
ENUM_OF = {
    "Hilbert": ENVELOPE_MODS.Hilbert,
    "Lowpass": ENVELOPE_MODS.Lowpass,
    "IQ": ENVELOPE_MODS.IQ,
    "PeakInterpolation": ENVELOPE_MODS.Peak,
}

# 干净信号上允许的负向过冲 (filtfilt 振铃 / 三次样条下冲, 实测量级 1e-5)
NEG_TOL = 1e-3


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _t(n: int = 256) -> np.ndarray:
    return np.linspace(0.0, 1.0, n, endpoint=False)


def _unit_sine(n: int = 256, f: float = 5.0) -> np.ndarray:
    """单位幅值正弦 —— 其解析包络恒为 1, 用作有闭式答案的基准。"""
    return np.sin(2 * np.pi * f * _t(n))


def _am(n: int = 256, fc: float = 20.0, fm: float = 2.0, depth: float = 0.5):
    """调幅信号 ``(1 + depth·cos(2πf_m t))·sin(2πf_c t)`` 及其理想包络。"""
    t = _t(n)
    env = 1.0 + depth * np.cos(2 * np.pi * fm * t)
    return env * np.sin(2 * np.pi * fc * t), env


def _env(S, method="Hilbert", **kw):
    """带默认参数的调用薄封装 (缺省走 PARAMS, 避免每个用例重复写)。"""
    p = dict(PARAMS)
    p.update(kw)
    return envelope(S, method, **p)


# --------------------------------------------------------------------------- #
# 1. 导出契约 (Utils 包本身没有 __all__, 按属性存在性断言)
# --------------------------------------------------------------------------- #
def test_utils_package_exposes_public_names():
    """三个公开符号都能从 ``Utils`` 顶层取到 (按属性存在性, 不靠 __all__)。"""
    for name in ("envelope", "ENVELOPE_MODS", "ENVELOPE_MODS_ARGS"):
        assert hasattr(U, name), f"Utils 顶层缺少 {name}"
    assert callable(U.envelope)
    assert isinstance(U.ENVELOPE_MODS, type) and issubclass(U.ENVELOPE_MODS, __import__("enum").Enum)


def test_utils_package_has_no_dunder_all_but_names_are_reachable():
    """``Utils`` 包**没有** ``__all__``; 若将来补上, 这三个名字必须包含在内。"""
    if hasattr(U, "__all__"):                      # 将来新增 __all__ 时的软校验
        assert {"envelope", "ENVELOPE_MODS", "ENVELOPE_MODS_ARGS"} <= set(U.__all__)
    assert getattr(U.envelope, "__module__", "") == "Modal_Decomposition.Utils.Envelope"


def test_module_all_is_complete_and_same_objects():
    """模块级 ``__all__`` 的 3 个名字都取得到, 且与包顶层是**同一对象**。"""
    import Modal_Decomposition.Utils.Envelope as E

    assert sorted(ENVELOPE_ALL) == ["ENVELOPE_MODS", "ENVELOPE_MODS_ARGS", "envelope"]
    for name in ENVELOPE_ALL:
        assert hasattr(E, name), f"{name} 在 __all__ 里但取不到"
        assert getattr(E, name) is getattr(U, name), f"{name} 在包顶层不是同一对象"


def test_literal_alias_matches_enum_values():
    """``ENVELOPE_MODS_ARGS`` 的 Literal 取值集合与枚举值集合一致。"""
    args = typing.get_args(ENVELOPE_MODS_ARGS)
    assert args == ("Hilbert", "Lowpass", "IQ", "PeakInterpolation")
    assert set(args) == {m.value for m in ENVELOPE_MODS}


def test_enum_members_and_naming_divergence():
    """成员名是 ``Peak``, 而它的**值**是 ``"PeakInterpolation"`` —— 字符串入口
    只有 ``"PeakInterpolation"`` 有效, ``"Peak"`` 会被判非法 (§5)。"""
    assert [m.name for m in ENVELOPE_MODS] == ["Hilbert", "Lowpass", "IQ", "Peak"]
    assert ENVELOPE_MODS.Peak.value == "PeakInterpolation"
    assert "Peak" not in {m.value for m in ENVELOPE_MODS}


def test_signature_defaults():
    """默认参数: ``method="Hilbert"``, ``fs/fc/cf/spline`` 全为 None。"""
    params = inspect.signature(envelope).parameters
    assert list(params) == ["S", "method", "fs", "fc", "cf", "spline", "kwargs"]
    assert params["method"].default == "Hilbert"
    for name in ("fs", "fc", "cf", "spline"):
        assert params[name].default is None


# --------------------------------------------------------------------------- #
# 2. 缺陷① 回归: 普通 ndarray 必须被接受
# --------------------------------------------------------------------------- #
def test_default_call_accepts_plain_ndarray():
    """回归: 普通 ndarray + 默认参数必须成功 (旧版 memmap 判据写反会直接抛)。"""
    S = np.sin(2 * np.pi * 5 * _t())
    env = envelope(S)                       # 不带 fs/fc/cf: Hilbert 无需先验
    assert isinstance(env, np.ndarray)
    assert env.shape == S.shape
    assert np.isfinite(env).all()
    # 单位幅值正弦的解析包络恒为 1
    assert np.allclose(env, 1.0, atol=1e-10)


def test_default_method_equals_explicit_hilbert():
    S = _unit_sine()
    assert np.array_equal(envelope(S), envelope(S, "Hilbert"))


def test_memmap_input_rejected(tmp_path):
    """memmap 仍被明确拒绝 (TypeError), 只是不再误伤普通 ndarray。"""
    path = tmp_path / "sig.dat"
    S = _unit_sine()
    mm = np.memmap(path, dtype=np.float64, mode="w+", shape=S.shape)
    mm[:] = S
    try:
        with pytest.raises(TypeError, match="memmap"):
            envelope(mm)
    finally:
        mm._mmap.close()


def test_non_ndarray_rejected():
    """list 不是 ndarray ⇒ TypeError (``Check.not_numpy_arr`` 的契约)。"""
    with pytest.raises(TypeError, match="np.ndarray"):
        envelope(list(_unit_sine()))
    with pytest.raises(TypeError, match="np.ndarray"):
        envelope(_unit_sine().tolist())


# --------------------------------------------------------------------------- #
# 3. 缺陷② 回归: 四种方法都必须真的跑起来
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("method", METHODS)
def test_method_returns_same_shape_and_finite(method):
    """四种 method 各返回 ``shape == S.shape`` 且全有限 —— 锁死 Enum 分派缺陷。"""
    S = _unit_sine()
    env = _env(S, method)
    assert isinstance(env, np.ndarray)
    assert env.shape == S.shape
    assert np.isfinite(env).all()


@pytest.mark.parametrize("method", METHODS)
def test_all_four_methods_are_dispatchable(method):
    """每种方法都真的算出非平凡结果 (均值为正), 不会落到 "Unknown envelope method"。"""
    env = _env(_unit_sine(), method)
    assert env.ndim == 1 and env.size == 256
    assert float(np.mean(env)) > 0.0
    assert float(np.max(env)) > 0.0


def test_methods_are_not_all_identical():
    """四种策略的实现路径确实不同 (至少 Hilbert 与 Lowpass 不逐位相同)。"""
    S = _unit_sine()
    assert not np.array_equal(_env(S, "Hilbert"), _env(S, "Lowpass"))
    assert not np.array_equal(_env(S, "Hilbert"), _env(S, "IQ"))


def test_hilbert_is_exact_on_unit_sine():
    """闭式解: ``|H[sin(2πft)]| ≡ 1`` (逐点, 非近似)。"""
    env = _env(_unit_sine(), "Hilbert")
    assert np.max(np.abs(env - 1.0)) < 1e-10


def test_hilbert_is_exact_on_clean_am():
    """干净 AM 上 Hilbert 包络与理想包络逐点一致 (实测误差 ~1e-15)。"""
    S, ideal = _am()
    env = _env(S, "Hilbert", fc=20.0)
    assert np.max(np.abs(env - ideal)) < 1e-12


# --------------------------------------------------------------------------- #
# 4. 字符串 与 Enum 成员两种写法等价
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("method", METHODS)
def test_string_and_enum_bitwise_identical(method):
    """``"Lowpass"`` 与 ``ENVELOPE_MODS.Lowpass`` 的结果**逐位相同**。"""
    S = _unit_sine()
    by_str = _env(S, method)
    by_enum = _env(S, ENUM_OF[method])
    assert np.array_equal(by_str, by_enum), f"{method}: 字符串与 Enum 结果不一致"


@pytest.mark.parametrize("method", METHODS)
def test_enum_equals_its_own_value_string(method):
    """Enum 成员与它自己的 ``.value`` 字符串等价 (Peak → "PeakInterpolation")。"""
    S = _unit_sine()
    member = ENUM_OF[method]
    assert np.array_equal(_env(S, member), _env(S, member.value))


def test_method_accepts_keyword_form():
    """``method=`` 关键字 + Enum 成员同样可用。"""
    S = _unit_sine()
    assert np.array_equal(
        envelope(S, method=ENVELOPE_MODS.Hilbert),
        envelope(S, method="Hilbert"),
    )


# --------------------------------------------------------------------------- #
# 5. 非法 method
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "bad",
    ["bogus", "Peak", "Lowpass ", "hilbert", "HILBERT", "", 5, 3.14, None, ("Hilbert",)],
)
def test_unknown_method_raises_value_error(bad):
    """非法 ``method`` 一律 ``ValueError``, 消息含 "Unknown envelope method"。

    注意 ``"Peak"`` 也在非法之列: 枚举**成员名**是 Peak, 但字符串入口认的是
    **值** ``"PeakInterpolation"`` (见 §1)。
    """
    with pytest.raises(ValueError, match="Unknown envelope method"):
        envelope(_unit_sine(), bad)


# --------------------------------------------------------------------------- #
# 6. 参数缺失: Lowpass / IQ 需要先验频率
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("method", ["Lowpass", "IQ"])
def test_missing_params_raise_with_either_message(method):
    """``Lowpass`` / ``IQ`` 缺 ``fs/fc/cf`` 时抛 ValueError, 消息含 "Either"。"""
    with pytest.raises(ValueError, match="Either"):
        envelope(_unit_sine(), method)
    with pytest.raises(ValueError, match="Either fc, fs or cf must be specified"):
        envelope(_unit_sine(), method, fs=None, fc=None, cf=None)


@pytest.mark.parametrize("method", METHODS)
def test_no_params_needed_by_hilbert_and_peak(method):
    """``Hilbert`` / ``PeakInterpolation`` 无需任何频率参数即可出结果。"""
    S = _unit_sine()
    if method in ("Lowpass", "IQ"):
        with pytest.raises(ValueError, match="Either"):
            envelope(S, method)
    else:
        assert np.isfinite(envelope(S, method)).all()


@pytest.mark.parametrize("method", ["Lowpass", "IQ"])
def test_enum_form_misses_friendly_check(method):
    """characterization: 前置检查只比较**字符串** (在 Enum 转换之前), 因此传
    Enum 成员且缺参时会漏过 "Either" 提示, 改由后续代码抛 ``TypeError``。
    修好前置检查后本用例会 XPASS/失败, 请同步更新 §6 与报告。"""
    with pytest.raises((TypeError, ValueError)):
        envelope(_unit_sine(), ENUM_OF[method])


@pytest.mark.parametrize(
    "method,kw",
    [
        ("Lowpass", dict(fs=FS)),            # 只有 fs: 缺 cf
        ("IQ", dict(fs=FS, fc=FC)),          # 缺 cf
        ("IQ", dict(fs=FS, cf=CF)),          # 缺 fc
        ("IQ", dict(fc=FC, cf=CF)),          # 缺 fs
    ],
)
def test_partial_params_still_raise(method, kw):
    """characterization: 只给"三个中的一个"并不够 —— ``Lowpass`` 需 fs+cf,
    ``IQ`` 需 fs+fc+cf; 缺的会在运算时抛 ``TypeError`` (不是静默的错误结果)。"""
    with pytest.raises((TypeError, ValueError)):
        envelope(_unit_sine(), method, **kw)


# --------------------------------------------------------------------------- #
# 7. 数值合理性
# --------------------------------------------------------------------------- #
def test_hilbert_mean_in_wide_range():
    """单位幅值正弦下 Hilbert 包络均值落在宽松区间 0.5–5。"""
    mean = float(np.mean(_env(_unit_sine(), "Hilbert")))
    assert 0.5 <= mean <= 5.0
    assert mean == pytest.approx(1.0, abs=1e-8)


@pytest.mark.parametrize("method", METHODS)
def test_envelopes_non_negative_on_clean_am(method):
    """干净 AM 上四种方法的包络都非负 (允许 filtfilt/样条的微小负向过冲)。"""
    S, _ = _am()
    env = _env(S, method, fc=20.0)
    assert float(np.min(env)) >= -NEG_TOL, f"{method}: 出现明显负包络"


@pytest.mark.parametrize("method", ["Hilbert", "IQ"])
@pytest.mark.parametrize("kind", ["sine", "am", "noise"])
def test_abs_based_methods_never_negative(method, kind):
    """``Hilbert = |z|`` 与 ``IQ = 2·sqrt(i²+q²)`` 都是非负构造, 严格 ≥ 0。"""
    if kind == "sine":
        S = _unit_sine()
    elif kind == "am":
        S, _ = _am()
    else:
        S = np.random.default_rng(0).standard_normal(256)
    env = _env(S, method)
    assert float(np.min(env)) >= 0.0


@pytest.mark.parametrize("method", ["Hilbert", "Lowpass", "PeakInterpolation"])
def test_envelope_mean_close_to_unit_amplitude(method):
    """单位幅值正弦上, 无需真载频的三种方法均值都接近 1 (±5%)。

    ``IQ`` 不在此列: ``fc=0.1`` 不是信号的 5 Hz 载频, 拍频把均值抬到 ~1.27,
    正是模块文档"只在拿到真实载频时准确"的说法 —— 单独见
    ``test_iq_tracks_amplitude_with_true_carrier``。
    """
    mean = float(np.mean(_env(_unit_sine(), method)))
    assert mean == pytest.approx(1.0, rel=0.05), f"{method}: mean={mean}"


def test_iq_tracks_amplitude_with_true_carrier():
    """IQ 在 ``fc`` 取真载频时均值接近 1 (文档: 只有给出真实载频才准)。"""
    S = _unit_sine(f=5.0)
    env = envelope(S, "IQ", fs=FS, fc=5.0, cf=CF)
    assert env.shape == S.shape and np.isfinite(env).all()
    assert float(np.mean(env)) == pytest.approx(1.0, rel=0.35)


def test_peak_interpolation_tracks_am_envelope():
    """PeakInterpolation 在干净 AM 上贴合理想包络 (实测平均绝对误差 ~0.012)。"""
    S, ideal = _am()
    env = _env(S, "PeakInterpolation", fc=20.0)
    assert float(np.mean(np.abs(env - ideal))) < 0.05


def test_lowpass_has_pi_over_2_gain_correction():
    """Lowpass 的 π/2 增益修正 (值为 1 的常量输入应得 π/2)。"""
    env = _env(np.ones(256), "Lowpass")
    assert np.allclose(env, np.pi / 2.0, atol=1e-8)


# --------------------------------------------------------------------------- #
# 8. 边界: 短信号 / 常量 / NaN / 空 / 1 点 / 2-D / 复数
# --------------------------------------------------------------------------- #
def test_short_signal_len8():
    """长度 8: Hilbert / PeakInterpolation 正常; Lowpass / IQ 因 filtfilt 的
    ``padlen=15`` 抛 ValueError (实测行为, 不是崩溃)。"""
    S = np.sin(2 * np.pi * 5 * _t(8))
    for method in ("Hilbert", "PeakInterpolation"):
        env = _env(S, method)
        assert env.shape == (8,) and np.isfinite(env).all()
    assert np.allclose(_env(S, "Hilbert"), 1.0, atol=1e-10)

    for method in ("Lowpass", "IQ"):
        with pytest.raises(ValueError, match="padlen"):
            _env(S, method)


def test_constant_signal():
    """常量信号: Hilbert = 1, Lowpass = π/2, IQ 有限且有界 (边缘振铃)。"""
    C = np.ones(256)
    assert np.allclose(_env(C, "Hilbert"), 1.0, atol=1e-12)
    assert np.allclose(_env(C, "Lowpass"), np.pi / 2.0, atol=1e-8)
    iq = _env(C, "IQ")
    assert iq.shape == C.shape and np.isfinite(iq).all()
    assert float(np.max(iq)) < 3.0


def test_constant_signal_peak_raises():
    """常量信号没有极值点 ⇒ PeakInterpolation 抛 ValueError。"""
    with pytest.raises(ValueError, match="at least 2 elements"):
        _env(np.ones(256), "PeakInterpolation")


@pytest.mark.parametrize("method", ["Hilbert", "Lowpass", "IQ"])
def test_nan_propagates(method):
    """含 NaN 的输入: 前三种方法把 NaN 传播到整段 (不静默丢弃)。"""
    S = _unit_sine()
    S = S.copy()
    S[10] = np.nan
    env = _env(S, method)
    assert env.shape == S.shape
    assert np.isnan(env).any()
    assert not np.isfinite(env).all()


def test_nan_rejected_by_peak_interpolation():
    """PeakInterpolation 对 NaN 抛 ValueError (而不是产出垃圾)。"""
    S = _unit_sine()
    S = S.copy()
    S[10] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        _env(S, "PeakInterpolation")


@pytest.mark.parametrize("method", ["Hilbert", "PeakInterpolation"])
def test_empty_input_raises(method):
    """空输入抛明确的 ValueError (不是 IndexError/静默空数组)。"""
    with pytest.raises(ValueError):
        envelope(np.zeros(0), method)


def test_single_sample():
    """长度 1: Hilbert 返回 ``[0.]``; PeakInterpolation 需要 ≥2 个峰值点。"""
    assert np.array_equal(envelope(np.array([0.0])), np.array([0.0]))
    with pytest.raises(ValueError, match="at least 2 elements"):
        envelope(np.array([0.0]), "PeakInterpolation")


def test_short_len3_peak_raises():
    """长度 3 的正弦只有 2 个极值点, 仍不足以插值 ⇒ ValueError。"""
    S = np.sin(2 * np.pi * 5 * _t(3))
    with pytest.raises(ValueError, match="at least 2 elements"):
        envelope(S, "PeakInterpolation")
    assert envelope(S).shape == (3,)


def test_2d_input_passthrough():
    """characterization: 默认 Hilbert 走 scipy.signal.hilbert, 2-D 输入沿末轴
    逐行处理并保持形状 (不是文档承诺的 1-D 契约, 但当前行为如此)。"""
    S = np.stack([_unit_sine(), 0.5 * _unit_sine()])
    env = envelope(S)
    assert env.shape == S.shape
    assert np.allclose(env[0], 1.0, atol=1e-10)
    assert np.allclose(env[1], 0.5, atol=1e-10)


def test_complex_input_raises():
    with pytest.raises(ValueError, match="must be real"):
        envelope(_unit_sine() + 0j)


# --------------------------------------------------------------------------- #
# 9. 确定性与"不改写调用方数组"
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("method", METHODS)
def test_deterministic(method):
    """同输入两次调用逐位相同。"""
    S = _unit_sine()
    assert np.array_equal(_env(S, method), _env(S, method))


@pytest.mark.parametrize("method", METHODS)
def test_does_not_mutate_input(method):
    """四种方法都不得改写调用方数组。"""
    S = _unit_sine()
    before = S.copy()
    _env(S, method)
    assert np.array_equal(S, before)


def test_output_does_not_alias_input():
    """返回的是新数组, 不与输入共享内存。"""
    S = _unit_sine()
    assert not np.shares_memory(envelope(S), S)


@pytest.mark.parametrize("method", METHODS)
def test_dtype_contract(method):
    """整数输入提升为 float64; float32 输入不崩且保持有限。"""
    S = (_unit_sine() * 100).astype(np.int32)
    env = _env(S, method)
    assert env.dtype == np.float64
    assert np.isfinite(env).all()

    env32 = _env(_unit_sine().astype(np.float32), method)
    assert env32.shape == (256,)
    assert np.isfinite(env32).all()
