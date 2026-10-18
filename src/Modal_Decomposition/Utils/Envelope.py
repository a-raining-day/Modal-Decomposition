"""
Envelope extraction (包络提取工具)。

与**模态分解迭代中的包络**区分开: 本模块是从信号中提取**幅度包络**的对外
工具 (AM/调制分析用), 提供四种策略:

* ``"Hilbert"`` (默认): 解析信号幅度 ``|H[s(t)]|`` —— 仅当信号近似为单分量
  AM/载波调制时包络正确 (噪声无偏; 多分量/宽带上会出拍频误差);
* ``"Lowpass"`` / ``"IQ"``: 全波整流 + 低通 / 正交解调 (需 fs、cf 等先验);
* ``"PeakInterpolation"``: 对信号峰值点做**样条插值**得到包络 (Spline 求包络,
  需 fs、fc)。

模态分解 (EMD / LMD 等) 筛分迭代里的"上/下包络"与"局部均值"是另一套概念 ——
它们是对**极值点**做样条插值 (见 ``Utils.Spline``) 得到的, 由分解方法在内部
实现, 不经过本工具模块。两者不要混用: 分解迭代中不用解析 Hilbert 包络作
筛分包络; 本模块的 Hilbert 模式只用于"确实需要解析幅度"的场合 (如对已分解
的单分量 IMF 做 AM 分析 / 瞬时频率)。

Layering (与 Utils 其余工具一致): 本模块自身不接触缓存; ``Utils.get_envelope()``
首次访问时惰性 import 并以键 ``"Modal_Decomposition.Utils.Envelope"`` 注册进
进程级 import 缓存。
"""

import numpy as np
from enum import Enum
from typing import Literal, Callable

from ..Base import Cache
from .Check import Check
from .Peaks import find_peaks
from .Hilbert import hilbert

__all__ = ["envelope", "ENVELOPE_MODS", "ENVELOPE_MODS_ARGS"]

ENVELOPE_MODS_ARGS = Literal["Hilbert", "Lowpass", "IQ", "PeakInterpolation"]
class ENVELOPE_MODS(Enum):
    Hilbert = "Hilbert"
    Lowpass = "Lowpass"
    IQ = "IQ"
    Peak = "PeakInterpolation"


def envelope(S: np.ndarray, method: ENVELOPE_MODS_ARGS | ENVELOPE_MODS = "Hilbert", fs: float = None, fc: float = None, cf: float = None, spline: Callable = None, **kwargs) -> np.ndarray:
    """
    Compute the amplitude envelope of a 1-D signal.

    :param S: input signal (np.ndarray, 1-D)
    :param method: envelope strategy; default "Hilbert" (see the report below)
    :param fs: sampling rate (required by Lowpass / IQ / PeakInterpolation)
    :param fc: carrier frequency (required by IQ / PeakInterpolation)
    :param cf: cut-off frequency (required by Lowpass / IQ)
    :param spline: spline function (required by PeakInterpolation)
    :return: envelope array with the same shape as S

    Default-mode report (tests/comparison/verify_envelope_modes.py,
    4 synthetic AM cases, fs=10 kHz, fc=1 kHz, cf=200 Hz; RMSE vs ground
    truth, edges trimmed):

    +-------------------+--------+------+-------+----------+----------------------------------+
    | method            | RMSE   | corr | ms/cal| params   | notes                            |
    +===================+========+======+=======+==========+==================================+
    | Hilbert (default) | ~0.013 | .999 | ~0.06 | none     | exact on clean AM (RMSE ~1e-15); |
    |                   |        |      |       |          | unbiased on noise (RMSE ~ noise |
    |                   |        |      |       |          | std); the only parameter-free    |
    |                   |        |      |       |          | mode -> chosen as the default.   |
    +-------------------+--------+------+-------+----------+----------------------------------+
    | Lowpass           | ~0.059 | 1.00 | ~0.37 | fs, cf   | pi/2 gain corrected (2026-09-06, |
    |                   |        |      |       |          | was ~0.39); per-case RMSE        |
    |                   |        |      |       |          | 0.017-0.174: best when cf >>     |
    |                   |        |      |       |          | modulation bandwidth (fm ~ cf    |
    |                   |        |      |       |          | attenuates the envelope); beats  |
    |                   |        |      |       |          | Hilbert on noisy AM (0.026 vs    |
    |                   |        |      |       |          | 0.051).                          |
    +-------------------+--------+------+-------+----------+----------------------------------+
    | IQ                | ~0.050 | 1.00 | ~0.33 | fs,fc,cf | accurate only with the true      |
    |                   |        |      |       |          | carrier frequency fc; otherwise  |
    |                   |        |      |       |          | beat terms bias the envelope.    |
    +-------------------+--------+------+-------+----------+----------------------------------+
    | PeakInterpolation | ~0.022 | .997 | ~0.21 | none     | most accurate on clean AM        |
    |                   |        |      |       |          | (RMSE ~1e-15); degrades under    |
    |                   |        |      |       |          | noise (spurious peaks);          |
    |                   |        |      |       |          | meaningless for non-AM signals.  |
    +-------------------+--------+------+-------+----------+----------------------------------+

    "Hilbert" is the default: zero required parameters, exact on clean AM and
    noise-unbiased; PeakInterpolation/IQ/Lowpass are only preferable when
    the carrier/sampling parameters are known and exact.
    """
    if method in ["Lowpass", "IQ"]:
        if (fc is None) and (fs is None) and (cf is None):
            raise ValueError("Either fc, fs or cf must be specified!")

    try:
        method = ENVELOPE_MODS(method)
    except ValueError:
        raise ValueError(f"Unknown envelope method: {method}")

    Check.not_numpy_arr(S)
    Check.not_numpy_memmap(S)

    env: np.ndarray = None

    _SCIPY_SIGNAL = Cache.import_module("scipy.signal")
    _SCIPY_INTERAPT = Cache.import_module("scipy.interpolate")

    peak_mod = kwargs.get("peak_mod", "numpy")
    hilbert_mod = kwargs.get("hilbert_mod", "scipy")

    peak = lambda x, **kwargs: find_peaks(x, mod="max", backend=peak_mod, **kwargs)

    def _lowpass(data, cutoff, order=4):
        nyq = 0.5 * fs
        b, a = _SCIPY_SIGNAL.butter(N=order, Wn=(cutoff / nyq), btype='low')
        return _SCIPY_SIGNAL.filtfilt(b, a, data)

    match method:
        case ENVELOPE_MODS.Hilbert:
            analytic = hilbert(S, mod=hilbert_mod, **kwargs)
            env = np.abs(analytic)

        case ENVELOPE_MODS.Lowpass:
            env_rect = np.abs(S)
            env = (np.pi / 2.0) * _lowpass(env_rect, cutoff=cf)

        case ENVELOPE_MODS.IQ:
            T = np.arange(len(S)) / fs
            i_comp = S * np.cos(2 * np.pi * fc * T)
            q_comp = S * np.sin(2 * np.pi * fc * T)

            i_base = _lowpass(i_comp, cutoff=cf)
            q_base = _lowpass(q_comp, cutoff=cf)

            env = 2 * np.sqrt(i_base ** 2 + q_base ** 2)

        case ENVELOPE_MODS.Peak:
            result = peak(S, **kwargs)
            x = result[0]
            y = result[1]

            f = _SCIPY_INTERAPT.CubicSpline if spline is None else spline
            f = f(x, y, **kwargs)
            env = f(np.arange(len(S)))

        case _:
            raise ValueError(f"Unknown envelope method: {method}")

    if env is None:
        raise ValueError(f"the envelope is None!")

    return env
