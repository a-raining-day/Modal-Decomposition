import pytest
import numpy as np
import matplotlib
matplotlib.use("Agg")  # 无头环境: 禁用交互后端, 防止 plt.show() 阻塞
from matplotlib import pyplot as plt

from src.Modal_Decomposition.Utils.Envelope import envelope
from src.Modal_Decomposition.Base.Cache import cache

@pytest.fixture
def am_signal():
    fs = 10000
    fc = 1000
    fm = 50
    m = 0.8
    t = np.arange(0, 0.2, 1/fs)
    S = (1 + m * np.cos(2*np.pi*fm*t)) * np.cos(2*np.pi*fc*t)
    true_env = 1 + m * np.cos(2*np.pi*fm*t)
    cf = 200
    return S, true_env, fs, fc, cf

def test_output_shape(am_signal):
    S, _, fs, fc, cf = am_signal
    for method in ["Hilbert", "Lowpass", "IQ", "PeakInterpolation"]:
        env = envelope(S, method=method, fs=fs, fc=fc, cf=cf)
        assert env.shape == S.shape, f"the shape of the output of the {method}."

def test_hilbert_accuracy(am_signal):
    S, true_env, fs, fc, cf = am_signal
    env = envelope(S, method="Hilbert", fs=fs, fc=fc, cf=cf)
    assert np.sqrt(np.mean((env[20:-20] - true_env[20:-20])**2)) < 0.01

def test_iq_accuracy(am_signal):
    S, true_env, fs, fc, cf = am_signal
    env = envelope(S, method="IQ", fs=fs, fc=fc, cf=cf)
    assert np.sqrt(np.mean((env[20:-20] - true_env[20:-20])**2)) < 0.05

def test_output_shape_show(am_signal):
    """
    fig/tests/envelope.png
    """

    envs = []
    S, _, fs, fc, cf = am_signal
    for method in ["Hilbert", "Lowpass", "IQ", "PeakInterpolation"]:
        envs.append(envelope(S, method=method, fs=fs, fc=fc, cf=cf))

    fig, axes = plt.subplots(2, 2, sharex=True, sharey=True)
    (ax1, ax2), (ax3, ax4) = axes

    ax1.plot(S, label="S")
    ax1.plot(envs[0], label="hilbert")
    ax1.set_title("hilbert")

    ax2.plot(S, label="S")
    ax2.plot(envs[1], label="Lowpass")
    ax2.set_title("Lowpass")

    ax3.plot(S, label="S")
    ax3.plot(envs[2], label="IQ")
    ax3.set_title("IQ")

    ax4.plot(S, label="S")
    ax4.plot(envs[3], label="PeakInterpolation")
    ax4.set_title("Peak Interpolation")

    plt.savefig("fig/tests/envelope.png", dpi=300)
    plt.close(fig)


# ------------------------------------------------------------------ #
# cache integration: the two scipy submodules are registered once
# ------------------------------------------------------------------ #
def test_envelope_registers_scipy_signal_in_cache(am_signal):
    S, _, fs, fc, cf = am_signal
    cache.clear()
    envelope(S, method="Hilbert", fs=fs, fc=fc, cf=cf)
    assert cache.check("scipy.signal")
    assert cache.get("scipy.signal").__name__ == "scipy.signal"
    desc = cache.describe("scipy.signal")
    assert "hilbert" in desc


def test_envelope_registers_scipy_interpolate_in_cache(am_signal):
    S, _, fs, fc, cf = am_signal
    cache.clear()
    envelope(S, method="PeakInterpolation", fs=fs, fc=fc, cf=cf)
    assert cache.check("scipy.interpolate")
    assert cache.get("scipy.interpolate").__name__ == "scipy.interpolate"
    desc = cache.describe("scipy.interpolate")
    assert "CubicSpline" in desc


def test_envelope_imports_submodules_lazily(am_signal):
    """Hilbert mode only needs scipy.signal; interpolate stays unregistered."""
    S, _, fs, fc, cf = am_signal
    cache.clear()
    envelope(S, method="Hilbert", fs=fs, fc=fc, cf=cf)
    assert not cache.check("scipy.interpolate")


def test_envelope_registers_each_submodule_exactly_once(am_signal):
    """重复调用 envelope 不会重复注册: 4 种模式后仍只有 2 条缓存记录
    (模块级注册统一由 Utils.get_envelope 完成; envelope() 自身只注册
    scipy 子模块)。"""
    S, _, fs, fc, cf = am_signal
    cache.clear()
    for method in ["Hilbert", "Lowpass", "IQ", "PeakInterpolation"]:
        envelope(S, method=method, fs=fs, fc=fc, cf=cf)
    assert len(cache()) == 2
    assert sorted(cache.names()) == ["scipy.interpolate", "scipy.signal"]
