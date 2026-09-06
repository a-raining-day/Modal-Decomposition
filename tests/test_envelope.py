import pytest
import numpy as np
from matplotlib import pyplot as plt

from src.Modal_Decomposition.Utils.Envelope import envelope

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
    plt.show()
