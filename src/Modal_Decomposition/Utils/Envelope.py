import numpy as np
from typing import Literal
from Hilbert import hilbert

def envelope(S: np.ndarray, method: Literal["Hilbert", "Lowpass", "IQ", "PeakInterpolation"], fs: float = None, fc: float = None, cf: float = None) -> np.ndarray:
    """
    :param S:
    :param method:
    :param fs: sampling rate
    :param fc: carrier frequency
    :param cf: cut-off frequency
    :return:
    """
    # don't need check_Time_and_Single inside

    try:
        from scipy.signal import hilbert, butter, filtfilt, find_peaks  # TODO: the import overhead can optimize
        from scipy.interpolate import CubicSpline

    except ImportError:
        raise ImportError("Scipy not installed")

    assert isinstance(S, np.ndarray), TypeError("The kind of input should be np.ndarray!")

    env: np.ndarray = None

    if method in ["Lowpass", "IQ"]:
        if (fc is None) and (fs is None) and (cf is None):
            raise ValueError("Either fc, fs or cf must be specified!")

    def _lowpass(data, cutoff, order=4):
        nyq = 0.5 * fs
        b, a = butter(N=order, Wn=(cutoff / nyq), btype='low')
        return filtfilt(b, a, data)

    match method:
        case "Hilbert":
            analytic = hilbert(S)
            env = np.abs(analytic)

        case "Lowpass":
            env_rect = np.abs(S)
            env = _lowpass(env_rect, cutoff=cf)

        case "IQ":
            T = np.arange(len(S)) / fs
            i_comp = S * np.cos(2 * np.pi * fc * T)
            q_comp = S * np.sin(2 * np.pi * fc * T)

            i_base = _lowpass(i_comp, cutoff=cf)
            q_base = _lowpass(q_comp, cutoff=cf)

            env = 2 * np.sqrt(i_base ** 2 + q_base ** 2)

        case "PeakInterpolation":
            min_distance = int(fs / (2 * max(fc, 200)))
            peaks, _ = find_peaks(S, distance=min_distance)

            cs = CubicSpline(peaks, S[peaks])
            env = cs(np.arange(len(S)))

        case _:
            raise ValueError(f"Unknown envelope method: {method}")

    if env is None:
        raise ValueError(f"the envelope is None!")

    return env

