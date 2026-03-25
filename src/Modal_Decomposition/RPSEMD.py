"""
Python version:  (must)
    3.10.11

Lib and Version:  (if None write None)
    numpy - 2.2.6

Only accessed by:  (must)
    Only __init__.py

Modify:  (must)
    2026.3.25

Description: (if None write None)
    Realize the RPSEMD
"""

import numpy as np
from .help_function import is_increasing
from .EMD import emd
from .COLOR import printc

def rpsemd(S, T=None, f=None, M=4, max_imf=None, fs=1.0, spline_kind: str = "cubic", nbsym: int = 2, emd_max_imf=-1):
    """
    :param emd_max_imf:
    :param nbsym:
    :param spline_kind:
    :param fs:
    :param T: time axis
    :param max_imf: max num of IMFs
    :param S: Signal
    :param f: auxiliary sine wave frequency
    :param M: num of phases
    :return: IMFs, Res -> np.ndarray(2-dim)
    """

    if not isinstance(S, np.ndarray):
        S = np.array(S)

    if f is None:
        N = len(S)
        fft_vals = np.fft.rfft(S)
        freqs = np.fft.rfftfreq(N, 1 / fs)

        magnitude = np.abs(fft_vals[1:])
        main_freq_idx = np.argmax(magnitude) + 1
        f = freqs[main_freq_idx]

        printc(f"warn：f is None | auto: {f:.2f} Hz", color="red")

    if T is None:
        T = np.arange(len(S))
        print(f"warn：T is None | deault：T = 0, 1, 2, ..., {len(S) - 1}")

    phi = np.array([2.0 * np.pi * i / M for i in range(M)])

    IMFs = []
    Res = S

    if max_imf is None:
        max_imf = int(np.log2(len(S)))

    for imf_idx in range(max_imf):
        orders = []
        for m in range(M):
            Am_t = np.sin(2 * np.pi * T * f + phi[m])
            Xm_t = Res + Am_t

            _IMFs, _Res = emd(Xm_t, T, spline_kind=spline_kind, nbsym=nbsym, max_imf=emd_max_imf)
            orders.append(_IMFs[0, :])

        IMF = np.average(orders, axis=0)
        IMFs.append(IMF)
        Res = Res - IMF

        if np.std(Res) < 1e-10 or is_increasing(Res) or is_increasing(-Res):
            break

    return np.array(IMFs), np.array(Res)