"""
Python version:  (must)
    3.10.11

Lib and Version:  (if None write None)
    numpy - 2.2.6

Only accessed by:  (must)
    Only __init__.py

Description: (if None write None)
    Realize the RPSEMD

Modify:  (must)
    2026.3.25 - Create
    2026.4.3  - Finish the Optimization of the RPSEMD.
"""

import numpy as np
from .EMD import emd
from typing import Union, Tuple
from .Utils import is_monotonic


def rpsemd \
    (
        S: Union[list, np.ndarray],
        T: Union[list, np.ndarray] = None,
        f=None, M=4, max_imf=None,
        fs=1.0,
        spline_kind: str = "cubic",
        nbsym: int = 2,
    ) -> Tuple[np.ndarray, np.ndarray, None]:
    """
    RPSEMD: Regenerated Phase-shifted Sinusoid-assisted EMD

    """

    if not isinstance(S, np.ndarray):
        S = np.array(S)

    if T is None:
        T = np.arange(len(S))

    IMFs = []
    Res = S

    if max_imf is None:
        max_imf = int(np.log2(len(S)))

    count = 0
    while True:
        if f is None:
            N = len(Res)
            fft_vals = np.fft.rfft(Res)
            freqs = np.fft.rfftfreq(N, 1 / fs)

            magnitude = np.abs(fft_vals[1:])  # filter the dc
            if len(magnitude) > 0:
                main_freq_idx = np.argmax(magnitude) + 1
                current_f = freqs[main_freq_idx]
            else:
                current_f = fs / N
        else:
            current_f = f

        phi = np.array([2.0 * np.pi * i / M for i in range(M)])  # generate the sin wave
        orders = []

        for m in range(M):
            Am_t = np.sin(2 * np.pi * T * current_f + phi[m])
            Xm_t = Res + Am_t

            _IMFs, _ = emd(Xm_t, T, spline_kind=spline_kind, nbsym=nbsym, max_imf=1)

            if len(_IMFs) > 0:
                imf1 = _IMFs[0] if _IMFs.shape[0] > 0 else _IMFs
                orders.append(imf1)

        if len(orders) == 0:
            break

        IMF = np.mean(orders, axis=0)
        IMFs.append(IMF)
        Res = Res - IMF

        count += 1

        if max_imf != -1 and count >= max_imf or len(Res) < 4 or np.std(Res) < 1e-10 or is_monotonic(Res):
            break

    if len(Res) > 0 and np.std(Res) > 1e-10:
        IMFs.append(Res)
        Res = np.zeros_like(Res)

    return np.array(IMFs), np.array(Res), None