import numpy as np
from .EMD import EMD
from . import is_increasing

def ceemd(S, T=None, fs=None, mean=0, std_dev=1, beta=1):

    N = len(S)

    if T is None:
        if fs is not None:
            # smaple
            dt = 1.0 / fs
            T = np.arange(N) * dt
        else:
            # default smaple f = 1
            T = np.arange(N)
            print(f"Warn: T is None，default T = [0, 1, 2, ..., {N - 1}]")

    white_noise = np.random.normal(N) * std_dev + mean

    S = S + white_noise * beta
    IMFs = []
    while True:
        _IMFs, _Res = EMD(S, T, fs)

        IMF = np.average(_IMFs, axis=0)
        IMFs.append(IMF)

        S -= IMF

        if is_increasing(S):
            Res = S
            break

    return np.array(IMFs), np.array(Res)