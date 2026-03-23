from PyEMD import EMD
import numpy as np
from COLOR.colorful_print import printc

EMD_cls = EMD

def emd(S, T=None, spline_kind: str = "cubic", nbsym: int = 2, max_imf=-1, fs=None):
    if not isinstance(S, np.ndarray):
        S = np.array(S)

    N = len(S)

    if T is None:
        if fs is not None:
            dt = 1.0 / fs  # smaple for time axis
            T = np.arange(N) * dt
        else:
            T = np.arange(N)  # default fs = 1
            printc(f"Warn: T is None，default T = [0, 1, 2, ..., {N - 1}]", color="red")

    EMD_cls = EMD(spline_kind, nbsym)

    EMD_cls.emd(S, T, max_imf=max_imf)