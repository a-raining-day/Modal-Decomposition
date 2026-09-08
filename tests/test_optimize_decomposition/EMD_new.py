"""
Empirical Mode Decomposition

Decomposes a 1-d signal into intrinsic mode functions (IMFs) and a
residual using the sifting process of PyEMD.

References
----------
10.1098/rspa.1998.0193
"""

import numpy as np
from dataclasses import dataclass
from typing import ClassVar, Literal

from src.Modal_Decomposition.Base import Config, Decomposer, DecompositionResult, Cache
from src.Modal_Decomposition._Registry import register_class
from src.Modal_Decomposition.Utils import Check_Time_and_Signal, get_hilbert, get_monotonicity

hilbert = get_hilbert()
monotony = get_monotonicity()

class EMD(Decomposer):
    name: ClassVar[str] = "EMD"

    def __init__(
        self,
        nbsym: int = 2,
        spline_kind: Literal["akima", "cubic", "pchip", "cubic_hermite", "slinear", "quadratic", "linear"] = "cubic",
        max_imf: int = -1,
    ):
        """
        Parameters
        ----------
        nbsym : int
            Number of extrema mirrored at the signal boundaries.
        spline_kind : str
            Interpolation kind for envelope fitting.
        max_imf : int
            Maximum number of IMFs; -1 decomposes completely.
        """
        self.nbsym = nbsym
        self.spline_kind = spline_kind
        self.max_imf = max_imf

    def decompose(self, S, T=None) -> DecompositionResult:
        """
        Decompose the signal into IMFs and a residual.
        """
        from PyEMD import EMD as PyEMD_EMD

        S, T, _N = Check_Time_and_Signal(S, T, ndim={1}, method=self.name)


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    # from scipy.signal import hilbert

    fs = 500  # 采样率 500 Hz，足够高
    T = 1.0  # 时长 1 秒
    t = np.arange(0, T, 1 / fs)  # 时间轴
    f = 5.0  # 信号频率 5 Hz，远小于 fs/2
    S = np.sin(2.0 * np.pi * f * t) + t

    analytic = hilbert.hilbert(S, mod="FHT")
    env = np.abs(analytic)

    plt.figure(figsize=(10, 4))
    plt.plot(t, S, label='Signal')
    plt.plot(t, env, label='Upper envelope')
    plt.plot(t, -env, label='Lower envelope')
    plt.legend()
    plt.show()