"""
Python version:  (must)
    3.10.11

Lib and Version:  (if None write None)
    numpy - 2.2.6

Only accessed by:  (must)
    Only __init__.py

Description: (if None write None)
    Realize the CEEMDAN.

Modify:  (must)
    2026.3.25 - Create.
"""

from typing import Union, Tuple
import numpy as np

def ceemdan \
    (
        S: Union[list, np.ndarray], T: Union[list, np.ndarray] = None,
        max_imf: int = -1,
        trials=10,
        noise_width=0.05,  # default: 0.05-0.3
        noise_seed=42,  # seed
        spline_kind='cubic',
        nbsym=2,  # Number of boundary symmetry points
        extrema_detection='parabol',
        parallel=False,
        processes=None,  # None = auto, int >= 1
        random_state=42,
        noise_scale=1.0,  # scale factor of noise
        noise_kind='normal',  # noise kind: 'normal', 'uniform'
        range_thr=0.01,  # Stop threshold
        total_power_thr=0.005
    ) -> Tuple[np.ndarray, np.ndarray, None]:

    """
    CEEMDAN: Complete Ensemble Empirical Mode Decomposition with Adaptive Noise

    :param S: Signal (1-dim)
    :param T: Time axis (1-dim). Default uniform, or input the Unix.
    :param max_imf: the num of the decomposed IMFs. | -1 means all.
    :param trials:
    :param noise_width:
    :param noise_seed:
    :param spline_kind:
    :param nbsym:
    :param extrema_detection:
    :param parallel:
    :param processes:
    :param random_state:
    :param noise_scale:
    :param noise_kind:
    :param range_thr:
    :param total_power_thr:
    :return: IMFs (n_IMFs, N), Res (N,), None
    """

    from PyEMD import CEEMDAN

    if not isinstance(S, np.ndarray):
        S = np.array(S)

    if S.ndim == 0:
        raise ValueError("The dim of the S must be 1-dim, not 0")

    elif S.ndim > 1:
        if 1 in S.shape:
            S = S.reshape(-1)

        else:
            raise ValueError(f"The dim of S must be 1-dim, not {S.ndim}")

    N = len(S)

    if T is None:  # if T is None, default generate uniform T-axis.
        T = np.arange(N)  # default fs = 1
        print(f"Warn: T is None，default T = [0, 1, 2, ..., {N - 1}]")

    else:
        if not isinstance(T, np.ndarray):
            T = np.array(T)

    CEEMDAN = CEEMDAN \
    (
        trials=trials,
        noise_width=noise_width,
        noise_seed=noise_seed,
        spline_kind=spline_kind,
        nbsym=nbsym,
        extrema_detection=extrema_detection,
        parallel=parallel,
        processes=processes,
        random_state=random_state,
        noise_scale=noise_scale,
        noise_kind=noise_kind,
        range_thr=range_thr,
        total_power_thr=total_power_thr
    )

    IMF_Residue = CEEMDAN.ceemdan(S, T, max_imf)

    IMFs = IMF_Residue[:-1, :]  # shape [n_imfs, len(S)]
    Res = IMF_Residue[-1, :]

    return IMFs, Res, None