from ewtpy import EWT1D
from typing import Tuple, Union, Optional
import numpy as np

def ewt \
(
    S,
    N: int = 5,
    log: int = 0,
    detect: str = "locmax",
    completion: int = 0,
    reg: str = 'average',
    lengthFilter: int = 10,
    sigmaFilter: int = 5,
    need_mfd: bool = False,
    need_boundaries: bool = False) -> Tuple[np.ndarray, np.ndarray] | Tuple[np.ndarray, np.ndarray, np.ndarray] | Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """

    :param S: Signal
    :param N:
    :param log:
    :param detect:
    :param completion:
    :param reg:
    :param lengthFilter:
    :param sigmaFilter:
    :return: IMFs(N, len(S)) (2-dim)
    """

    ewt, mfb, boundaries = EWT1D(S, N, log, detect, completion, reg, lengthFilter, sigmaFilter)
    ewt = ewt.T
    mfb = mfb.T

    if need_mfd and not need_boundaries:
        return ewt[:-1, :], ewt[-1, :], mfb

    if need_boundaries and not need_mfd:
        return ewt[:-1, :], ewt[-1, :], boundaries

    if need_mfd and need_boundaries:
        return ewt[:-1, :], ewt[-1, :], mfb, boundaries

    if not need_mfd and not need_boundaries:
        return ewt[:-1, :], ewt[-1, :]


if __name__ == '__main__':
    import numpy as np

    S = np.random.rand(1, 500).squeeze()

    EWT, mfb, boundarire = ewt(S)

    print(EWT)

    print(type(EWT))