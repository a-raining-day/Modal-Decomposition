import numpy as np
from typing import Union, Tuple

def Check_Time_and_Signal(S: Union[list, np.ndarray], T: Union[list, np.ndarray]=None, verbose: bool=False) -> Tuple[np.ndarray, np.ndarray, int]:
    if not isinstance(S, list) and not isinstance(S, np.ndarray):
        raise TypeError("The api for predictor model only accept one parameter -> ArrayLike(list | np.ndarray)")

    if not isinstance(S, np.ndarray):
        S = np.array(S)

    if S.ndim == 0:
        raise ValueError("The dim of the output of the predictor model must be 1-dim!")

    elif S.ndim == 2:
        if 1 not in S.shape:
            raise ValueError(
                "If the dim of the output of the predictor model is 2-dim, the shape must be (1, N) or (N, 1)")

        else:
            S = S.squeeze()

    elif S.ndim >= 3:
        raise ValueError("The dim of the predictor model must be a array.")

    N = len(S)

    if T is None:  # if T is None, default generate uniform T-axis.
        T = np.arange(N)  # default fs = 1
        if verbose:
            print(f"Warn: T is None，default T = [0, 1, 2, ..., {N - 1}]")

    else:
        if not isinstance(T, np.ndarray):
            T = np.array(T)

        if len(T) != len(S):
            raise ValueError("The length of T must be equal to Signal.")

    return S, T, N