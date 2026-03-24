import numpy as np
from numba import njit

def old_is_increasing(S) -> bool:
    diff = np.ediff1d(S)
    epsilon = 1e-8
    return np.all(diff > epsilon)

@njit
def is_increasing(S, rtol=1e-8, atol=1e-8):
    for i in range(len(S)-1):
        diff = S[i+1] - S[i]
        if diff <= atol + rtol * abs(S[i]):
            return False
    return True