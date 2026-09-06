import numpy as np

def _hilbert(S: np.ndarray) -> np.ndarray:
    from scipy.signal import hilbert

    return hilbert(S)