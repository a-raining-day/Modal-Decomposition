import numpy as np
from ...Base import Cache

def _hilbert(S: np.ndarray, **kwargs) -> np.ndarray:
    verbose = kwargs.get("verbose", False)
    ss = Cache.import_module("scipy.signal", verbose=verbose)

    return ss.hilbert(S)