import numpy as np
from typing import Literal

from . import (
    _fft,
    _fht,
    _fir,
    _kramers_kronig,
    _scipy_backend,
    _wave
)

def hilbert \
(
    S: np.ndarray,
    mod: Literal \
    ["Scipy", "Fourth-order Wave", "FHT", "SB-Hilbert", "Kramers-Kronig", "FIR", "HST", "FFT", "Optional-order FIR"]
) -> np.ndarray:

    match mod:
        case "Scipy":
            return _scipy_backend._hilbert(S)

        case "Fourth-order Wave":
            ...

        case "FHT":
            return _fht._hilbert(S)

