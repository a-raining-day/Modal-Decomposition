import numpy as np
from typing import Literal, Callable

_SCIPY_HILBERT: Callable | None = None

def hilbert \
(
    S: np.ndarray,
    mode: Literal[
    "Scipy", "Fourth-order Wave", "FHT", "SB-Hilbert", "Kramers-Kronig", "FIR", "HST", "FFT", "Optional-order FIR"
    ],
    cpp: bool = False
) -> np.ndarray:
    """
    the security of the input should finish on the outside, such as: EMD eta. methods.

    method              | complexity    | precision | speed    | decision
    Fourth-order Wave   | O(n)          | high      | fastest  |
    FHT                 | O(n log n)    | mid       | fast     |
    SB-Hilbert          | < O(n log n)  | mid       | fast     |
    Kramers-Kronig      | high          | highest   | slowest  |
    FIR                 | O(L)          | high      | slow     |
    HST                 | O(n log n)    | high      | fast     |
    FFT                 | O(n log n)    | high      | fast     |
    Optional-Order FIR  | O(order)      | mid       | mid      |

    :param S:
    :param mode:
    :param cpp:
    :return:
    """

    # TODO: cpp=True -> Means use pybind11 to optimize. Now default False.
    global _SCIPY_HILBERT

    match mode:
        case "Scipy":
            if _SCIPY_HILBERT is None:
                try:
                    from scipy.signal import hilbert
                    _SCIPY_HILBERT = hilbert
                except ImportError:
                    raise ImportError("the SciPy is not installed")

            return _SCIPY_HILBERT(S)

        case "Fourth-order Wave":
            ...

        case "FHT":
            ...

        case "SB-Gilbert":
            ...

        case "Kramers-Kronig":
            ...

        case "FIR":
            ...

        case "window-FFT":
            ...

        case "HST":
            ...

        case "FFT":
            ...

        case "Optional-order FIR":
            ...

        case _:
            raise ValueError(f"Unknown mode: {mode}")