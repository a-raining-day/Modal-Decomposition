"""
Utility subpackage: validation, monotonicity, memory policy, and seed
management.
"""

from .Check import Check_Time_and_Signal, detect_dtype, is_uniform, require_ndim, to_signal
from .Memory import get_available_memory, get_memory_policy, set_absolute_limit, set_memmap_ratio, should_use_memmap
from .Monotonicity import Monotony, is_monotonic, monotonic
from .Seed import get_seed, resolve_seed, set_seed

__all__ = [
    "Check_Time_and_Signal",
    "is_uniform",
    "require_ndim",
    "to_signal",
    "detect_dtype",
    "get_available_memory",
    "set_memmap_ratio",
    "set_absolute_limit",
    "get_memory_policy",
    "should_use_memmap",
    "Monotony",
    "monotonic",
    "is_monotonic",
    "set_seed",
    "get_seed",
    "resolve_seed",
]


"""
method              | complexity    | precision | speed    | decision
Fourth-order Wave   | O(n)          | high      | fastest  |
FHT                 | O(n log n)    | mid       | fast     |
SB-Hilbert          | < O(n log n)  | mid       | fast     |
Kramers-Kronig      | high          | highest   | slowest  |
FIR                 | O(L)          | high      | slow     |
HST                 | O(n log n)    | high      | fast     |
FFT                 | O(n log n)    | high      | fast     |
Optional-Order FIR  | O(order)      | mid       | mid      |
"""