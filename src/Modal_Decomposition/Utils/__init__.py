"""
Python version:
    3.10.11

Role:  (if None write None)
    Entrance of Utils

Lib and Version:  (if None write None)
    numpy - 2.2.6

Only accessed by:  (must)
    All

Description: (if None write None)
    As the entrance of the utils sub-lib.

Modify:  (must)
    2026.3.30 -
"""

from .EnvironmentMemory import EnvironmentMemory
from .NumpyNdarray_MemoryCalculator import view_memory, root_memory
from .Monotonicity import monotonic_increasing, monotonic_decreasing
from .Check import Check_Time_and_Signal

def is_monotonic(arr) -> bool:
    if monotonic_decreasing(arr) or monotonic_increasing(arr):
        return True
    else:
        return False