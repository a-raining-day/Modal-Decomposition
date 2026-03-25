"""
Python version:  (must)
    3.10.11

Lib and Version:  (if None write None)
    EMD-signal - 1.9.0

Only accessed by:  (must)
    Only __init__.py

Modify:  (must)
    2026.3.25

Description: (if None write None)
    Realize the EEMD.
"""

from PyEMD import EEMD

Origin_EEMD = EEMD

EEMD = EEMD()
def eemd(S):
    return EEMD.eemd(S)