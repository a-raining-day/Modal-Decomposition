"""
Modal Decomposition:
    LMD、CEEMDAN、EFD、CEEFD、VMD、EEMD、FMD、EWT、SSA、RPSEMD、CEEMD、MEMD、ICEEMDAN、EMD
"""
import numpy as np
from typing import Literal

from .EFD import EFD
from .CEEFD import CEEFD
from .VMD import vmd
from .EEMD import Origin_EEMD, eemd
from .FMD import fmd
from .EWT import ewt
from .SSA import SSA
from .RPSEMD import rpsemd
from .CEEMD import ceemd
from .MEMD import memd
from .ICEEMDAN import iceemdan
from .LMD import lmd
from .SVMD import SVMD
from .EMD import emd

__all__ = ["Class", "Function"]

def is_increasing(S, threshold=2, tolerance: Literal["high", "mid", "low"]="high") -> bool:
    def count_extrema(x):
        interior = x[1:-1]
        left = x[:-2]
        right = x[2:]
        maxima = (interior > left) & (interior > right)
        minima = (interior < left) & (interior < right)
        return np.sum(maxima) + np.sum(minima)

    if tolerance == "high":
        ans_3 = count_extrema(S) <= threshold
        return ans_3

    elif tolerance == "mid":
        diff = np.diff(S)

        sign_changes = np.diff(np.sign(diff))

        count = np.sum(np.abs(sign_changes) == 2)

        ans_1 = count <= threshold  # one way to check
        return ans_1

    else:
        # the second way
        ans_2 = np.all(S[1:] >= S[:-1]) or np.all(S[1:] <= S[:-1])
        return ans_2

ceefd_cls = CEEFD
ceefd_real_cls = CEEFD()
ssa_cls = SSA()
svmd_cls = SVMD()

# class | You can initial yourself class
Origin_CEEFD = ceefd_cls
Origin_EEMD = Origin_EEMD
Origin_SSA = SSA
Origin_SVMD = SVMD


class Class:
    CEEFD = Origin_CEEFD
    EEMD = Origin_EEMD
    SSA = Origin_SSA
    SVMD = Origin_SVMD


class Function:
    # function | default function for modal decomposition
    EFD = EFD
    CEEFD = ceefd_real_cls.ceefd
    CEEMDAN = ceefd_real_cls.ceemdan
    VMD = vmd
    EEMD = eemd
    FMD = fmd
    EWT = ewt
    SSA = ssa_cls.decompose
    RPSEMD = rpsemd
    CEEMD = ceemd
    MEMD = memd
    ICEEMDAN = iceemdan
    LMD = lmd
    SVMD = svmd_cls.decompose
    EMD = emd

