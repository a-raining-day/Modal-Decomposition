"""
Modal Decomposition:
    LMD、CEEMDAN、EFD、CEEFD、VMD、EEMD、FMD、EWT、SSA、RPSEMD、CEEMD、MEMD、ICEEMDAN、EMD

GitHub url: https://github.com/a-raining-day/Modal-Decomposition

Python version:
    3.10.11

Role:  (if None write None)
    As the entrance of the lib

Lib and Version:  (if None write None)
    numpy - 2.2.6
	typing - 4.15.0

Only accessed by:  (must)
    All

Modify:  (must)
    2026.3.25

Description: (if None write None)
    As the entrance of the lib

Modify:
    Optimize the cost of import, from 5.001s to 0.747s. Put some heavy lib into internal of the function
"""
from .help_function import is_increasing

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

__all__ = ["Class", "Function", "is_increasing"]

ceefd_cls = CEEFD
ceefd_real_cls = CEEFD()
ssa_cls = SSA()
svmd_cls = SVMD()

Origin_CEEFD = ceefd_cls
Origin_EEMD = Origin_EEMD
Origin_SSA = SSA
Origin_SVMD = SVMD


class Class:
    # class | You can initial yourself class
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

