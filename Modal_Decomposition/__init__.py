"""
提供通用接口:
    LMD、CEEMDAN、EFD、CEEFD、VMD、EEMD、FMD、EWT、SSA、RPSEMD、CEEMD、MEMD、ICEEMDAN
"""
import numpy as np

from .EMD import EMD
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


def is_increasing(S) -> bool:
    diff = np.ediff1d(S)
    epsilon = 1e-8
    return np.all(diff > epsilon)

ceefd_cls = CEEFD
ceefd_real_cls = CEEFD()
ssa_cls = SSA()
svmd_cls = SVMD()

# class | You can initial yourself class
Origin_CEEFD = ceefd_cls
Origin_EEMD = Origin_EEMD
Origin_SSA = SSA
Origin_SVMD = SVMD

# function | default function for modal decomposition
EFD = EMD
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
SVMD = svmd_cls.extract_mode