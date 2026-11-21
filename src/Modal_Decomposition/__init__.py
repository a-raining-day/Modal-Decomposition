"""
Modal Decomposition

A unified library for modal decomposition methods:
EMD, EEMD, CEEMD, CEEMDAN, ICEEMDAN, RPSEMD (EMD family) and
EFD, CEEFD, MEMD, SSA, FMD, LMD, VMD, EWT, SVMD.

Public interface
----------------
- ``Class.X``     : decomposer class; use ``Class.X(**params).decompose(S)``.
- ``Function.X``  : facade function; use ``Function.X(S, **params)``.
- ``set_seed``    : set the process-level global random seed.
- ``get_seed``    : read the current global random seed.

Every method returns a ``DecompositionResult`` with ``.IMFs``, ``.Res``,
``.info`` and ``.config``. ``Res`` is None for methods without a residual
concept (SSA, VMD).
"""

from .Base import *
from .Utils import *
from .Error import *
from .Define import *
from .Operator import *

from .EMD import EMD, f_EMD, EMDConfig
from .EEMD import EEMD, f_EEMD, EEMDConfig
from .CEEMD import CEEMD, f_CEEMD, CEEMDConfig
from .CEEMDAN import CEEMDAN, f_CEEMDAN, CEEMDANConfig
from .ICEEMDAN import ICEEMDAN, f_ICEEMDAN, ICEEMDANConfig
from .RPSEMD import RPSEMD, f_RPSEMD, RPSEMDConfig

from .EFD import EFD, f_EFD, EFDConfig
from .CEEFD import CEEFD, f_CEEFD, CEEFDConfig
from .MEMD import MEMD, f_MEMD, MEMDConfig
from .SSA import SSA, f_SSA, SSAConfig
from .FMD import FMD, f_FMD, FMDConfig
from .LMD import LMD, f_LMD, LMDConfig
from .VMD import VMD, f_VMD, VMDConfig
from .EWT import EWT, f_EWT, EWTConfig
from .SVMD import SVMD, f_SVMD, SVMDConfig

import warnings
import numpy as np

__all__ = [
    "Class",
    "Function",
    "set_seed",
    "get_seed",
    "__version__",
    "__author__",
    "__email__",
    "__license__",
    "__url__",
    "__description__",
]

__version__ = "0.5.1"
__author__ = "a-raining-day(Mao)"
__email__ = "2215269365@qq.com"
__license__ = "Apache 2.0"
__url__ = "https://github.com/a-raining-day/Modal-Decomposition"
__description__ = "A comprehensive modal decomposition library"


def set_seed(seed: int, lock: bool = False, verbose: bool = True) -> int:
    """
    设置进程级全局随机种子。

    同时写入两处, 因为库里两种随机源并存:

    * ``Base.Seed.GLOBAL_SEED`` —— 由 FMD 等显式读种子的方法使用;
    * ``numpy`` 全局 RNG (``np.random.seed``) —— 由 EEMD / CEEMD / CEEMDAN /
      ICEEMDAN 等直接调用 ``np.random.normal`` 的加噪方法使用。

    Returns
    -------
    int
        设置后的全局种子。
    """
    np.random.seed(int(seed))
    return Seed.set(seed, lock=lock, verbose=verbose)

def get_seed() -> int:
    """读取当前进程级全局随机种子。"""
    return Seed.get()

def set_allocable_cpu_ratio(cpu_ratio: float) -> None:
    """设置并行时可用 cpu 数量, 默认为 2/3"""
    if isinstance(cpu_ratio, float):
        if 0 < cpu_ratio <= 1:
            if cpu_ratio == 1:
                warnings.warn("the `cpu ratio` is 1! means all cpu will be used!")
            else:
                pass
        else:
            raise ValueError("the `cpu ratio` must be between 0 and 1!")
    else:
        raise ValueError("the `cpu ratio` must be float!")

    global DEFAULT_ALLOCABLE_CPU_RATIO
    DEFAULT_ALLOCABLE_CPU_RATIO = cpu_ratio

def set_numpy_default_type(dtype: float | np.floating) -> None:
    if not isinstance(dtype, float) and not isinstance(dtype, np.floating):
        raise ValueError("the `dtype` must be float or np.floating!")

    global DEFAULT_NUMPY_TYPE
    DEFAULT_NUMPY_TYPE = dtype

class Class:
    EMD = EMD
    EEMD = EEMD
    CEEMD = CEEMD
    CEEMDAN = CEEMDAN
    ICEEMDAN = ICEEMDAN
    RPSEMD = RPSEMD
    EFD = EFD
    CEEFD = CEEFD
    MEMD = MEMD
    SSA = SSA
    FMD = FMD
    LMD = LMD
    VMD = VMD
    EWT = EWT
    SVMD = SVMD

class Function:
    EMD = f_EMD
    EEMD = f_EEMD
    CEEMD = f_CEEMD
    CEEMDAN = f_CEEMDAN
    ICEEMDAN = f_ICEEMDAN
    RPSEMD = f_RPSEMD
    EFD = f_EFD
    CEEFD = f_CEEFD
    MEMD = f_MEMD
    SSA = f_SSA
    FMD = f_FMD
    LMD = f_LMD
    VMD = f_VMD
    EWT = f_EWT
    SVMD = f_SVMD


