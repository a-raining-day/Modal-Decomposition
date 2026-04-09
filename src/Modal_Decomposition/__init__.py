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

Dependence:
    antropy
    colorama
    einops
    EMD-signal
    ewtpy
    numba
    numpy
    scipy
    vmdpy

Modify:
    2026.3.25 - Optimize the cost of import, from 5.001s to 0.747s. Put some heavy lib into internal of the function
    2026.3.26 - Optimize the description of the type of input and output. now, the dim of input and output is more clear.
    2026.3.29 - Optimize the SSA.decompose function, time changed from 40min averagely to 2s averagely.
    2026.3.30 - Rebuilding All.
    2026.3.30 - Optimize the function of judging monotonicity.
    2026.4.3  - Finish the Optimization of the modal decomposition method. Except the MEMD method.
    2026.4.4  - Add the parameter to describe lib.
    2026.4.6  - Change the position of the entrance of importing 'threading'. Try to reduce the cost of the import.
    2026.4.9  - Fix the import error of Class.VMD. From "vmdpy.EWT1D" to "vmdpy.VMD".
"""


__version__ = "0.1.1"
__author__ = "a-raining-day(Mao)"
__email__ = "None, Only QQ email."
__license__ = "Apache 2.0"
__url__ = "https://github.com/a-raining-day/Modal-Decomposition"
__description__ = "A comprehensive modal decomposition library"


from importlib import import_module

from .CEEFD import ceefd
from .CEEMD import ceemd
from .CEEMDAN import ceemdan
from .EEMD import eemd
from .EFD import efd
from .EMD import emd
from .EWT import ewt
from .FMD import fmd
from .ICEEMDAN import iceemdan
from .LMD import lmd
from .MEMD import memd
from .RPSEMD import rpsemd
from .SSA import SSA, ssa
from .SVMD import svmd
from .VMD import vmd


import warnings
warnings.warn("The MEMD is rebuilding...")


__all__ = \
    [
        "Function", "Class",
        "__version__",
        "__author__",
        "__email__",
        "__license__",
        "__url__",
        "__description__"
    ]


class Class:
    import threading

    __cache = {}

    CEEFD = ceefd

    @classmethod
    def EEMD(cls, **kwargs):
        if "EEMD" not in cls.__cache:
            try:
                Module = import_module("PyEMD").EEMD
                cls.__cache["EEMD"] = Module
                return Module(**kwargs)
            except ImportError:
                raise ImportError("No PyEMD, Please use `pip install EMD-signal`")

        else:
            return cls.__cache["EEMD"](**kwargs)

    @classmethod
    def EWT1D(cls, **kwargs):
        if "EWT1D" not in cls.__cache:
            try:
                Module = import_module("ewtpy").EWT1D
                cls.__cache["EWT1D"] = Module
                return Module(**kwargs)
            except ImportError:
                raise ImportError("No ewtpy, Please use `pip install ewtpy`")

        else:
            return cls.__cache["EWT1D"](**kwargs)

    SSA = SSA
    SVMD = SVMD

    @classmethod
    def VMD(cls, **kwargs):
        if "EWT1D" not in cls.__cache:
            try:
                Module = import_module("vmdpy").VMD
                cls.__cache["VMD"] = Module
                return Module(**kwargs)
            except ImportError:
                raise ImportError("No vmdpy, Please use `pip install vmdpy`")

        else:
            return cls.__cache["VMD"](**kwargs)

class Function:
    # function | default function for modal decomposition
    # the IMFs (2-dim) means: (K, len(Signal)) (K is the num of IMFs)
    # CEEFD = ceefd_real_cls.ceefd

    CEEFD = Class.CEEFD(fs=1.0, min_peak_distance=10, envelop_iter=3)
    CEEMD = ceemd
    CEEMDAN = ceemdan
    EEMD = eemd
    EFD = efd
    EMD = emd
    EWT = ewt
    FMD = fmd
    ICEEMDAN = iceemdan
    LMD = lmd
    MEMD = memd
    RPSEMD = rpsemd
    SSA = ssa
    SVMD = svmd
    VMD = vmd

    # __cache = \
    # {
    #     "CEEFD": CEEFD,
    #     "CEEMD": CEEMD,
    #     "CEEMDAN": CEEMDAN,
    #     "EEMD": EEMD,
    #     "EFD": EFD,
    #     "EMD": EMD,
    #     "EWT": EWT,
    #     "FMD": FMD,
    #     "ICEEMDAN": ICEEMDAN,
    #     "LMD": LMD,
    #     "MEMD": MEMD,
    #     "RPSEMD": RPSEMD,
    #     "SSA": SSA,
    #     "SVMD": SVMD,
    #     "VMD": VMD
    # }


if __name__ == '__main__':
    Function.AAA