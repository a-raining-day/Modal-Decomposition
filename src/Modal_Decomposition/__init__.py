"""
Modal Decomposition

A unified library for modal decomposition methods:
LMD, CEEMDAN, EFD, CEEFD, VMD, EEMD, FMD, EWT, SSA, RPSEMD, CEEMD, MEMD,
ICEEMDAN, EMD, SVMD, and the optional EWTpy (ewtpy-backed EWT).

Public interface
----------------
- ``Class.X``     : decomposer class; use ``Class.X(**params).decompose(S, T)``.
- ``Function.X``  : facade function; use ``Function.X(S, T=None, **params)``.
- ``set_seed``    : set the process-level global random seed.
- ``get_seed``    : read the current global random seed.

Every method returns a ``DecompositionResult`` with ``.IMFs``, ``.Res``,
``.info`` and ``.config``. ``Res`` is None for methods without a residual
concept (SSA, VMD).

``EWT`` 是本库自研实现(不依赖 ewtpy); ``EWTpy`` 是可选适配层, 只在真正
``decompose`` 时经 ``Base.Cache.cache.import_module`` 惰性导入 ``ewtpy``,
未安装该可选依赖时不影响其余方法。
"""

from .Base import *
from .Utils import *
from .Error import *
from .Define import *
from .Operator import *

from .EMD import EMD, f_EMD, EMDConfig

__all__ = [
    "Class",
    "Function",
    "__version__",
    "__author__",
    "__email__",
    "__license__",
    "__url__",
    "__description__",
]

__version__ = "0.4.0"
__author__ = "a-raining-day(Mao)"
__email__ = "2215269365@qq.com"
__license__ = "Apache 2.0"
__url__ = "https://github.com/a-raining-day/Modal-Decomposition"
__description__ = "A comprehensive modal decomposition library"


class Class:
    EMD = EMD

class Function:
    EMD = f_EMD
