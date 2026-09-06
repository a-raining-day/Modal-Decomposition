"""
Modal Decomposition

A unified library for modal decomposition methods:
LMD, CEEMDAN, EFD, CEEFD, VMD, EEMD, FMD, EWT, SSA, RPSEMD, CEEMD, MEMD,
ICEEMDAN, EMD, SVMD.

Public interface
----------------
- ``Class.X``     : decomposer class; use ``Class.X(**params).decompose(S, T)``.
- ``Function.X``  : facade function; use ``Function.X(S, T=None, **params)``.
- ``set_seed``    : set the process-level global random seed.
- ``get_seed``    : read the current global random seed.

Every method returns a ``DecompositionResult`` with ``.IMFs``, ``.Res``,
``.info`` and ``.config``. ``Res`` is None for methods without a residual
concept (SSA, VMD).
"""

from importlib.metadata import PackageNotFoundError, version as _pkg_version

from . import \
(
    CEEFD,
    CEEMD,
    CEEMDAN,
    EEMD,
    EFD,
    EMD,
    EWT,
    FMD,
    ICEEMDAN,
    LMD,
    MEMD,
    RPSEMD,
    SSA,
    SVMD,
    VMD
)  # noqa: F401  (registration side effects)

from ._Registry import _ClassRegistry
from .Base import Name, Reference
from .Utils import get_seed, set_seed, set_absolute_limit, set_memmap_ratio

__all__ = [
    "Class",
    "Function",
    "set_seed",
    "get_seed",
    "set_memmap_ratio",
    "set_absolute_limit",
    "__version__",
    "__author__",
    "__email__",
    "__license__",
    "__url__",
    "__description__",
]

try:
    __version__ = _pkg_version("Modal-Decomposition")
except PackageNotFoundError:
    __version__ = "0.1.6+src"

__author__ = "a-raining-day(Mao)"
__email__ = "2215269365@qq.com"
__license__ = "Apache 2.0"
__url__ = "https://github.com/a-raining-day/Modal-Decomposition"
__description__ = "A comprehensive modal decomposition library"


class _Namespace:
    """
    Read-only namespace over a registry snapshot.
    """

    def __init__(self, entries):
        self.__dict__.update(entries)

    def __dir__(self):
        return sorted(set(super().__dir__()) | set(self.__dict__))

    def __repr__(self):
        return f"<Namespace {sorted(self.__dict__)}>"


def _compose_doc(name: str, cls: type) -> str:
    """
    Compose the facade docstring from the class, its __init__ and the
    metadata tables.
    """
    import inspect

    full_name = Name.get(name, name)
    reference = Reference.get(name)

    class_doc = (cls.__doc__ or "").strip()
    init_doc = (cls.__init__.__doc__ or "").strip()

    parts = [full_name]

    if class_doc:
        parts.append(class_doc)

    if init_doc:
        parts.append("Parameters\n----------\n" + init_doc)

    parts.append(
        "Returns\n"
        "-------\n"
        "DecompositionResult\n"
        "    .IMFs   : np.ndarray\n"
        "        decomposed modes\n"
        "    .Res    : np.ndarray | None\n"
        "        residual\n"
        "    .info   : dict\n"
        "        method-specific diagnostics\n"
        "    .config : " + name + "Config\n"
        "        effective parameter snapshot of this run"
    )

    if reference:
        parts.append("References\n----------\n" + reference)

    if "seed" in inspect.signature(cls.__init__).parameters:
        parts.append(
            "Random seed\n"
            "-----------\n"
            "If Modal_Decomposition.set_seed was called with a non-None value, "
            "the global seed overrides this method's local seed and a "
            "UserWarning is emitted."
        )

    return "\n\n".join(parts)


def _make_facade(name: str, cls: type):
    """
    Build Function.X from Class.X.
    """
    def facade(S, T=None, **kwargs):
        return cls(**kwargs).decompose(S, T)

    facade.__name__ = name
    facade.__qualname__ = f"Function.{name}"
    facade.__doc__ = _compose_doc(name, cls)
    return facade


Class = _Namespace(dict(sorted(_ClassRegistry.items())))
Function = _Namespace(
    {k: _make_facade(k, v) for k, v in sorted(_ClassRegistry.items())}
)
