"""
Pluggable decomposition-method workers for the memory pipeline.

Every method registered in ``Modal_Decomposition.Class`` gets a worker from
:func:`make_worker` automatically, so the same matrix runner can be inserted
for any decomposition method (EMD, CEEMDAN, VMD, EWT, ...) without further
changes. Methods whose optional third-party backend (e.g. PyEMD) is missing
report an ``error`` / ``skipped`` record instead of crashing the matrix.
"""

from __future__ import annotations

from typing import Callable, Optional

import Modal_Decomposition as MD

__all__ = [
    "available_methods",
    "make_worker",
    "METHOD_PARAMS",
]

# Per-method constructor defaults the workers pass to keep runs bounded and
# deterministic. Extend this dict when inserting a new method into the matrix.
METHOD_PARAMS: dict[str, dict] = {
    "EMD": {"max_imf": 3},
}


def available_methods() -> list[str]:
    """Names of every decomposition method registered in the package."""
    return sorted(MD.Class.__dict__)


def make_worker(method: str) -> Callable:
    """
    Build ``run(S, T=None, params=None)`` for a registered method.

    ``run`` instantiates ``Class.<method>(**params)`` and calls
    ``decompose(S, T)``. Decompositions whose constructor/backend imports
    fail raise (``ImportError`` and friends) and are recorded by the runner.
    """
    if method not in MD.Class.__dict__:
        raise ValueError(
            f"unknown method {method!r}; registered: {available_methods()}"
        )

    cls = getattr(MD.Class, method)

    def run(S, T=None, params: Optional[dict] = None):
        kwargs = dict(METHOD_PARAMS.get(method, {}))
        if params:
            kwargs.update(params)
        return cls(**kwargs).decompose(S, T)

    run.__name__ = method
    run.__qualname__ = f"worker_{method}"
    return run
