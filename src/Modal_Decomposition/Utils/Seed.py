"""
Two-level random seed management.

Global seed state plus resolution of effective per-method seeds. Global
seed overrides a non-None local seed and emits a UserWarning.
"""

import operator
import warnings

__all__ = ["set_seed", "get_seed", "resolve_seed"]

_GLOBAL_SEED = None


def _coerce_seed(seed) -> int | None:
    """
    Validate a seed value.

    Parameters
    ----------
    seed : int | None
        Seed candidate.

    Returns
    -------
    int | None
        The validated integer seed or None.

    Raises
    ------
    TypeError
        On bool or non-integer values.
    ValueError
        On negative values.
    """
    if seed is None:
        return None
    if isinstance(seed, bool):
        raise TypeError("seed must be an int or None, got bool")
    try:
        value = operator.index(seed)
    except TypeError:
        raise TypeError(f"seed must be an int or None, got {type(seed).__name__}")
    if value < 0:
        raise ValueError(f"seed must be non-negative, got {value}")
    return value


def set_seed(seed) -> None:
    """
    Set the process-level global seed.

    The global seed, when not None, overrides any non-None local seed of a
    decomposition method. Call before starting decompositions; changing it
    during a run is not thread-safe.

    Parameters
    ----------
    seed : int | None
        Non-negative integer, or None to clear the global seed.
    """
    global _GLOBAL_SEED
    _GLOBAL_SEED = _coerce_seed(seed)


def get_seed() -> int | None:
    """
    Return the current global seed, or None when unset.
    """
    return _GLOBAL_SEED


def resolve_seed(local_seed, method: str) -> tuple[int | None, bool]:
    """
    Resolve the effective seed for a method.

    Parameters
    ----------
    local_seed : int | None
        The method's local seed parameter.
    method : str
        Method name used in the override warning.

    Returns
    -------
    tuple[int | None, bool]
        (effective_seed, overridden). ``overridden`` is True when a non-None
        local seed was replaced by the global seed.
    """
    local = _coerce_seed(local_seed)
    global_seed = _GLOBAL_SEED

    if global_seed is not None:
        if local is not None:
            warnings.warn(
                f"global seed {global_seed} overrides local seed {local} for method {method}",
                UserWarning,
                stacklevel=2,
            )
            return global_seed, True
        return global_seed, False

    return local, False
