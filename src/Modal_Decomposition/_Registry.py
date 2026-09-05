"""
Class registry for decomposition methods.

Method modules register their Decomposer subclass with the decorator at
import time; the package entry point builds the public namespaces from
``_ClassRegistry``.
"""

__all__ = ["register_class", "_ClassRegistry"]

_ClassRegistry: dict[str, type] = {}


def register_class(name: str):
    """
    Register a Decomposer subclass under ``name``.

    The class attribute ``cls.name`` must equal the registry key.

    Raises
    ------
    TypeError
        If the decorated object is not a class.
    ValueError
        If ``cls.name`` mismatches the key or the key is already registered.
    """
    def dec(cls):
        if not isinstance(cls, type):
            raise TypeError(f"Registry entry {name!r} must be a class")
        if getattr(cls, "name", None) != name:
            raise ValueError(
                f"{cls.__name__}.name must equal registry key {name!r}"
            )
        if name in _ClassRegistry:
            raise ValueError(f"Duplicate registration for {name!r}")
        _ClassRegistry[name] = cls
        return cls

    return dec
