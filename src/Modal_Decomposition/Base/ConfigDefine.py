"""
Base class for per-method configuration snapshots.
"""

from dataclasses import asdict

__all__ = ["Config"]


class Config:
    """
    Immutable parameter snapshot of a decomposition run.

    Subclasses are frozen, keyword-only dataclasses defined next to their
    method. Instances record the effective parameter values of one run.
    """

    def to_dict(self) -> dict:
        """
        Return the configuration as a plain dict.
        """
        return asdict(self)
