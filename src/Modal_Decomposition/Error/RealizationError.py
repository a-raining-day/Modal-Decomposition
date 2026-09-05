"""
Error types of the package.
"""


class RealizationError(Exception):
    """
    Raised when a declared capability is not yet implemented.
    """


__all__ = ["RealizationError"]
