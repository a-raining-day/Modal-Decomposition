"""
Error subpackage.
"""

class RealizationError(Exception):
    """
    Raised when a declared capability is not yet implemented.
    """

class PreError(Exception):
    """
    Raised error when the preposition is incomplete.
    """