"""
Base package: contracts, metadata tables, and the config base class.
"""

from .ClassDefine import Decomposer, DecompositionResult
from .ConfigDefine import Config
from .ConstDefine import Name, Reference

__all__ = ["Decomposer", "DecompositionResult", "Config", "Name", "Reference"]
