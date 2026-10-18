"""
Base class for per-method configuration snapshots.
"""

from dataclasses import asdict
from enum import Enum

__all__ = ["Config"]


def _to_plain(value):
    """
    递归把 ``Enum`` 成员换成它的 ``.value``, 让 ``to_dict()`` 真正是 "plain dict"
    (可直接 ``json.dumps``)。``bool``/``int``/``float``/``str``/``None`` 原样返回。
    """
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {k: _to_plain(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(_to_plain(v) for v in value)
    if isinstance(value, list):
        return [_to_plain(v) for v in value]
    if isinstance(value, set):
        return {_to_plain(v) for v in value}
    return value


class Config:
    """
    Immutable parameter snapshot of a decomposition run.

    Subclasses are frozen, keyword-only dataclasses defined next to their
    method. Instances record the effective parameter values of one run.
    Enum-valued parameters are stored as Enum members on the snapshot (type
    annotations stay honest) and are expanded to their ``.value`` only when
    the snapshot is exported through :meth:`to_dict`.
    """

    def to_dict(self) -> dict:
        """
        Return the configuration as a plain dict.

        ``Enum`` members (including ones nested inside ``dict`` / ``tuple`` /
        ``list`` values) are expanded to their ``.value``, so the result is
        JSON-serialisable.
        """
        return {k: _to_plain(v) for k, v in asdict(self).items()}
