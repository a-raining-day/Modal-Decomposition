"""
Process-wide memory policy: decide when disk-backed (``np.memmap``) processing
is preferable to in-RAM processing.

Two mutually exclusive strategies are available; ``USE_RATIO_STRATEGY``
selects which one is active (setting one always switches away from the
other):

- Ratio strategy (``USE_RATIO_STRATEGY = True``): use memmap when the
  projected usage reaches ``MEMMAP_RATIO_LIMIT`` (default ``0.6``, i.e. 60%)
  of the remaining available memory.
- Absolute strategy (``USE_RATIO_STRATEGY = False``): use memmap when the
  projected usage reaches ``ABSOLUTE_MEMMAP_LIMIT_BYTES`` (default 2 GiB).

This module only makes decisions -- it never performs IO. The policy is
process-wide state and applies to every caller of :func:`should_use_memmap`
(currently the input layer ``Check_Time_and_Signal`` and the chunk-size
adaptation of the monotonicity checks).
"""

import time
from typing import Optional

try:
    import psutil
except ImportError:  # psutil is a declared dependency; degrade gracefully
    psutil = None

from ..Base.ConstDefine import SIZE

__all__ = [
    "USE_RATIO_STRATEGY",
    "MEMMAP_RATIO_LIMIT",
    "ABSOLUTE_MEMMAP_LIMIT_BYTES",
    "should_use_memmap",
    "set_memmap_ratio",
    "set_absolute_limit",
    "get_memory_policy",
    "get_available_memory",
]

# Process-wide policy state. Exactly one strategy is active at any time.
USE_RATIO_STRATEGY: bool = True
MEMMAP_RATIO_LIMIT: float = 0.6
#: 默认绝对阈值 = 2GB (大小统一取自 Base.ConstDefine.SIZE)。
ABSOLUTE_MEMMAP_LIMIT_BYTES: int = 2 * SIZE["1GB"]

# One-second cache for the available-memory reading so per-call overhead
# stays negligible for small-array callers.
_cache_ts: float = 0.0
_cache_value: int = 0


def get_available_memory(force: bool = False) -> Optional[int]:
    """
    Return the remaining available system memory in bytes.

    The value is cached for one second; pass ``force=True`` for a fresh
    reading. Returns ``None`` when psutil is unavailable.
    """
    global _cache_ts, _cache_value

    if psutil is None:
        return None

    now = time.monotonic()
    if not force and _cache_value > 0 and now - _cache_ts < 1.0:
        return _cache_value

    _cache_value = int(psutil.virtual_memory().available)
    _cache_ts = now
    return _cache_value


def set_memmap_ratio(ratio: float) -> None:
    """
    Switch to the ratio strategy: memmap when the projected usage reaches
    ``ratio`` of the remaining available memory.

    Parameters
    ----------
    ratio : float
        Threshold fraction in ``(0, 1]``.

    Raises
    ------
    ValueError
        On a ratio outside ``(0, 1]``.
    """
    global USE_RATIO_STRATEGY, MEMMAP_RATIO_LIMIT

    if not isinstance(ratio, (int, float)) or not (0.0 < ratio <= 1.0):
        raise ValueError(f"ratio must be in (0, 1], got {ratio!r}")

    MEMMAP_RATIO_LIMIT = float(ratio)
    USE_RATIO_STRATEGY = True


def set_absolute_limit(limit_bytes: int) -> None:
    """
    Switch to the absolute strategy: memmap when the projected usage reaches
    ``limit_bytes``.

    Parameters
    ----------
    limit_bytes : int
        Threshold in bytes; must be positive.

    Raises
    ------
    ValueError
        On a non-positive limit.
    """
    global USE_RATIO_STRATEGY, ABSOLUTE_MEMMAP_LIMIT_BYTES

    if int(limit_bytes) <= 0:
        raise ValueError(f"limit_bytes must be positive, got {limit_bytes!r}")

    ABSOLUTE_MEMMAP_LIMIT_BYTES = int(limit_bytes)
    USE_RATIO_STRATEGY = False


def get_memory_policy() -> dict:
    """Return a read-only view of the current policy."""
    return {
        "use_ratio_strategy": USE_RATIO_STRATEGY,
        "memmap_ratio_limit": MEMMAP_RATIO_LIMIT,
        "absolute_memmap_limit_bytes": ABSOLUTE_MEMMAP_LIMIT_BYTES,
    }


def should_use_memmap(nbytes: int, extra: int = 0) -> bool:
    """
    Whether the projected memory usage should be served from a memmap.

    Parameters
    ----------
    nbytes : int
        Size of the data in bytes.
    extra : int
        Additional projected working memory in bytes.

    Notes
    -----
    The ratio strategy needs psutil; without psutil the decision degrades to
    the absolute limit (``False`` when the limit is unset).
    """
    total = int(nbytes) + int(extra)

    if USE_RATIO_STRATEGY:
        available = get_available_memory()
        if available is not None:
            return total >= MEMMAP_RATIO_LIMIT * available
        # psutil unavailable: degrade to the absolute judgement.
        return (
            ABSOLUTE_MEMMAP_LIMIT_BYTES > 0
            and total >= ABSOLUTE_MEMMAP_LIMIT_BYTES
        )

    return total >= ABSOLUTE_MEMMAP_LIMIT_BYTES
