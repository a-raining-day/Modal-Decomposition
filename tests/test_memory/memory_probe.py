"""
Memory probing utilities for the decomposition memory-test pipeline.

* :class:`PeakMonitor` samples the process RSS while a callable runs and
  reports baseline / peak / delta RSS together with the wall time. Sampling
  is coarse (a running thread), so very short calls (< ~2 ms) may under-report
  a transient peak -- keep measured runs comfortably above that.
* :func:`array_info` records shape / dtype / nbytes / backing of any array.
* :func:`monotonic_pattern` builds deterministic 1-d float64 inputs whose
  shape is strictly increasing, strictly decreasing or random.
"""

from __future__ import annotations

import os
import tempfile
import threading
import time
from typing import Callable, Optional

import numpy as np
import psutil

__all__ = [
    "process_rss",
    "PeakMonitor",
    "array_info",
    "monotonic_pattern",
    "dtype_of",
    "DEFAULT_DTYPES",
    "PATTERNS",
    "DEFAULT_SIZES_BYTES",
]

PATTERNS = ("increasing", "random")

DEFAULT_DTYPES = ("float16", "float32", "float64")

# Data sizes (bytes) under test, as requested: {1MB 20MB 100MB 500MB 1GB 3GB}.
DEFAULT_SIZES_BYTES = (
    1 * 1024 ** 2,
    20 * 1024 ** 2,
    100 * 1024 ** 2,
    500 * 1024 ** 2,
    1 * 1024 ** 3,
    3 * 1024 ** 3,
)

# Generation settings: below this size signals are built in RAM; larger ones
# are streamed into a temporary memmap so building them never exhausts RAM.
_RAM_BUILD_LIMIT_BYTES = 256 * 1024 * 1024
_GEN_CHUNK_ELEMS = 4 * 1024 * 1024

_ITEMSIZE = {
    "float16": 2,
    "float32": 4,
    "float64": 8,
}


def dtype_of(name: str) -> np.dtype:
    """Map a dtype name ("float16"/"float32"/"float64") to a numpy dtype."""
    try:
        return np.dtype(name)
    except TypeError:
        raise ValueError(f"unsupported dtype {name!r}; use one of {DEFAULT_DTYPES}") from None


def process_rss(pid: Optional[int] = None) -> int:
    """Current RSS of ``pid`` (default: this process) in bytes."""
    return int(psutil.Process(pid if pid is not None else os.getpid()).memory_info().rss)


def _value_at(i: int, n: int, pattern: str) -> float:
    """Deterministic sample value for index ``i`` of length ``n``."""
    if n <= 1:
        return 0.0
    x = -1.0 + 2.0 * i / (n - 1)          # increasing in [-1, 1]
    if pattern == "increasing":
        return x
    if pattern == "decreasing":
        return -x
    raise ValueError(f"pattern {pattern!r} is not monotonic; use increasing/random")


def monotonic_pattern(pattern: str, size_bytes: int, dtype: str = "float64", seed: int = 0) -> np.ndarray:
    """
    Build a deterministic input signal of the requested size and dtype.

    Parameters
    ----------
    pattern : {"increasing", "random"}
        ``increasing``: strictly monotone ramp. ``random``: seeded noise.
    size_bytes : int
        Total size of the array in bytes (n = size_bytes // itemsize).
    dtype : {"float16", "float32", "float64"}
        Element dtype.
    seed : int
        Seed for the random pattern.

    Returns
    -------
    np.ndarray
        Float array (``np.memmap`` for large sizes) of length ``n``.
    """
    if pattern not in PATTERNS:
        raise ValueError(f"unknown pattern {pattern!r}, expected one of {PATTERNS}")

    dt = dtype_of(dtype)
    itemsize = dt.itemsize
    if size_bytes <= 0:
        raise ValueError(f"size_bytes must be positive, got {size_bytes}")
    n = int(size_bytes) // itemsize
    if n == 0:
        raise ValueError(
            f"size_bytes {size_bytes} is smaller than one {dt} element"
        )

    if size_bytes <= _RAM_BUILD_LIMIT_BYTES:
        return _build_ram(pattern, n, dt, seed)

    return _build_memmap(pattern, n, dt, seed)


def _build_ram(pattern: str, n: int, dt: np.dtype, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed) if pattern == "random" else None
    if rng is not None:
        return rng.standard_normal(n, dtype=np.float64).astype(dt)
    # monotone ramp: linspace in float64 then cast (float16-safe, no ties)
    return np.linspace(-1.0, 1.0, n).astype(dt)


def _build_memmap(pattern: str, n: int, dt: np.dtype, seed: int) -> np.memmap:
    """Stream the signal into a temporary memmap so peak RAM stays bounded."""
    fd, path = tempfile.mkstemp(prefix="md_signal_", suffix=".dat")
    os.close(fd)
    out = np.memmap(path, dtype=dt, mode="w+", shape=(n,))

    if pattern == "random":
        rng = np.random.default_rng(seed)  # sequential draws are deterministic
        for start in range(0, n, _GEN_CHUNK_ELEMS):
            end = min(start + _GEN_CHUNK_ELEMS, n)
            out[start:end] = rng.standard_normal(end - start).astype(dt)
    else:  # monotone ramp; compute per chunk in float64 to keep the math exact
        for start in range(0, n, _GEN_CHUNK_ELEMS):
            end = min(start + _GEN_CHUNK_ELEMS, n)
            idx = np.arange(start, end)
            out[start:end] = (-1.0 + 2.0 * idx / (n - 1)).astype(dt)
    out.flush()
    return out


class PeakMonitor:
    """
    Measure baseline / peak / delta RSS and wall time of a callable.

    Attributes
    ----------
    baseline, peak, delta : int
        Bytes of RSS before the call and at its observed maximum; ``delta``
        is ``peak - baseline``.
    wall_s : float
        Wall-clock seconds of the measured call.
    n_samples : int
        Number of RSS samples taken.
    """

    def __init__(self, interval_s: float = 0.001):
        self.interval_s = interval_s
        self.baseline: int = 0
        self.peak: int = 0
        self.wall_s: float = 0.0
        self.n_samples: int = 0

    @property
    def delta(self) -> int:
        return self.peak - self.baseline

    def measure(self, fn: Callable[[], object]) -> object:
        """
        Run ``fn`` while sampling RSS; returns ``fn()``'s result.
        """
        baseline = process_rss()
        samples: list[int] = []
        stop = threading.Event()

        def _sample() -> None:
            while not stop.is_set():
                samples.append(process_rss())
                stop.wait(self.interval_s)

        sampler = threading.Thread(target=_sample, daemon=True)
        start = time.perf_counter()
        sampler.start()
        try:
            result = fn()
        finally:
            self.wall_s = time.perf_counter() - start
            stop.set()
            sampler.join()
            # trailing samples catch allocations that linger after the call
            samples.append(process_rss())
            samples.append(process_rss())

        self.baseline = baseline
        self.peak = max(samples) if samples else baseline
        self.n_samples = len(samples)
        return result


def array_info(arr) -> Optional[dict]:
    """
    Introspect an array-like (or ``None``) into a serializable dict.

    Returns
    -------
    dict or None
        ``None`` for a ``None`` input.
    """
    if arr is None:
        return None
    # np.asanyarray preserves the memmap subclass (np.asarray strips it).
    a = np.asanyarray(arr)
    return {
        "shape": list(a.shape),
        "dtype": str(a.dtype),
        "nbytes": int(a.nbytes),
        "backing": "memmap" if isinstance(a, np.memmap) else "ndarray",
    }
