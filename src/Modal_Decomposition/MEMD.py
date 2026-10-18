"""
Multivariate Empirical Mode Decomposition

Projects a multichannel signal onto direction vectors sampled on a
hypersphere and averages the per-direction envelopes.

References
----------
10.48550/arXiv.2206.00926
"""

import numpy as np
from dataclasses import dataclass
from enum import Enum
from typing import ClassVar, Literal

from .Define import Config, Decomposer, DecompositionResult, DEFAULT_NUMPY_TYPE
from .Base import Cache
from .Utils import Check, Check_Time_and_Signal, monotonic

__all__ = ["MEMD", "MEMDConfig", "MEMD_SPLINE_MODS", "MEMD_SPLINE_MODS_ARGS", "f_MEMD"]

# `scipy.interpolate.interp1d` 支持的插值方式
MEMD_SPLINE_MODS_ARGS = Literal[
    "linear", "nearest", "zero", "slinear", "quadratic", "cubic", "previous", "next",
]
class MEMD_SPLINE_MODS(Enum):
    Linear = "linear"
    Nearest = "nearest"
    Zero = "zero"
    SLinear = "slinear"
    Quadratic = "quadratic"
    Cubic = "cubic"
    Previous = "previous"
    Next = "next"

"""
Config
"""
@dataclass(frozen=True, kw_only=True)
class MEMDConfig(Config):
    """
    Effective parameters of a MEMD run.

    ``d`` / ``k`` / ``max_imf`` hold the effective values of the last run,
    i.e. ``None`` arguments are already resolved from the signal shape.
    """
    d: int | None
    k: int | None
    max_imf: int | None
    sd_thresh: float
    max_iter: int
    spline_kind: MEMD_SPLINE_MODS | MEMD_SPLINE_MODS_ARGS
    dtype: type | np.dtype | None


"""
Class and Function of decomposer
"""
class MEMD(Decomposer):
    name: ClassVar[str] = "MEMD"

    def __init__(
        self,
        d: int | None = None,
        k: int | None = None,
        max_imf: int | None = None,
        sd_thresh: float = 0.2,
        max_iter: int = 10,
        spline_kind: MEMD_SPLINE_MODS_ARGS | MEMD_SPLINE_MODS = "linear",
        dtype: type | np.dtype | None = None,
        **kwargs,
    ):
        """
        Parameters
        ----------
        d : int | None
            Number of channels; None infers it from the signal.
        k : int | None
            Number of directional vectors; None selects d * 128.
        max_imf : int | None
            Maximum number of IMFs; None selects log2(N).
        sd_thresh : float
            Sifting stopping threshold.
        max_iter : int
            Maximum sifting iterations per IMF.
        spline_kind : MEMD_SPLINE_MODS | MEMD_SPLINE_MODS_ARGS
            Interpolation kind for envelope fitting (``interp1d`` kind).
        dtype : type | np.dtype | None
            Working dtype of the returned IMFs. ``None`` keeps the input
            precision; integer/bool inputs are promoted to float64.
        """
        self.d: int | None = d
        self.k: int | None = k
        self.max_imf: int | None = max_imf
        self.sd_thresh: float = sd_thresh
        self.max_iter: int = max_iter
        self.spline_kind: MEMD_SPLINE_MODS | MEMD_SPLINE_MODS_ARGS = spline_kind
        self.dtype: type | np.dtype | None = dtype

        unknown = set(kwargs) - {"safe", "copy"}
        if unknown:
            raise TypeError(f"unexpected keyword arguments: {unknown}")

        self.safe: str = kwargs.get("safe", "unknown")
        self.copy: bool = kwargs.get("copy", True)

        # 运行态: 每次 decompose 开头都会重置
        self.imfs: np.ndarray | None = None
        self.res: np.ndarray | None = None
        self._effective: tuple[int, int, int] | None = None

        self.param_check()

    def param_check(self) -> None:
        if self.d is not None:
            if not isinstance(self.d, int) or isinstance(self.d, bool):
                raise TypeError("The type of the d must be int or None!")

            if self.d < 1:
                raise ValueError("the `d` must be >= 1 or None (inferred from the signal)")

        if self.k is not None:
            if not isinstance(self.k, int) or isinstance(self.k, bool):
                raise TypeError("The type of the k must be int or None!")

            if self.k < 1:
                raise ValueError("the `k` must be >= 1 or None (defaults to d * 128)")

        if self.max_imf is not None:
            if not isinstance(self.max_imf, int) or isinstance(self.max_imf, bool):
                raise TypeError("The type of the max_imf must be int or None!")

            if self.max_imf < 1:
                raise ValueError("the `max_imf` must be >= 1 or None (defaults to log2(N))")

        if not isinstance(self.sd_thresh, (int, float)) or isinstance(self.sd_thresh, bool):
            raise TypeError("the `sd_thresh` must be int or float")

        if not (self.sd_thresh > 0):
            raise ValueError("the `sd_thresh` must be > 0")

        if not isinstance(self.max_iter, int) or isinstance(self.max_iter, bool):
            raise TypeError("the `max_iter` must be int")

        if self.max_iter < 1:
            raise ValueError("the `max_iter` must be >= 1")

        self.spline_kind = Check.validate_mod(self.spline_kind, MEMD_SPLINE_MODS)

        if self.dtype is not None and np.dtype(self.dtype).kind != "f":
            raise ValueError("the `dtype` must be a float numpy dtype or None.")

        if not isinstance(self.copy, bool):
            raise TypeError("the `copy` must be bool")

        self.safe = Check.validate_safe(self.safe)

    def config_generate(self) -> MEMDConfig:
        if self._effective is None:
            d, k, max_imf = self.d, self.k, self.max_imf
        else:
            d, k, max_imf = self._effective

        return MEMDConfig(
            d=d,
            k=k,
            max_imf=max_imf,
            sd_thresh=self.sd_thresh,
            max_iter=self.max_iter,
            spline_kind=self.spline_kind,
            dtype=self.dtype,
        )

    def decompose(self, S, T=None) -> DecompositionResult:
        """
        Decompose the multichannel signal into IMFs and a residual.

        A 1-D signal is treated as a single channel ``(1, N)`` and keeps the
        univariate ``(k, N)`` layout on the way out; a genuine 2-D input
        ``(d, N)`` -- channels on the first axis -- returns ``(k, d, N)``.
        The now axis is validated but not used by the algorithm.
        """
        # 每次调用都从干净状态开始 (校验失败也不留下上一轮的结果)
        self.imfs = None
        self.res = None
        self._effective = None

        # MEMD 是多通道方法: 1-D 单通道与 2-D (channels, N) 都合法, 不能因 2-D 输入报错
        S, N = Check_Time_and_Signal(S, T=T, ndim={1, 2}, method=self.name,
                                     dtype=self.dtype, need_T=False,
                                     safe=self.safe, copy=self.copy)

        # f16 会让样条出非有限值, int/bool 连 `h - mean` 都做不了
        if not (S.dtype.kind == "f" and S.dtype.itemsize >= 4):
            S = Check.to_dtype(S, np.dtype(DEFAULT_NUMPY_TYPE))

        single_channel: bool = S.ndim == 1
        if single_channel:
            S = S.reshape(1, -1)  # 统一按 (channels, now) 计算

        d_infer = S.shape[0]

        d = d_infer if self.d is None else self.d
        if d != d_infer:
            raise ValueError(f"Declared d={d} but signal has shape {S.shape}")

        k = d * 128 if self.k is None else self.k
        max_imf = int(np.log2(N)) if self.max_imf is None else self.max_imf

        T_axis = np.arange(N, dtype=np.float64)

        vectors = _generate_hammersley_points(k, d)  # shape: (d, k)

        imfs = []
        residue = S.copy()

        for imf_idx in range(max_imf):
            h = residue.copy()

            for iter_num in range(self.max_iter):
                h_old = h
                mean_envelope = _compute_local_mean(h, vectors, T_axis, spline_kind=self.spline_kind)

                h_new = h - mean_envelope
                denominator = np.sum(h_old ** 2)
                # 全零残差: 分子分母同为 0 -> 已无变化, 视为收敛, 不制造 0/0 警告
                sd = 0.0 if denominator <= 0 else np.sum((h_old - h_new) ** 2) / denominator
                h = h_new

                if sd < self.sd_thresh or iter_num == self.max_iter - 1:
                    break

            imfs.append(h.copy())

            residue = residue - h

            if _should_stop(residue):
                break

        if imfs:
            imfs_array = np.asarray(imfs, dtype=S.dtype)  # shape: (n_imfs, d, N)
        else:
            imfs_array = np.empty((0, d, N), dtype=S.dtype)

        if single_channel:  # 单通道输入按一元布局 (k, N) / (N,) 返回
            imfs_array = imfs_array.reshape(imfs_array.shape[0], N)
            residue = residue.reshape(N)

        self.imfs = imfs_array
        self.res = np.asarray(residue, dtype=S.dtype)
        self._effective = (d, k, max_imf)

        return DecompositionResult(
            IMFs=self.imfs,
            Res=self.res,
            info={},
            config=self.config_generate(),
        )


def _generate_hammersley_points(k, d):
    vectors = np.zeros((d, k))

    primes = _generate_primes(d - 1)

    for i in range(k):
        point = np.zeros(d)

        point[0] = _radical_inverse_vdc(i)

        for j in range(1, d):
            base = primes[j - 1] if j - 1 < len(primes) else primes[-1]
            point[j] = _radical_inverse(i, base)

        norm = np.linalg.norm(point)
        if norm > 0:
            point = point / norm

        vectors[:, i] = point

    return vectors


def _radical_inverse_vdc(index):
    """Van der Corput."""
    bits = index
    bits = (bits << 16) | (bits >> 16)
    bits = ((bits & 0x55555555) << 1) | ((bits & 0xAAAAAAAA) >> 1)
    bits = ((bits & 0x33333333) << 2) | ((bits & 0xCCCCCCCC) >> 2)
    bits = ((bits & 0x0F0F0F0F) << 4) | ((bits & 0xF0F0F0F0) >> 4)
    bits = ((bits & 0x00FF00FF) << 8) | ((bits & 0xFF00FF00) >> 8)
    return float(bits) / 2 ** 32


def _radical_inverse(index, base):
    result = 0.0
    f = 1.0 / base
    i = index
    while i > 0:
        result += f * (i % base)
        i = i // base
        f = f / base
    return result


def _generate_primes(n):
    if n <= 0:
        return []

    primes = []
    num = 2
    while len(primes) < n:
        is_prime = True
        for p in primes:
            if p * p > num:
                break
            if num % p == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(num)
        num += 1

    return primes


def _compute_local_mean(signal, vectors, T, spline_kind: MEMD_SPLINE_MODS_ARGS | MEMD_SPLINE_MODS = "linear"):
    """
    signal: now (d, N)
    vectors: directional vectors (d, k)
    T: now axis
    """
    spline_kind = Check.validate_mod(spline_kind, MEMD_SPLINE_MODS)

    find_peaks = Cache.import_module("scipy.signal").find_peaks
    interp1d = Cache.import_module("scipy.interpolate").interp1d

    d, N = signal.shape
    k = vectors.shape[1]

    current_projections = np.dot(signal.T, vectors)

    mean_envelope = np.zeros((d, N))

    for dir_idx in range(k):
        proj = current_projections[:, dir_idx]

        max_indices, _ = find_peaks(proj)
        min_indices, _ = find_peaks(-proj)

        if len(max_indices) == 0:
            max_indices = np.array([0, N - 1])
        else:
            if max_indices[0] != 0:
                max_indices = np.concatenate([[0], max_indices])
            if max_indices[-1] != N - 1:
                max_indices = np.concatenate([max_indices, [N - 1]])

        if len(min_indices) == 0:
            min_indices = np.array([0, N - 1])
        else:
            if min_indices[0] != 0:
                min_indices = np.concatenate([[0], min_indices])
            if min_indices[-1] != N - 1:
                min_indices = np.concatenate([min_indices, [N - 1]])

        for ch in range(d):
            max_values = signal[ch, max_indices]
            if len(max_indices) >= 4:
                try:
                    max_spline = interp1d(max_indices, max_values, kind=spline_kind.value, fill_value='extrapolate')
                    max_env = max_spline(T)
                except Exception:
                    max_env = np.interp(T, max_indices, max_values)
            else:
                max_env = np.interp(T, max_indices, max_values)

            min_values = signal[ch, min_indices]
            if len(min_indices) >= 4:
                try:
                    min_spline = interp1d(min_indices, min_values, kind=spline_kind.value, fill_value='extrapolate')
                    min_env = min_spline(T)
                except Exception:
                    min_env = np.interp(T, min_indices, min_values)
            else:
                min_env = np.interp(T, min_indices, min_values)

            mean_envelope[ch] += (max_env + min_env) / 2

    mean_envelope = mean_envelope / k

    return mean_envelope


def _should_stop(residue):
    find_peaks = Cache.import_module("scipy.signal").find_peaks

    d, N = residue.shape

    for ch in range(d):
        if monotonic(residue[ch]):
            return True

        max_indices, _ = find_peaks(residue[ch])
        min_indices, _ = find_peaks(-residue[ch])
        if len(max_indices) + len(min_indices) <= 2:
            return True

    return False


"""
other function for module
"""
def f_MEMD(S: list | np.ndarray, T=None, *, d: int | None = None, k: int | None = None, max_imf: int | None = None, sd_thresh: float = 0.2, max_iter: int = 10, spline_kind: MEMD_SPLINE_MODS_ARGS | MEMD_SPLINE_MODS = "linear", dtype: type | np.dtype | None = None, **kwargs) -> DecompositionResult:
    memd = MEMD(
        d=d,
        k=k,
        max_imf=max_imf,
        sd_thresh=sd_thresh,
        max_iter=max_iter,
        spline_kind=spline_kind,
        dtype=dtype,
        **kwargs
    )

    return memd.decompose(S, T)
