"""
Cyclic Envelop Empirical Fourier Decomposition

Segments the spectral envelope at its peaks and reconstructs modes from
the segmented spectrum.

算法
----
1. 对信号做 FFT, 取单边幅度谱 ``|F[0:N//2+1]|``;
2. 对幅度谱做 ``envelop_iter`` 轮"盒式平滑 + 取上包络"迭代, 得到**谱包络**;
3. 在包络上用 ``find_peaks`` 找峰 (峰间距不小于 ``min_peak_distance``);
4. 相邻峰的中点作为频带边界, 首尾补 ``0`` 与 ``N//2`` (DC 与 Nyquist);
5. 每个频带构造成 0/1 掩码 (正频带 + 其共轭对称的负频), 逐频带反变换得到模态;
6. 残差 = 信号 − Σ模态 (精确重构)。

频带约定 (bin convention)
------------------------
每个频带取**闭区间** ``[a, b]`` (含 ``b``), 最后一个频带含 Nyquist 端点 ——
这样各频带恰好覆盖 ``0 … N//2`` 的每个 bin 一次, 既无重复也无遗漏, 于是
``Σ模态 + Res == S`` 在数值精度内精确成立。

References
----------
10.3969/j.issn.1001-4551.2023.07.001
"""

import numpy as np
from dataclasses import dataclass
from typing import ClassVar

from .Define import Config, Decomposer, DecompositionResult, DEFAULT_NUMPY_TYPE
from .Base import Cache
from .Utils import Check, Check_Time_and_Signal

__all__ = ["CEEFD", "CEEFDConfig", "f_CEEFD"]

#: 谱包络平滑窗占单边谱长度的比例 (原实现写死 0.05)。
_ENVELOP_WINDOW_FRACTION = 0.05


"""
Config
"""
@dataclass(frozen=True, kw_only=True)
class CEEFDConfig(Config):
    """
    Effective parameters of a CEEFD run.
    """
    fs: int | float
    min_peak_distance: int
    envelop_iter: int
    dtype: type | np.dtype | None


"""
Class and Function of decomposer
"""
class CEEFD(Decomposer):
    name: ClassVar[str] = "CEEFD"

    def __init__(
        self,
        fs: int | float = 1.0,
        min_peak_distance: int = 10,
        envelop_iter: int = 3,
        dtype: type | np.dtype | None = None,
        **kwargs,
    ):
        """
        Parameters
        ----------
        fs : int | float
            Sampling frequency (> 0). ``CEEFD`` 本身在**归一化频率**上分割
            (边界就是 ``0 … N//2`` 的 bin 下标), 故 ``fs`` 不参与分割; 它只用于
            把边界换算成物理频率 (Hz) 供诊断使用 —— 见 ``info["boundaries_hz"]``。
        min_peak_distance : int
            Minimum bin distance between envelope peaks (>= 1); 1 disables the
            distance filter.
        envelop_iter : int
            Number of envelope smoothing iterations (>= 0); 0 returns the raw
            magnitude spectrum as the envelope.
        dtype : type | np.dtype | None
            Working dtype of the returned modes. ``None`` keeps the input
            precision; integer/bool inputs are promoted to float64.
        kwargs
            ``safe`` (输入校验强度) 与 ``copy`` (校验时是否复制调用方输入) 的透传;
            ``copy=True`` (默认) 保证绝不改写调用方数组。

        Raises
        ------
        ValueError
            On a non-positive / non-finite ``fs``, a ``min_peak_distance`` below
            1, a negative ``envelop_iter``, or a non-float ``dtype``.
        TypeError
            On an unknown keyword argument or a non-bool ``copy``.
        """
        self.fs: int | float = fs
        self.min_peak_distance: int = min_peak_distance
        self.envelop_iter: int = envelop_iter
        self.dtype: type | np.dtype | None = dtype

        unknown = set(kwargs) - {"safe", "copy"}
        if unknown:
            raise TypeError(f"unexpected keyword arguments: {unknown}")

        self.safe: str = kwargs.get("safe", "unknown")
        self.copy: bool = kwargs.get("copy", True)

        self.param_check()

        # decompose 每次都会重置; 先建好属性, 免得未分解时读取抛 AttributeError
        self.imfs: np.ndarray = None
        self.res: np.ndarray = None

    # ------------------------------------------------------------------ #
    # params
    # ------------------------------------------------------------------ #
    def param_check(self) -> None:
        if isinstance(self.fs, bool) or not isinstance(self.fs, (int, float, np.floating)):
            raise ValueError(f"fs must be a number > 0, got {self.fs!r}")
        fs = float(self.fs)
        if not np.isfinite(fs) or fs <= 0.0:
            raise ValueError(f"fs must be positive and finite, got {self.fs!r}")
        self.fs = fs

        if isinstance(self.min_peak_distance, bool) or not isinstance(
            self.min_peak_distance, (int, np.integer)
        ):
            raise ValueError(f"min_peak_distance must be an int, got {self.min_peak_distance!r}")
        min_peak_distance = int(self.min_peak_distance)
        if min_peak_distance < 1:
            raise ValueError(
                f"min_peak_distance must be >= 1, got {self.min_peak_distance!r}"
            )
        self.min_peak_distance = min_peak_distance

        if isinstance(self.envelop_iter, bool) or not isinstance(self.envelop_iter, (int, np.integer)):
            raise ValueError(f"envelop_iter must be an int, got {self.envelop_iter!r}")
        envelop_iter = int(self.envelop_iter)
        if envelop_iter < 0:
            raise ValueError(f"envelop_iter must be >= 0, got {self.envelop_iter!r}")
        self.envelop_iter = envelop_iter

        if self.dtype is not None and np.dtype(self.dtype).kind != "f":
            raise ValueError("the `dtype` must be a float numpy dtype or None.")

        if not isinstance(self.copy, bool):
            raise TypeError("the `copy` must be bool")

    def config_generate(self) -> CEEFDConfig:
        return CEEFDConfig(
            fs=self.fs,
            min_peak_distance=self.min_peak_distance,
            envelop_iter=self.envelop_iter,
            dtype=self.dtype,
        )

    # ------------------------------------------------------------------ #
    # decompose
    # ------------------------------------------------------------------ #
    def decompose(self, S, T=None) -> DecompositionResult:
        """
        Decompose the signal into IMFs and a residual.

        Returns
        -------
        DecompositionResult
            ``IMFs`` (K, N) —— 由低到高排列的频带模态; ``Res`` (N,) 为精确余量
            (``ΣIMFs + Res == S``); ``info`` 含 ``boundaries`` (bin 下标)、
            ``boundaries_hz`` (物理频率)、``peaks``、``envelope``、``n_bands``。
            无峰 (或所有频带都过窄) 时 ``IMFs`` 为空 ``(0, N)``、``Res == S``。
        """
        scipy_signal = Cache.import_module("scipy.signal")

        # CEEFD 是纯频域方法, 不使用时间轴; 用 need_T=False 省掉 arange(N) 那份临时数组
        S, N = Check_Time_and_Signal(S, T=None, ndim={1}, method=self.name,
                                     dtype=self.dtype, need_T=False, safe=self.safe, copy=self.copy)

        if not (S.dtype.kind == "f" and S.dtype.itemsize >= 4):
            S = Check.to_dtype(S, np.dtype(DEFAULT_NUMPY_TYPE))

        # 每次调用都从干净状态开始: 实例上不留上一次的模态/残差
        self.imfs = None
        self.res = None

        out_dtype = S.dtype

        fft_signal = np.fft.fft(S)
        half = N // 2 + 1
        mag_spectrum = np.abs(fft_signal[:half])

        envelope = self._compute_spectral_envelope(mag_spectrum)
        peaks, _properties = scipy_signal.find_peaks(envelope, distance=self.min_peak_distance)

        # 无峰: 无法分割 —— 按库内契约返回 0 个模态 + 整条信号作为残差。
        if peaks.size == 0:
            self.imfs = np.empty((0, N), dtype=out_dtype)
            self.res = np.array(S, dtype=out_dtype, copy=True)
            return DecompositionResult(
                IMFs=self.imfs,
                Res=self.res,
                info={
                    "boundaries": np.array([0, half - 1], dtype=np.int64),
                    "boundaries_hz": np.array([0.0, self.fs / 2.0]),
                    "peaks": peaks,
                    "envelope": envelope,
                    "mag_spectrum": mag_spectrum,
                    "n_bands": 0,
                },
                config=self.config_generate(),
            )

        # 边界: 首 = DC, 内部 = 相邻峰中点, 末 = Nyquist。整型下标。
        boundaries = np.empty(peaks.size + 1, dtype=np.int64)
        boundaries[0] = 0
        boundaries[-1] = half - 1
        if peaks.size > 1:
            boundaries[1:-1] = (peaks[1:] + peaks[:-1]) // 2

        # 逐频带重建。频带取闭区间 [a, b], 故各带恰好覆盖 0…half-1 一次。
        modes: list = []
        for i in range(boundaries.size - 1):
            a = int(boundaries[i])
            b = int(boundaries[i + 1])
            if b - a < 1:
                # 宽度不足 2 个 bin, 掩码会退化 —— 跳过该带 (其能量留在 Res)。
                continue
            modes.append(self._extract_mode(fft_signal, a, b))

        if modes:
            # 模态本身在 float64 上重建, 末端再回到工作精度
            modes_f64 = np.asarray(modes, dtype=DEFAULT_NUMPY_TYPE)
            IMFs = modes_f64.astype(out_dtype, copy=False)
            # 残差由总和反推: 精确重构硬性成立 (数值精度内)。
            Res = (np.asarray(S, dtype=DEFAULT_NUMPY_TYPE) - modes_f64.sum(axis=0)).astype(out_dtype, copy=False)
        else:
            IMFs = np.empty((0, N), dtype=out_dtype)
            Res = np.array(S, dtype=out_dtype, copy=True)

        self.imfs = IMFs
        self.res = Res

        info = {
            "boundaries": boundaries,
            "boundaries_hz": boundaries * (self.fs / N),
            "peaks": peaks,
            "envelope": envelope,
            "mag_spectrum": mag_spectrum,
            "n_bands": IMFs.shape[0],
        }

        return DecompositionResult(
            IMFs=IMFs,
            Res=Res,
            info=info,
            config=self.config_generate(),
        )

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    def _compute_spectral_envelope(self, mag_spectrum: np.ndarray) -> np.ndarray:
        """
        谱包络: ``envelop_iter`` 轮"盒式平滑 + 取上包络"。

        窗长按单边谱长度的 ``_ENVELOP_WINDOW_FRACTION`` 计算, 但**钳到至少 1 且必为
        整数** —— 原实现直接用 ``int(0.05 * n)``: ``n < 20`` 时得到长度 0 的空窗
        (``np.convolve`` 抛 ``ValueError: v cannot be empty``), 某些 n 下
        ``0.05 * n`` 还带浮点误差使 ``int()`` 后仍非整型 (抛 ``TypeError``)。
        """
        scipy_signal = Cache.import_module("scipy.signal")

        n = int(mag_spectrum.size)
        window_size = max(1, int(_ENVELOP_WINDOW_FRACTION * n))
        window = scipy_signal.windows.boxcar(window_size)
        window = window / window.sum()          # 归一化, 与旧实现的 /window_size 等价

        envelope = np.array(mag_spectrum, dtype=np.float64, copy=True)
        for _ in range(self.envelop_iter):
            envelope = np.convolve(envelope, window, mode="same")
            envelope = np.maximum(envelope, mag_spectrum)
        return envelope

    @staticmethod
    def _extract_mode(fft_signal: np.ndarray, first: int, last: int) -> np.ndarray:
        """
        取 ``[first, last]`` (闭区间, 正频 bin) 这一段频谱重建一个模态。

        负频由共轭对称补齐; DC 与 Nyquist 是自共轭点, 只计一次 (``(N-k) % N``
        在 ``k=0`` 与 ``k=N//2`` 时回落到自身, 天然不重复)。
        """
        N = fft_signal.size
        bins = np.arange(first, last + 1, dtype=np.int64)
        mask = np.zeros(N, dtype=bool)
        mask[bins] = True
        mask[(N - bins) % N] = True
        return np.fft.ifft(np.where(mask, fft_signal, 0.0)).real


"""
other function for module
"""
def f_CEEFD(S: list | np.ndarray, T=None, *, fs: int | float = 1.0, min_peak_distance: int = 10, envelop_iter: int = 3, dtype: type | np.dtype | None = None, **kwargs) -> DecompositionResult:
    ceefd = CEEFD(
        fs=fs,
        min_peak_distance=min_peak_distance,
        envelop_iter=envelop_iter,
        dtype=dtype,
        **kwargs
    )

    return ceefd.decompose(S, T)
