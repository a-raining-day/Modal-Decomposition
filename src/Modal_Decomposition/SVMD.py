"""
Successive Variational Mode Decomposition

Extracts modes one after another by minimizing a variational criterion
against the residual spectrum.

References
----------
10.1016/j.sigpro.2020.107610
"""

import numpy as np
from dataclasses import dataclass
from enum import Enum
from typing import ClassVar, Literal

from .Define import Config, Decomposer, DecompositionResult, DEFAULT_NUMPY_TYPE
from .Base import Cache
from .Error import RealizationError
from .Utils import Check, Check_Time_and_Signal

__all__ = ["SVMD", "SVMDConfig", "SVMD_BACKEND_MODS", "SVMD_BACKEND_MODS_ARGS", "f_SVMD"]

SVMD_BACKEND_MODS_ARGS = Literal["numpy", "numba"]
class SVMD_BACKEND_MODS(Enum):
    Numpy = "numpy"
    Numba = "numba"

"""
Config
"""
@dataclass(frozen=True, kw_only=True)
class SVMDConfig(Config):
    """
    Effective parameters of an SVMD run.
    """
    num_modes: int
    alpha: float
    tau: float
    tol: float
    max_iter: int
    backend: SVMD_BACKEND_MODS | SVMD_BACKEND_MODS_ARGS
    dtype: type | np.dtype | None


"""
Class and Function of decomposer
"""
class SVMD(Decomposer):
    name: ClassVar[str] = "SVMD"

    def __init__(
        self,
        num_modes: int = 3,
        alpha: float = 2000.0,
        tau: float = 0.0,
        tol: float = 1e-7,
        max_iter: int = 500,
        backend: SVMD_BACKEND_MODS_ARGS | SVMD_BACKEND_MODS = "numpy",
        dtype: type | np.dtype | None = None,
        **kwargs,
    ):
        """
        Parameters
        ----------
        num_modes : int
            Number of modes.
        alpha : float
            Compactness constraint.
        tau : float
            Dual-ascent step.
        tol : float
            Convergence tolerance.
        max_iter : int
            Maximum number of iterations.
        backend : SVMD_BACKEND_MODS_ARGS | SVMD_BACKEND_MODS
            Implementation backend, one of ``SVMD_BACKEND_MODS``. The numba
            backend is not yet implemented and raises RealizationError.
        dtype : type | np.dtype | None
            Working dtype of the returned modes. ``None`` keeps the input
            precision; integer/bool inputs are promoted to float64.
        """
        # 归一化口径与原实现一致: 先整数化, 再夹到 >= 1
        self.num_modes: int = max(1, int(num_modes))
        self.alpha: float = float(alpha)
        self.tau: float = float(tau)
        self.tol: float = float(tol)
        self.max_iter: int = int(max_iter)
        self.backend: SVMD_BACKEND_MODS | SVMD_BACKEND_MODS_ARGS = backend
        self.dtype: type | np.dtype | None = dtype

        unknown = set(kwargs) - {"safe", "copy"}
        if unknown:
            raise TypeError(f"unexpected keyword arguments: {unknown}")

        self.safe: str = kwargs.get("safe", "unknown")
        self.copy: bool = kwargs.get("copy", True)

        self.param_check()

        self.imfs: np.ndarray = None
        self.res: np.ndarray = None
        self._freqs: np.ndarray = None
        self._signal_hat: np.ndarray = None
        self._lambda_hat: np.ndarray = None
        self._modes_hat: list = None

    def param_check(self) -> None:
        if not isinstance(self.num_modes, int) or isinstance(self.num_modes, bool):
            raise TypeError("the type of `num_modes` must be int!")

        if self.num_modes < 1:
            raise ValueError("the `num_modes` must be >= 1")

        # 实例上一律存 Enum; 传字符串(或 Enum 本身)也在这里统一规范化
        self.backend = Check.validate_mod(self.backend, SVMD_BACKEND_MODS)

        if self.dtype is not None and np.dtype(self.dtype).kind != "f":
            raise ValueError("the `dtype` must be a float numpy dtype or None.")

        if not isinstance(self.copy, bool):
            raise TypeError("the `copy` must be bool")

        self.safe = Check.validate_safe(self.safe)

    def config_generate(self) -> SVMDConfig:
        return SVMDConfig(
            num_modes=self.num_modes,
            alpha=self.alpha,
            tau=self.tau,
            tol=self.tol,
            max_iter=self.max_iter,
            backend=self.backend,
            dtype=self.dtype,
        )

    def decompose(self, S, T=None) -> DecompositionResult:
        """
        Decompose the signal into modes and a residual.

        The time axis is accepted for API symmetry only; SVMD works in the
        frequency domain and never reads it.
        """
        scipy_fft = Cache.import_module("scipy.fft")

        # SVMD 是纯频域方法, 不使用时间轴; 用 need_T=False 省掉 arange(N) 那份临时数组
        S, N = Check_Time_and_Signal(S, T=None, ndim={1}, method=self.name,
                                     dtype=self.dtype, need_T=False, safe=self.safe, copy=self.copy)

        if not (S.dtype.kind == "f" and S.dtype.itemsize >= 4):  # int/bool 撑不住频域的浮点中间量, f16 精度不足
            S = Check.to_dtype(S, np.dtype(np.float64))

        # 每次调用都从干净状态开始: 否则第二次 decompose 会沿用上一次的谱/状态缓存
        self.imfs = None
        self.res = None
        self._freqs = None
        self._signal_hat = None
        self._lambda_hat = None
        self._modes_hat = None

        if self.backend is SVMD_BACKEND_MODS.Numba:
            # 原实现惰性 import 的 ._SVMD_numba 占位模块已不存在 (恒抛 RealizationError),
            # 这里直接抛出同一个异常, 不再制造 ModuleNotFoundError。
            raise RealizationError("the numba backend of SVMD is not implemented yet; use backend='numpy'")

        signal = np.array(S, dtype=DEFAULT_NUMPY_TYPE).ravel()
        if N < 8:
            raise ValueError("Signal length must be >= 8")

        K = self.num_modes
        eps = 1e-12

        self._freqs = scipy_fft.fftfreq(N, d=1.0)
        freqs = self._freqs

        self._signal_hat = scipy_fft.fft(signal)
        signal_hat = self._signal_hat

        self._lambda_hat = np.zeros(N, dtype=np.complex128)
        lambda_hat = self._lambda_hat

        modes = np.zeros((K, N), dtype=np.float64)
        modes_hat = [np.zeros(N, dtype=np.complex128) for _ in range(K)]
        self._modes_hat = modes_hat

        freqs_abs = np.abs(freqs)
        freq_range = freqs_abs.max() - freqs_abs.min()

        # 初值: 在 [0.1, 0.4) 上均匀取 K 个偏移量。K >= 4 时 arange 的步长取不满 K 个点
        # (K=4 只有 3 个), 会与下面的 omega[k] 逐点使用冲突, 这里回退到同一区间的 linspace。
        offsets = np.arange(0.1, 0.4, 0.4 / K)
        if offsets.size < K:
            offsets = np.linspace(0.1, 0.4, K, endpoint=False)

        omega = freqs_abs.min() + (offsets[:K] + 0.1) * freq_range

        denominators = []
        for k in range(K):
            diff = freqs - omega[k]
            denominators.append(1.0 + self.alpha * diff * diff)

        sum_modes_hat = np.zeros(N, dtype=np.complex128)
        recon_hat = np.zeros(N, dtype=np.complex128)

        prev_recon = np.zeros(N, dtype=np.float64)

        for iteration in range(self.max_iter):
            for k in range(K):
                # `sum_modes_hat` 是**全量**模态和(跨外层迭代增量维护), 因此
                # `sum_modes_hat - modes_hat[k]` 才是 "除第 k 个之外的模态和"。
                # 旧代码每次外层迭代都把它清零、且只累积 i<k, 于是减掉的是 k 自己的旧值,
                # 相当于把 +u_k^old 反馈回分子; 又因 tau=0 对偶项恒为 0, 没有纠偏力,
                # 幅值随迭代指数放大 (iters=500 时达 5e6 倍信号幅值)。
                u_k_old = modes_hat[k]
                numerator = signal_hat - (sum_modes_hat - u_k_old) + 0.5 * lambda_hat

                u_k_new = numerator / denominators[k]
                modes_hat[k] = u_k_new
                sum_modes_hat += u_k_new - u_k_old

                power_sq = np.abs(modes_hat[k]) ** 2

                omega_num = np.sum(freqs * power_sq)
                omega_den = np.sum(power_sq) + eps

                omega[k] = omega_num / omega_den

                diff = freqs - omega[k]
                denominators[k] = 1.0 + self.alpha * diff * diff

            np.copyto(recon_hat, sum_modes_hat)

            lambda_hat += self.tau * (signal_hat - recon_hat)

            recon_time = np.real(scipy_fft.ifft(recon_hat))

            error = np.linalg.norm(recon_time - prev_recon) / (np.linalg.norm(signal) + eps)

            if error < self.tol and iteration > 1:
                break

            np.copyto(prev_recon, recon_time)

        for k in range(K):
            modes[k] = np.real(scipy_fft.ifft(modes_hat[k]))

        # 输出回到工作 dtype (dtype=None 时即输入精度); 残差按**已落盘的** modes 求,
        # 于是 sum(IMFs) + Res 恒等于 S, 剩余误差只来自浮点加法本身。
        self.imfs = np.asarray(modes, dtype=S.dtype)
        self.res = np.asarray(signal - np.sum(self.imfs, axis=0, dtype=DEFAULT_NUMPY_TYPE), dtype=S.dtype)

        return DecompositionResult(
            IMFs=self.imfs,
            Res=self.res,
            info={},
            config=self.config_generate(),
        )


"""
other function for module
"""
def f_SVMD(S: list | np.ndarray, T=None, *, num_modes: int = 3, alpha: float = 2000.0, tau: float = 0.0, tol: float = 1e-07, max_iter: int = 500, backend: SVMD_BACKEND_MODS_ARGS | SVMD_BACKEND_MODS = "numpy", dtype: type | np.dtype | None = None, **kwargs) -> DecompositionResult:
    svmd = SVMD(
        num_modes=num_modes,
        alpha=alpha,
        tau=tau,
        tol=tol,
        max_iter=max_iter,
        backend=backend,
        dtype=dtype,
        **kwargs
    )

    return svmd.decompose(S, T)
