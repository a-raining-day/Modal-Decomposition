"""
Filtered Mode Decomposition (FMD)。

迭代更新一个有限脉冲响应 (FIR) 滤波器, 使周期峭度 (periodic kurtosis) 准则
最大化, 据此逐阶提取窄带模态; 每阶模态确定后从残差中扣除, 直到模态数达到上限、
残差能量足够小或残差极值不足。

References
----------
10.1109/TIE.2022.3156156
"""

import numpy as np
import warnings
from dataclasses import dataclass
from typing import ClassVar

from .Define import Config, Decomposer, DecompositionResult, DEFAULT_NUMPY_TYPE, SAFE_MODS, SAFE_ARGS, check_output_memmap, finalize_output
from .Base import Cache, Seed, Global_Seed
from .Utils import Check, Check_Time_and_Signal

__all__ = ["FMD", "FMDConfig", "f_FMD"]


"""
配置
"""
@dataclass(frozen=True, kw_only=True)
class FMDConfig(Config):
    """
    一次 FMD 运行的有效参数快照 (实例参数校验后固化, 供结果回查)。
    """
    M: int | None
    L_factor: int
    K: int
    max_iter: int
    tol: float
    num_hand: int | None
    seed: int | None
    dtype: type | np.dtype | None
    solver: str


"""
分解器类与入口函数
"""
class FMD(Decomposer):
    name: ClassVar[str] = "FMD"

    def __init__(
        self,
        M: int | None = None,
        L_factor: int = 2,
        K: int = 10,
        max_iter: int = 50,
        tol: float = 1e-6,
        num_hand: int | None = None,
        seed: int | None = None,
        dtype: type | np.dtype | None = None,
        solver: str = "ck",
        **kwargs
    ):
        """
        Parameters
        ----------
        M : int | None
            目标模态的周期; None 时自动估计。
        L_factor : int
            滤波器长度系数: L = L_factor * M（默认 2）。Gram 矩阵组装 O(L^2·K) 与
            广义特征分解 O(L^3) 都随 L 增长，所以这个系数是 FMD 的主要成本旋钮；
            2 是 A/B/C 三例时间-质量扫描后的取值（见
            `docs/test/methods/FMD_Native_Engine_Report.md`）。
        K : int
            模态数上限; -1 表示完全分解 (上限取 ``log2(N) + 2``)。
        max_iter : int
            每阶滤波器的最大迭代次数。
        tol : float
            滤波器更新的收敛容差 (本次与上次系数差的范数)。
        num_hand : int | None
            随机初始滤波器的候选个数; None 时按信噪比 / 复杂度自动估计。
        seed : int | None
            本地随机种子; 与全局种子同时设置时由全局种子覆盖。
        dtype : type | np.dtype | None
            返回模态的工作 dtype。None 沿用输入精度; int / bool 输入提升为 float64。
        """
        self.M: int | None = M
        self.L_factor: int = L_factor
        self.K: int = K
        self.max_iter: int = max_iter
        self.tol: float = tol
        self.num_hand: int | None = num_hand
        self.seed: int | None = seed
        self.dtype: type | np.dtype | None = dtype
        self.solver: str = solver

        # 未知参数白名单: 只接受 safe / copy
        unknown = set(kwargs) - {"safe", "copy"}
        if unknown:
            raise TypeError(f"unexpected keyword arguments: {unknown}")

        self.safe: SAFE_MODS | SAFE_ARGS = kwargs.get("safe", "unknown")
        self.copy: bool = kwargs.get("copy", True)

        self.param_check()

    def param_check(self) -> None:
        if self.M is not None:
            if not isinstance(self.M, int) or isinstance(self.M, bool):
                raise TypeError("the `M` must be int or None.")

            if self.M < 1:
                raise ValueError("the `M` must >= 1.")

        if not isinstance(self.L_factor, int) or isinstance(self.L_factor, bool):
            raise TypeError("the `L_factor` must be int.")

        if self.L_factor < 1:
            raise ValueError("the `L_factor` must >= 1.")

        if not isinstance(self.K, int) or isinstance(self.K, bool):
            raise TypeError("the `K` must be int.")

        if self.K != -1 and self.K < 1:
            raise ValueError("the `K` must be -1 or >= 1.")

        if not isinstance(self.max_iter, int) or isinstance(self.max_iter, bool):
            raise TypeError("the `max_iter` must be int.")

        if self.max_iter < 1:
            raise ValueError("the `max_iter` must >= 1.")

        if not isinstance(self.tol, (int, float)) or isinstance(self.tol, bool):
            raise TypeError("the `tol` must be int or float.")

        if not (self.tol > 0):
            raise ValueError("the `tol` must > 0.")

        if self.num_hand is not None:
            if not isinstance(self.num_hand, int) or isinstance(self.num_hand, bool):
                raise TypeError("the `num_hand` must be int or None.")

            if self.num_hand < 0:
                raise ValueError("the `num_hand` must >= 0.")

        if self.seed is not None:
            # bool 是 int 的子类, 单独拒绝; numpy 整数一并接受并归一为 int
            if isinstance(self.seed, bool) or not isinstance(self.seed, (int, np.integer)):
                raise TypeError("the `seed` must be int or None.")

            if int(self.seed) < 0:
                raise ValueError("the `seed` must be non-negative.")

            self.seed = int(self.seed)

        if self.solver not in ("ck", "mckd"):
            raise ValueError(
                f"the `solver` must be 'ck' or 'mckd', got {self.solver!r}"
            )

        if self.dtype is not None and np.dtype(self.dtype).kind != "f":
            raise ValueError("the `dtype` must be a float numpy dtype or None.")

        if not isinstance(self.copy, bool):
            raise TypeError("the `copy` must be bool")

        self.safe = Check.validate_safe(self.safe)

    def config_generate(self) -> FMDConfig:
        return FMDConfig(
            M=self.M,
            L_factor=self.L_factor,
            K=self.K,
            max_iter=self.max_iter,
            tol=self.tol,
            num_hand=self.num_hand,
            seed=self.seed,
            dtype=self.dtype,
            solver=self.solver,
        )

    def decompose(self, S, T=None, output_memmap: bool = False) -> DecompositionResult:
        """
        分解信号得到各阶模态与残差。

        时间轴只参与校验, 不参与算法。模态全部累积在**局部变量**里 (实例上不存在
        imfs / res 之类的跨调用状态), 所以同一实例的第二次调用天然从干净状态开始,
        不会沿用上次结果。
        """
        # 形参校验先于任何计算: 传错类型应立即失败, 而不是跑完整个分解再报错
        check_output_memmap(output_memmap)
        scipy_signal = Cache.import_module("scipy.signal")
        scipy_linalg = Cache.import_module("scipy.linalg")

        # FMD 是纯时域滤波方法, 不使用时间轴; need_T=False 省掉 arange(N) 那份临时数组
        S, N = Check_Time_and_Signal(S, T=None, ndim={1}, method=self.name,
                                     dtype=self.dtype, need_T=False,
                                     safe=self.safe, copy=self.copy)

        if not (S.dtype.kind == "f" and S.dtype.itemsize >= 4):
            # int/bool 会让下面的浮点运算直接报错; f16 精度不足以撑住特征分解
            S = Check.to_dtype(S, np.dtype(DEFAULT_NUMPY_TYPE))

        S = np.asarray(S).ravel()  # 全程只读: 不改写调用方输入
        N = int(S.size)

        effective_seed: int | None = _resolve_seed(self.seed, self.name)

        # 退化输入 (常量 / 全零): 没有任何可分离的周期成分, 直接返回空模态 + 原信号
        if N < 10 or np.allclose(S, S[0], atol=1e-8):
            return finalize_output(
                DecompositionResult(
                    IMFs=np.empty((0, N), dtype=S.dtype),
                    Res=np.array(S, copy=True),
                    info={"M": None, "L": None, "num_hand": None, "modes": 0, "seed": effective_seed},
                    config=self.config_generate(),
                ),
                output_memmap,
            )

        rng = np.random.default_rng(effective_seed)

        mean_S = np.mean(S)
        std_S = np.std(S) + 1e-12
        S_norm = (S - mean_S) / std_S

        if self.M is None:
            M = _robust_period_estimation(S_norm, 1.0)
        else:
            M = self.M
        M = int(np.clip(M, 2, N // 4))
        L = int(np.clip(self.L_factor * M, 10, N // 2))

        if self.num_hand is None or self.num_hand <= 0:
            num_hand = _auto_estimate_num_candidates(S_norm, M, L, 1.0)
        else:
            num_hand = self.num_hand

        residual = S_norm.copy()
        modes_list = []

        # ---- 低频"特征模态"支路 ----------------------------------------- #
        # 相关峭度目标偏向冲击/故障特征, 对缓慢趋势与低频调频不敏感 (实测 case B
        # 的 amfm 只有 0.02-0.04)。参考实现的初始 Hanning 滤波器组会先给出一条低频
        # 带, 这里用同为"零相位、保形"的 Savitzky-Golay 平滑 (窗口 = 一个周期 2M+1,
        # 二次多项式 — 二次趋势被无损保留) 取出该低频分量; 只有当它占能量 >=2% 时
        # 才占用一阶, 否则完全不启用 (双音信号 A 的占比 <1%, 因此行为不变)。
        trend_mode = None
        if (self.K == -1 or self.K >= 2) and N >= 2 * M + 3:
            trend = scipy_signal.savgol_filter(S_norm, 2 * M + 1, 2)
            trend_energy = float(np.sum(trend ** 2))
            if trend_energy > 0.02 * float(np.sum(S_norm ** 2)):
                trend_mode = trend

        # K == -1 是"完全分解", 但不能靠下面两个收尾条件停下来: "窗口太小" 判断
        # 用的是常数 N (窗口从不收缩), 1e-10 的能量阈值对噪声残差不可达, 而噪声
        # 残差恒有 >= 3 个极值。自动模态数因此按 log2(N) + 2 设上限 (LMD 的自动
        # 阶数取 log2(N), 这里在同样量级上留 2 阶余量)。
        k_limit = int(np.log2(N)) + 2 if self.K == -1 else self.K

        k = 0
        while k < k_limit:
            # 第 0 阶优先给低频特征模态 (见上面的 trend_mode 说明)
            if trend_mode is not None:
                mode_energy = float(np.dot(trend_mode, trend_mode))
                if mode_energy > 0:
                    mode = trend_mode * (float(np.dot(residual, trend_mode)) / mode_energy)
                    if np.std(mode) >= 1e-8:
                        modes_list.append(mode)
                        residual = residual - mode
                        k += 1
                trend_mode = None
                continue

            x = residual.copy()
            Nx = len(x)

            if Nx - L + 1 <= 2 * M + 2:
                break

            try:
                X = np.lib.stride_tricks.sliding_window_view(x, L)
            except AttributeError:
                rows = Nx - L + 1
                X = np.zeros((rows, L))
                for i in range(rows):
                    X[i] = x[i:i+L]

            K_len = X.shape[0]
            M_eff = min(M, K_len - 1)

            X_short = X[:K_len - M_eff, :]
            X_shift = X[M_eff:, :]

            h = _advanced_filter_init(X_short, X_shift, num_hand=num_hand, rng=rng)
            h_prev = h.copy()

            for it in range(self.max_iter):
                y = X @ h
                y_short = y[:K_len - M_eff]
                y_shift = y[M_eff:]

                y2_sum = np.sum(y_short ** 2)
                if y2_sum < 1e-12:
                    break

                w = y_short * y_shift
                y2 = y_short ** 2

                A = (X_short.T * w) @ X_shift
                B = (X_short.T * y2) @ X_short

                A = 0.5 * (A + A.T)
                B = 0.5 * (B + B.T)

                B_trace = np.trace(B)
                reg = 1e-6 * B_trace / L if B_trace > 0 else 1e-6
                B_reg = B + reg * np.eye(L) + 1e-12 * np.eye(L)

                try:
                    if self.solver == "mckd":
                        # MCKD 的一步闭式更新: 把权重 w / y2 当作常数, 解线性方程
                        # B h = A h_old (对 B^{-1}A 做一次幂迭代) 再归一化。与 ck
                        # 的区别只在"如何求解同一个 CK 比值": ck 精确求广义特征对,
                        # mckd 走经典的一步线性化迭代。
                        h_new = scipy_linalg.solve(B_reg, A @ h_prev, assume_a="sym")
                    else:
                        eigvals, eigvecs = scipy_linalg.eigh(A, B_reg, subset_by_index=[L-1, L-1])
                        h_new = eigvecs[:, 0].real
                except np.linalg.LinAlgError:
                    try:
                        B_inv = np.linalg.pinv(B_reg, rcond=1e-12)
                        eig_vals, eig_vecs = scipy_linalg.eig(B_inv @ A)
                        max_idx = np.argmax(np.real(eig_vals))
                        h_new = np.real(eig_vecs[:, max_idx])
                    except np.linalg.LinAlgError:
                        h_new = h_prev.copy()

                h_norm = np.linalg.norm(h_new)
                if h_norm > 1e-12:
                    h_new = h_new / h_norm

                h_diff = np.linalg.norm(h_new - h)
                if h_diff < self.tol or (it > 0 and h_diff < 0.01):
                    h = h_new
                    break

                h_prev = h.copy()
                h = h_new

            mode = _bounded_convolution(x, h)

            # 单位范数的带通滤波器对窄带分量有约 sqrt(L) 的固有增益 (单音输入可放
            # 大十几倍), 直接相减会让残差逐阶放大并最终数值发散。用最小二乘投影把
            # 模态标定回残差的量纲:
            #     alpha = <x, mode> / <mode, mode>
            # 于是 ||x - alpha*mode|| <= ||x||, 残差能量单调不增; 只改幅度, 不改模态
            # 形状, 也不改算法结构。
            mode_energy = float(np.dot(mode, mode))
            if mode_energy <= 0:
                break
            mode = mode * (float(np.dot(x, mode)) / mode_energy)

            mode_norm = np.linalg.norm(mode)
            if mode_norm < 1e-12 or np.std(mode) < 1e-8:
                break

            modes_list.append(mode)
            residual = residual - mode
            k += 1

            if np.sum(residual ** 2) < 1e-10 * np.sum(S_norm ** 2):
                break

            peaks_max = scipy_signal.argrelextrema(residual, np.greater)[0]
            peaks_min = scipy_signal.argrelextrema(residual, np.less)[0]
            if len(peaks_max) + len(peaks_min) < 3:
                break

        # 模态在归一化空间里算完, 再换回物理量纲与工作 dtype (f32 输入 -> f32 输出)
        if modes_list:
            IMFs = (np.asarray(modes_list) * std_S).astype(S.dtype, copy=False)
        else:
            IMFs = np.empty((0, N), dtype=S.dtype)

        # 残差定义为 "原信号 - 各阶之和": 与算法内部的 residual 只差浮点尾数,
        # 但能保证重构恒等式 IMFs.sum(0) + Res == S 在浮点运算下成立
        Res = S - np.sum(IMFs, axis=0)

        info = {
            "M": M,
            "L": L,
            "num_hand": num_hand,
            "modes": len(modes_list),
            "seed": effective_seed,
        }

        return finalize_output(
            DecompositionResult(
                IMFs=IMFs,
                Res=Res,
                info=info,
                config=self.config_generate(),
            ),
            output_memmap,
        )


def _resolve_seed(local_seed: int | None, method: str) -> int | None:
    """
    两级随机种子解析 (全局 > 本地 > 默认)。

    ``Base.Seed`` 的 GLOBAL_SEED 恒定非 None (默认 42), 所以把 "仍等于导入时的
    默认值" 视为 "全局种子没有被显式设置": 此时优先本地 seed, 两者都没有就退回
    该默认值 —— 这样同一实例重复调用逐位可复现, 不依赖 ``np.random`` 的全局状态。
    """
    global_seed = Seed.get()

    if global_seed is None or global_seed == Global_Seed:
        return local_seed if local_seed is not None else global_seed

    if local_seed is not None and int(local_seed) != int(global_seed):
        warnings.warn(
            f"global seed {global_seed} overrides local seed {local_seed} for method {method}",
            UserWarning,
            stacklevel=3,
        )

    return global_seed


def _auto_estimate_num_candidates(signal: np.ndarray, M: int, L: int, fs: float) -> int:
    N = len(signal)

    snr_estimate = _estimate_snr(signal)

    complexity = _estimate_signal_complexity(signal)

    base_num = 10  # 候选基数, 下面按信噪比 / 复杂度 / 长度 / 周期追加后裁剪到 [5, 70]

    if snr_estimate < 10:
        base_num += 20
    elif snr_estimate < 20:
        base_num += 10

    if complexity > 0.7:
        base_num += 15
    elif complexity > 0.4:
        base_num += 8

    if N > 5000:
        base_num += 5
    elif N > 1000:
        base_num += 2

    if M < 50:
        base_num += 5

    return int(np.clip(base_num, 5, 70))


def _estimate_snr(signal: np.ndarray) -> float:
    scipy_signal = Cache.import_module("scipy.signal")

    analytic = scipy_signal.hilbert(signal)
    envelope = np.abs(analytic)

    # savgol 的窗长必须是 <= N 的奇数: 短信号 (退化用例) 按信号长度收缩, 否则 scipy 报错
    window_length = min(51, int(signal.size) if int(signal.size) % 2 == 1 else int(signal.size) - 1)
    window_length = max(3, window_length)
    envelope_smooth = scipy_signal.savgol_filter(envelope, window_length=window_length, polyorder=min(3, window_length - 1))
    noise_est = np.std(envelope - envelope_smooth)
    signal_est = np.std(envelope_smooth)

    if noise_est < 1e-12:
        return 100.0

    snr = 20 * np.log10(signal_est / noise_est)
    return max(0, snr)


def _estimate_signal_complexity(signal: np.ndarray) -> float:
    N = len(signal)

    m = 2
    r = 0.2 * np.std(signal)

    def _maxdist(xi, xj):
        return np.max(np.abs(xi - xj))

    def _phi(m):
        x = np.array([signal[i:i + m] for i in range(N - m + 1)])
        C = np.zeros(len(x))

        for i in range(len(x)):
            d = [_maxdist(x[i], x[j]) for j in range(len(x)) if j != i]
            C[i] = np.sum(np.array(d) <= r) / (len(x) - 1)

        return np.sum(np.log(C + 1e-12)) / len(x)

    if N > 2 * m + 1:
        apen = _phi(m) - _phi(m + 1)
        apen_norm = min(max(apen, 0), 2) / 2
    else:
        apen_norm = 0.5

    spec = np.abs(np.fft.rfft(signal))
    spec = spec[spec > 0]
    p = spec / np.sum(spec)
    entropy = -np.sum(p * np.log2(p + 1e-12))
    max_entropy = np.log2(len(p))
    spectral_entropy_norm = entropy / max_entropy if max_entropy > 0 else 0

    zero_crossings = np.sum(np.diff(np.signbit(signal)))
    zcr_norm = zero_crossings / (N - 1)

    complexity = 0.4 * apen_norm + 0.4 * spectral_entropy_norm + 0.2 * zcr_norm
    return np.clip(complexity, 0, 1)


def _robust_period_estimation(x: np.ndarray, fs: float) -> int:
    scipy_signal = Cache.import_module("scipy.signal")

    n = len(x)
    x = x - np.mean(x)

    max_lag = min(1000, n // 2)
    autocorr = np.correlate(x, x, mode='full')[n-1:n-1+max_lag]
    peaks, _ = scipy_signal.find_peaks(autocorr, distance=5, prominence=0.1*np.max(autocorr))
    period_ac = peaks[0] if len(peaks) > 0 and peaks[0] > 0 else 10

    analytic = scipy_signal.hilbert(x)
    envelope = np.abs(analytic)
    n_fft = 2**int(np.ceil(np.log2(len(envelope))))
    spec = np.abs(np.fft.rfft(envelope, n=n_fft))
    freqs = np.fft.rfftfreq(n_fft, 1/fs)

    valid_freqs = (freqs > 0.01*fs) & (freqs < 0.45*fs)
    if np.any(valid_freqs):
        f_main = freqs[valid_freqs][np.argmax(spec[valid_freqs])]
        period_env = int(round(fs / f_main)) if f_main > 0 else 10
    else:
        period_env = 10

    period = int(round(0.7*period_ac + 0.3*period_env))
    return max(2, min(period, n // 4))


def _advanced_filter_init(X_short: np.ndarray, X_shift: np.ndarray, num_hand: int = 10, rng=None) -> np.ndarray:
    scipy_linalg = Cache.import_module("scipy.linalg")

    L = X_short.shape[1]

    candidates = []

    for i in range(-2, 3):
        h = np.zeros(L)
        pos = L//2 + i
        if 0 <= pos < L:
            h[pos] = 1.0
        candidates.append(h)

    try:
        _, _, Vh = scipy_linalg.svd(X_short, full_matrices=False)
        svd_h = Vh[0, :]
        candidates.append(svd_h)
    except Exception:
        pass

    for _ in range(num_hand):
        h_rand = rng.standard_normal(L)
        candidates.append(h_rand)

    best_ck = -1e12
    best_h = np.zeros(L)
    best_h[L//2] = 1.0

    for h_candidate in candidates:
        h_norm = h_candidate / (np.linalg.norm(h_candidate) + 1e-12)
        y = X_short @ h_norm
        y_shift = X_shift @ h_norm
        y2_sum = np.sum(y ** 2)
        if y2_sum < 1e-12:
            continue
        ck = np.sum((y * y_shift) ** 2) / (y2_sum ** 2)
        if ck > best_ck:
            best_ck = ck
            best_h = h_norm

    return best_h


def _bounded_convolution(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    L = len(h)
    N = len(x)

    if L >= N:
        return np.convolve(x, h, mode='same')

    pad = L // 2
    x_pad = np.pad(x, (pad, pad), mode='reflect')
    y_pad = np.convolve(x_pad, h, mode='valid')

    start_idx = (len(y_pad) - N) // 2
    return y_pad[start_idx:start_idx+N]


"""
模块级便捷入口
"""
def f_FMD(S: list | np.ndarray, T=None, output_memmap: bool = False, *, M: int | None = None, L_factor: int = 2, K: int = 10, max_iter: int = 50, tol: float = 1e-06, num_hand: int | None = None, seed: int | None = None, dtype: type | np.dtype | None = None, solver: str = "ck", **kwargs) -> DecompositionResult:
    """构造 :class:`FMD` 并调用 :meth:`FMD.decompose`, 参数与构造器一致。"""
    fmd = FMD(
        M=M,
        L_factor=L_factor,
        K=K,
        max_iter=max_iter,
        tol=tol,
        num_hand=num_hand,
        seed=seed,
        dtype=dtype,
        solver=solver,
        **kwargs
    )

    return fmd.decompose(S, T, output_memmap=output_memmap)
