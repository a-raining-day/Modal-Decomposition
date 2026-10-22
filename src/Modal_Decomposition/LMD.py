"""
Local Mean Decomposition (局部均值分解)。

由极值插值迭代出局部均值与包络估计, 逐阶分离乘积函数 (PF)。

包络概念区分 (Hilbert vs Spline)
--------------------------------
- 模态分解筛分迭代里的"包络 / 局部均值"是对**极值**做插值得到 —— 本方法默认
  ``envelope="midpoint"`` (Smith 经典筛分): 幅度包络用保形 Pchip 插值,
  局部均值用自然三次样条 (均经 ``scipy.interpolate`` 提供; 极值端点镜像由
  本模块顶层的 ``mirror_extrema`` 完成)。自然三次样条在极值之间会过冲, 幅度经
  乘法累积后可放大到 ~1e12 量级 (重构损失 ~2.4e-4), 故幅度一律走 Pchip;
  最后一次筛分的局部均值按经典 LMD 语义留在残差中。
- 解析 Hilbert 包络 ``a(t) = |H[h(t)]|`` **不是**筛分迭代中的包络: 只有确实
  需要解析幅度时才使用 (``envelope="hilbert"`` 显式选择, 单次解调 / 实验
  对比)。该解调一旦进入迭代就会在多分量信号上发散 (包络拍频、极值数无界增长),
  故该变体只做单次解调。Hilbert 后端经 ``hilbert_mod`` 传入, 取 ``Utils.Hilbert``
  的 ``HILBERT_MODS`` 枚举: ``hilbert_mod="scipy"`` (默认, scipy.signal.hilbert)
  或 ``hilbert_mod="FHT"`` (第三方 C 内核, 见 ``Utils.Hilbert``)。实例上存的
  是 ``HILBERT_MODS`` 成员 (Enum), 不保存裸字符串。

References
----------
10.1098/rsif.2005.0058
"""

import numpy as np
from dataclasses import dataclass
from enum import Enum
from typing import ClassVar, Literal

from .Define import Config, Decomposer, DecompositionResult, DEFAULT_NUMPY_TYPE, check_output_memmap, finalize_output
from .Base import Cache
from .Utils import (
    Check,
    Check_Time_and_Signal,
    hilbert,
    HILBERT_MODS,
    HILBERT_MODS_ARGS,
    monotonic,
)

__all__ = ["LMD", "LMDConfig", "LMD_ENVELOPE_MODS", "LMD_ENVELOPE_MODS_ARGS", "f_LMD"]

LMD_ENVELOPE_MODS_ARGS = Literal["midpoint", "hilbert"]
class LMD_ENVELOPE_MODS(Enum):
    Midpoint = "midpoint"
    Hilbert = "hilbert"

"""
Config
"""
@dataclass(frozen=True, kw_only=True)
class LMDConfig(Config):
    """
    一次 LMD 运行的有效参数。
    """
    max_pf: int | None
    max_iter: int
    eps: float
    eps_stable: float
    min_amp: float
    max_amp: float
    converge_mean: float
    smooth_window: int
    envelope: LMD_ENVELOPE_MODS | LMD_ENVELOPE_MODS_ARGS
    hilbert_mod: HILBERT_MODS | HILBERT_MODS_ARGS
    dtype: type | np.dtype | None


"""
Class and Function of decomposer
"""
class LMD(Decomposer):
    name: ClassVar[str] = "LMD"

    def __init__(
        self,
        max_pf: int | None = None,
        max_iter: int = 37,
        eps: float = 0.05,
        eps_stable: float = 1e-12,
        min_amp: float = 1e-12,
        max_amp: float = 1e12,
        converge_mean: float = 1e-3,
        smooth_window: int = 5,
        envelope: LMD_ENVELOPE_MODS_ARGS | LMD_ENVELOPE_MODS = "midpoint",
        hilbert_mod: HILBERT_MODS_ARGS | HILBERT_MODS = "scipy",
        **kwargs,
    ):
        """
        Parameters
        ----------
        max_pf : int | None
            最大乘积函数阶数。``None`` 取 log2(num_imfs), ``-1`` 表示分解到底。
        max_iter : int
            每阶乘积函数的最大筛分迭代次数 (仅 midpoint 包络)。
        eps : float
            包络收敛阈值 (仅 midpoint 包络)。
        eps_stable : float
            接受乘积函数所需的最小能量。
        min_amp : float
            幅度估计的下界。
        max_amp : float
            幅度估计的上界。
        converge_mean : float
            局部均值收敛阈值 (仅 midpoint 包络)。
        smooth_window : int
            幅度估计的 Savitzky-Golay 窗口长度。
        envelope : LMD_ENVELOPE_MODS_ARGS | LMD_ENVELOPE_MODS
            幅度估计策略: ``"midpoint"`` (默认) = 经典 Smith 筛分, 对极值做 Pchip
            插值, 属于模态分解意义上的包络; ``"hilbert"`` = 每阶 PF 单次解析解调
            (``|H[h(t)]|``), 只给确实需要解析包络的场合 —— 它不是筛分包络,
            迭代不收敛。
        hilbert_mod : HILBERT_MODS_ARGS | HILBERT_MODS
            ``envelope="hilbert"`` 时的 Hilbert 后端, 取 ``Utils.Hilbert`` 的
            ``HILBERT_MODS``: ``"scipy"`` (默认, scipy.signal.hilbert) 或
            ``"FHT"`` (编译好的第三方 C 内核); 枚举里其余已声明的名字可以传,
            但分派器未实现。取值不做大小写归一, 必须与枚举值一致。
        hilbert_backend : {"scipy", "FHT"} | None
            **已废弃**, 仅为兼容旧调用方保留在 ``f_LMD`` 签名里; 不再生效。
        dtype : type | np.dtype | None
            返回乘积函数的工作 dtype。``None`` 沿用输入精度; int/bool 输入提升为 float64。
        **kwargs
            ``safe`` (校验级别, 见 ``Utils.Check``) 与 ``copy`` (让分解器持有自己的信号副本)。

        策略对比 (n=16384):

        * ``midpoint`` (默认): 经典 Smith 语义, PF 平滑; 重构误差 ~1e-15
          (自然三次样条过冲会把幅度放大到 1e12 量级, 误差达 ~2.4e-4 = 2 ulp(1e12),
          故幅度必须用保形 Pchip); 三者中最慢 (~0.13-0.19 s, 每阶多轮筛分);
          多音 / 宽带信号的分离较弱。
        * ``hilbert`` + Scipy: 最快 (~0.009 s, 每阶单次解调, 约 20x midpoint);
          精确重构 (~1e-15); AM-FM / 双音恢复最好 (0.99+); 趋势恢复较弱
          (~0.7 相关); 解调只能单次 (迭代不收敛); 包络对噪声敏感。
        * ``hilbert`` + FHT (第三方 C): 与 Scipy 路径逐位一致 (差 ~1e-16);
          独立 C 后端 (SAO Discrete Hilbert/Fourier/Hartley Transforms);
          单次 Hilbert 调用比 scipy 的 FFT 慢 ~1.9x (在 LMD 总耗时里被稀释);
          需要编译好的 ``_fht_native`` 扩展 (缺失时回退 NumPy 参考实现)。
        """
        self.max_pf: int | None = max_pf
        self.max_iter: int = max_iter
        self.eps: float = eps
        self.eps_stable: float = eps_stable
        self.min_amp: float = min_amp
        self.max_amp: float = max_amp
        self.converge_mean: float = converge_mean
        self.smooth_window: int = smooth_window
        self.envelope: LMD_ENVELOPE_MODS | LMD_ENVELOPE_MODS_ARGS = envelope
        self.hilbert_mod: HILBERT_MODS | HILBERT_MODS_ARGS = hilbert_mod


        unknown = set(kwargs) - {"safe", "copy", "dtype"}
        if unknown:
            raise TypeError(f"unexpected keyword arguments: {unknown}")

        self.safe: str = kwargs.get("safe", "unknown")
        self.copy: bool = kwargs.get("copy", True)
        self.dtype: type | np.dtype | None = kwargs.get("dtype", None)

        self.imfs: np.ndarray | None = None
        self.res: np.ndarray | None = None

        self.param_check()

    def param_check(self) -> None:
        self.envelope = Check.validate_mod(self.envelope, LMD_ENVELOPE_MODS)
        self.hilbert_mod = Check.validate_mod(self.hilbert_mod, HILBERT_MODS)

        if self.dtype is not None and np.dtype(self.dtype).kind != "f":
            raise ValueError("the `dtype` must be a float numpy dtype or None.")

        if not isinstance(self.copy, bool):
            raise TypeError("the `copy` must be bool")

    def config_generate(self) -> LMDConfig:
        return LMDConfig(
            max_pf=self.max_pf,
            max_iter=self.max_iter,
            eps=self.eps,
            eps_stable=self.eps_stable,
            min_amp=self.min_amp,
            max_amp=self.max_amp,
            converge_mean=self.converge_mean,
            smooth_window=self.smooth_window,
            envelope=self.envelope,
            hilbert_mod=self.hilbert_mod,
            dtype=self.dtype,
        )

    def decompose(self, S, T=None, output_memmap: bool = False) -> DecompositionResult:
        """
        把信号分解为乘积函数与残差。
        """
        # 形参校验先于任何计算: 传错类型应立即失败, 而不是跑完整个分解再报错
        check_output_memmap(output_memmap)
        scipy_signal = Cache.import_module("scipy.signal")
        argrelextrema = scipy_signal.argrelextrema
        savgol_filter = scipy_signal.savgol_filter

        # LMD 只在样本序号上插值, 不使用外部时间轴
        S, N = Check_Time_and_Signal(S, T=None, ndim={1}, method=self.name,dtype=self.dtype, need_T=False, safe=self.safe, copy=self.copy)

        if not (S.dtype.kind == "f" and S.dtype.itemsize >= 4):  # int/bool 会让包络除法直接报错, f16 会让样条出非有限值
            S = Check.to_dtype(S, np.dtype(np.float64))

        # 每次调用都从干净状态开始, 同一实例可反复复用
        self.imfs = None
        self.res = None

        n_samples = S.size

        if n_samples < 8:
            raise ValueError("LMD requires S length >= 8 (Smith 2005 standard)")

        t = np.arange(n_samples, dtype=np.float64)

        max_pf = self.max_pf
        if max_pf is None:
            max_pf = int(np.log2(n_samples))

        match self.envelope:
            case LMD_ENVELOPE_MODS.Hilbert:
                PFs, residue = self._sift_hilbert(S, n_samples, t, max_pf,
                                                  argrelextrema, savgol_filter)
            case _:
                PFs, residue = self._sift_midpoint(S, n_samples, t, max_pf,
                                                   argrelextrema, savgol_filter)

        self.imfs = np.array(PFs, dtype=S.dtype) if len(PFs) \
            else np.empty((0, n_samples), dtype=S.dtype)
        self.res = residue

        info = \
        {
            "envelope": self.envelope,
            "hilbert_mod": self.hilbert_mod,
            "max_pf": max_pf,
        }

        return finalize_output(
            DecompositionResult(
                IMFs=self.imfs,
                Res=self.res,
                info=info,
                config=self.config_generate(),
            ),
            output_memmap,
        )

    # ------------------------------------------------------------------
    def _sift_midpoint(self, S, n_samples, t, max_pf, argrelextrema, savgol_filter):
        """
        极值中点筛分, 带两道防幅度爆炸的护栏 (背景见模块 docstring):

        * 幅度包络用保形 Pchip 插值 (不过冲), 且
        * 累积幅度与 PF 采样都钳在**原始**信号尺度的 1e4 倍以内
          (病态解调点上的极小包络会把尖峰放大)。

        最后一次筛分的局部均值按经典 LMD 语义有意留在残差里, 它不属于重构误差。
        """
        residue = S.copy()
        PFs: list[np.ndarray] = []

        # 物理尺度锚在**原始**信号上, 不用残差: 一旦有尖峰漏进 PF, 残差尺度会被抬高
        amp_ref = max(float(np.max(np.abs(S))), self.min_amp)
        amp_cap = min(self.max_amp, 1e4 * amp_ref)
        pf_cap = min(self.max_amp, 1e4 * amp_ref)

        while max_pf == -1 or len(PFs) < max_pf:
            h = residue.copy()
            a_total = np.ones(n_samples, dtype=np.float64)
            converged = False

            for _ in range(self.max_iter):
                max_loc = argrelextrema(h, np.greater)[0]
                min_loc = argrelextrema(h, np.less)[0]
                ext_idx = np.unique(np.concatenate([max_loc, min_loc]))

                if len(ext_idx) < 3:
                    break

                ext_idx, ext_vals = mirror_extrema(
                    ext_idx, h[ext_idx], nbsym=0,
                    edges=(float(h[0]), float(h[-1]), n_samples),
                )

                t_mid = (ext_idx[:-1] + ext_idx[1:]) / 2.0
                m_vals = (ext_vals[:-1] + ext_vals[1:]) / 2.0
                a_vals = np.abs(ext_vals[:-1] - ext_vals[1:]) / 2.0

                m_t = _safe_interpolate(t_mid, m_vals, t)
                a_t = _safe_interpolate_pchip(t_mid, a_vals, t)

                a_t = np.clip(a_t, self.min_amp, self.max_amp)
                a_t = savgol_filter(a_t, self.smooth_window, 2)

                if _check_convergence(m_t, a_t, self.eps, self.converge_mean):
                    converged = True
                    break

                s_new = (h - m_t) / a_t
                a_total = np.clip(a_total * a_t, self.min_amp, amp_cap)
                h = s_new

            if not converged:
                break

            current_pf = np.clip(a_total * s_new, -pf_cap, pf_cap)

            if np.any(np.isnan(current_pf)) or np.any(np.isinf(current_pf)):
                break

            if np.sum(current_pf ** 2) < self.eps_stable:
                break

            PFs.append(current_pf)
            residue -= current_pf

            if (
                monotonic(residue, mod="monotonic", safe="safe") or
                np.sum(residue ** 2) < self.eps_stable or
                len(argrelextrema(residue, np.greater)[0]) + len(argrelextrema(residue, np.less)[0]) <= 2
            ):
                break

        return PFs, residue

    # ------------------------------------------------------------------
    def _sift_hilbert(self, S, n_samples, t, max_pf, argrelextrema, savgol_filter):
        """
        Hilbert 包络变体: 每阶 PF 只做一次解调。

            a(t) = |H[h(t)]| (平滑后),  s_FM = (h - m_t) / a,
            PF = a * s_FM = h - m_t

        该解调不迭代: 多分量信号上解析包络会拍频, 筛分随之发散 (极值数无界增长)。
        """
        residue = S.copy()
        PFs: list[np.ndarray] = []

        while max_pf == -1 or len(PFs) < max_pf:
            h = residue.copy()

            max_loc = argrelextrema(h, np.greater)[0]
            min_loc = argrelextrema(h, np.less)[0]
            ext_idx = np.unique(np.concatenate([max_loc, min_loc]))

            if len(ext_idx) < 3:
                break

            ext_idx, ext_vals = mirror_extrema(
                ext_idx, h[ext_idx], nbsym=0,
                edges=(float(h[0]), float(h[-1]), n_samples),
            )

            t_mid = (ext_idx[:-1] + ext_idx[1:]) / 2.0
            m_vals = (ext_vals[:-1] + ext_vals[1:]) / 2.0

            m_t = _safe_interpolate(t_mid, m_vals, t)
            a_t = np.abs(hilbert(h, self.hilbert_mod))

            a_t = np.clip(a_t, self.min_amp, self.max_amp)
            a_t = savgol_filter(a_t, self.smooth_window, 2)

            s_new = (h - m_t) / a_t
            current_pf = a_t * s_new

            if np.any(np.isnan(current_pf)) or np.any(np.isinf(current_pf)):
                break

            if np.sum(current_pf ** 2) < self.eps_stable:
                break

            PFs.append(current_pf)
            residue -= current_pf

            if (
                monotonic(residue, mod="monotonic", safe="safe") or
                np.sum(residue ** 2) < self.eps_stable or
                len(argrelextrema(residue, np.greater)[0]) + len(argrelextrema(residue, np.less)[0]) <= 2
            ):
                break

        return PFs, residue


"""
other function for module
"""
def f_LMD(S: list | np.ndarray, T=None, output_memmap: bool = False, *, max_pf: int | None = None, max_iter: int = 37, eps: float = 0.05, eps_stable: float = 1e-12, min_amp: float = 1e-12, max_amp: float = 1e12, converge_mean: float = 0.001, smooth_window: int = 5, envelope: LMD_ENVELOPE_MODS_ARGS | LMD_ENVELOPE_MODS = "midpoint", hilbert_mod: HILBERT_MODS_ARGS | HILBERT_MODS = "scipy", hilbert_backend: Literal["Scipy", "scipy", "FHT"] | None = None, dtype: type | np.dtype | None = None, **kwargs) -> DecompositionResult:
    lmd = LMD(
        max_pf=max_pf,
        max_iter=max_iter,
        eps=eps,
        eps_stable=eps_stable,
        min_amp=min_amp,
        max_amp=max_amp,
        converge_mean=converge_mean,
        smooth_window=smooth_window,
        envelope=envelope,
        hilbert_mod=hilbert_mod,
        dtype=dtype,
        **kwargs
    )

    return lmd.decompose(S, T, output_memmap=output_memmap)
