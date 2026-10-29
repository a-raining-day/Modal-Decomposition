"""
Empirical Fourier Decomposition (经验傅里叶分解)。

在幅值谱的局部极大值处切分频带, 逐带做零相位滤波并反变换得到各阶 IMF;
纯频域方法, 不依赖时间轴。

References
----------
10.1016/j.ymssp.2021.108155
"""

import numpy as np
from dataclasses import dataclass
from typing import List, ClassVar

from .Define import Config, Decomposer, DecompositionResult, DEFAULT_NUMPY_TYPE, check_output_memmap, finalize_output
from .Base import Cache
from .Utils import Check, Check_Time_and_Signal

__all__ = ["EFD", "EFDConfig", "f_EFD"]


"""
Config
"""
@dataclass(frozen=True, kw_only=True)
class EFDConfig(Config):
    """
    一次 EFD 运行的有效参数。
    """
    max_IMFs: int
    dtype: type | np.dtype | None


"""
Class and Function of decomposer
"""
class EFD(Decomposer):
    name: ClassVar[str] = "EFD"

    def __init__(self, max_IMFs: int = -1, dtype: type | np.dtype | None = None, **kwargs):
        """
        Parameters
        ----------
        max_IMFs : int
            最大 IMF 阶数; ``-1`` 表示返回全部频带对应的 IMF。
        dtype : type | np.dtype | None
            返回 IMF 的工作 dtype。``None`` 沿用输入精度; int/bool 输入提升为 float64。
        """
        self.max_IMFs: int = max_IMFs
        self.dtype: type | np.dtype | None = dtype

        unknown = set(kwargs) - {"safe", "copy"}
        if unknown:
            raise TypeError(f"unexpected keyword arguments: {unknown}")

        self.safe: str = kwargs.get("safe", "unknown")
        self.copy: bool = kwargs.get("copy", True)

        self.param_check()

    def param_check(self) -> None:
        if not isinstance(self.max_IMFs, int) or isinstance(self.max_IMFs, bool):
            raise TypeError("The type of the max_IMFs must be int!")

        if self.max_IMFs != -1 and self.max_IMFs <= 0:
            raise ValueError("Invalid value! Do you want use -1?")

        if self.dtype is not None and np.dtype(self.dtype).kind != "f":
            raise ValueError("the `dtype` must be a float numpy dtype or None.")

        if not isinstance(self.copy, bool):
            raise TypeError("the `copy` must be bool")

    def config_generate(self) -> EFDConfig:
        return EFDConfig(max_IMFs=self.max_IMFs, dtype=self.dtype)

    def decompose(self, S, T=None, output_memmap: bool = False) -> DecompositionResult:
        """
        把信号分解为 IMF 与残差。

        残差取 ``S - ΣIMFs`` 并补回均值, 因此 ``ΣIMFs + Res`` 精确等于输入信号。
        """
        # 形参校验先于任何计算: 传错类型应立即失败, 而不是跑完整个分解再报错
        check_output_memmap(output_memmap)
        scipy_signal = Cache.import_module("scipy.signal")

        # EFD 是纯频域方法, 不使用时间轴; 用 need_T=False 省掉 arange(N) 那份临时数组
        S, N = Check_Time_and_Signal(S, T=None, ndim={1}, method=self.name,
                                     dtype=self.dtype, need_T=False, safe=self.safe, copy=self.copy)

        if not (S.dtype.kind == "f" and S.dtype.itemsize >= 4):
            S = Check.to_dtype(S, np.dtype(np.float64))

        # 退化输入 (常量 / 全零): 去均值后没有能量, 频谱里也没有峰, 分不出任何模态
        MEAN: float = np.mean(S)
        S = S - MEAN
        if np.max(np.abs(S)) <= np.finfo(float).tiny:
            empty = np.empty((0, N), dtype=S.dtype)
            return finalize_output(
                DecompositionResult(
                    IMFs=empty,
                    Res=S + MEAN,
                    info={"mean": MEAN, "fft_signal": None, "magnitude": None, "phase": None},
                    config=self.config_generate()
                ),
                output_memmap,
            )

        F = np.fft.fft(S)
        magnitude = np.abs(F)
        phase = np.angle(F)

        # 单边幅值谱: 下标 0 = 直流, 下标 M-1 = Nyquist (实信号共轭对称)
        M = N // 2 + 1
        one_sided = magnitude[:M]

        # ---- 频谱分段 (EFD 论文的 Segm_tec) ------------------------------ #
        # 候选 = 严格局部极大 + 两端点; 取幅值最大的 max_IMFs 个; 边界取相邻
        # 候选之间的幅值最小处。旧实现只保留 max_IMFs-2 个极大, 在双音信号上
        # 会把两个音吞进同一个带 (实测 case A: 某阶占 99.5% 能量)。
        is_local_max = np.zeros(M, dtype=bool)
        if M >= 3:
            is_local_max[1:-1] = ((one_sided[1:-1] > one_sided[:-2])
                                  & (one_sided[1:-1] > one_sided[2:]))
        is_local_max[0] = True                       # 端点也是候选 (与 Segm_tec 一致)
        is_local_max[-1] = True

        candidates = np.flatnonzero(is_local_max)

        if self.max_IMFs != -1 and candidates.size > self.max_IMFs:
            keep = np.argsort(-one_sided[candidates], kind="stable")[: self.max_IMFs]
            candidates = np.sort(candidates[keep])

        # 每个内部边界: 相邻候选之间 (含两端) 幅值最小处
        edges = [0]
        for left, right in zip(candidates[:-1], candidates[1:]):
            if right <= left:
                continue
            edges.append(int(left + np.argmin(one_sided[left:right + 1])))
        edges.append(M - 1)
        edges = np.unique(np.asarray(edges, dtype=np.int64))

        # 各带是频谱上的**不重叠**区间 [lo, hi); 最后一个带含 Nyquist,
        # 因此 ΣIMFs 恰好等于 S - mean (重构恒等式由分段完整性保证)。
        bands = [(int(lo), int(hi)) for lo, hi in zip(edges[:-1], edges[1:]) if hi > lo]
        if not bands:                                # 退化: 单带覆盖全谱
            bands = [(0, M - 1)]
        bands[-1] = (bands[-1][0], M)

        # 顶带右界收到"最后一个保留极大之后的幅值最小处": 该极大右侧的噪声尾巴
        # 不属于任何频带, 由残差吸收 (参考实现同样把这段丢掉; 我们的区别是
        # 低端 0..第一个极小 仍然作为一阶, 因此趋势分量不会被丢)。
        # 残差按 S - ΣIMFs 计算, 所以重构恒等式不受影响。
        if candidates.size and candidates[-1] < M - 1:
            cut = int(candidates[-1] + np.argmin(one_sided[candidates[-1]:]))
            if cut > bands[-1][0]:
                bands[-1] = (bands[-1][0], cut)

        # 直接预分配 (K, N), 不用"list 追加 + asarray" —— 后者会让模态峰值翻倍
        K: int = len(bands)
        IMFs = np.empty((K, N), dtype=S.dtype)
        spec = np.empty(M, dtype=complex)
        # 每个带: 半谱置零后拷入原始频点, 一次实数逆变换。
        # 旧实现每带都算一遍 exp(1j*phase) 并做全长复数 ifft, 白付 2 倍以上代价。
        for idx, (lo, hi) in enumerate(bands):
            spec[:] = 0.0
            spec[lo:hi] = F[lo:hi]
            IMFs[idx] = np.fft.irfft(spec, n=N)

        Res = S - np.sum(IMFs, axis=0) + MEAN

        info = \
        {
            "mean": MEAN,
            "fft_signal": F,
            "magnitude": magnitude,
            "phase": phase,
        }

        return finalize_output(
            DecompositionResult(
                IMFs,
                Res,
                info,
                self.config_generate(),
            ),
            output_memmap,
        )


"""
other function for module
"""
def f_EFD(S: list | np.ndarray, T=None, output_memmap: bool = False, *, max_IMFs: int = -1, dtype: type | np.dtype | None = None, **kwargs) -> DecompositionResult:
    efd = EFD(
        max_IMFs=max_IMFs,
        dtype=dtype,
        **kwargs
    )

    return efd.decompose(S, T, output_memmap=output_memmap)
