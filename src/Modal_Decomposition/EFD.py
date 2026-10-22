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

        edge_magnitude = magnitude[: N // 2 + 1].copy()[1:] * 2  # 去掉直流分量 (下标 0)
        freq_N = N // 2

        local_maximum_points_tuple = scipy_signal.argrelmax(edge_magnitude)
        local_maximum_points = local_maximum_points_tuple[0]
        local_maximum = edge_magnitude[local_maximum_points]

        if self.max_IMFs != -1:
            # 分段规则产生 (内部极大值 + 2) 个频带, 所以想要 m 个 IMF 只需保留 m-2 个
            # 内部极大值 (两端各贡献一个边界); 取能量最大的那几个极大值。
            zipped = sorted(zip(local_maximum_points.tolist(), local_maximum.tolist()),
                            reverse=True, key=lambda x: x[1])
            local_maximum_points = sorted(p for p, _ in zipped[: max(0, self.max_IMFs - 2)])

        bounds = np.concatenate(([0], local_maximum_points, [freq_N - 1])).astype(np.int64)
        bounds = np.unique(bounds)
        bounds = np.sort(bounds)

        if self.max_IMFs == 1:
            # 只要 1 阶时整条正频率谱就是一个频带
            wn = np.array([0, freq_N - 1])
        else:
            wn = []  # 零相位滤波器的频带边界 (频点下标)
            for p in range(len(bounds) - 1):
                next_point = bounds[p + 1]
                current_point = bounds[p]

                if edge_magnitude[current_point] == edge_magnitude[next_point]:
                    wn.append(current_point)
                else:
                    wn.append(current_point + np.argmin(edge_magnitude[current_point:next_point + 1]))
            wn = np.concatenate(([0], wn, [freq_N - 1]))

        # 每个频带只由一对边界下标定义: 其支撑集就是闭区间 [wn[e-1], wn[e]]。
        # 不在这里物化 N/2 长的 0/1 掩码 —— 那会让内存变成 O(N × 频带数)。
        bands = [(int(wn[e - 1]), int(wn[e])) for e in range(1, len(wn))]

        # 直接预分配 (K, N), 不用"list 追加 + asarray" —— 后者会让模态峰值翻倍
        K: int = len(bands)
        IMFs = np.empty((K, N), dtype=S.dtype)
        M = N // 2 + 1
        # 实信号的频谱共轭对称: 只取正频率半边, 负频率半边按共轭补回即可
        for idx, (last_point, current_point) in enumerate(bands):
            filtered_magnitude = np.zeros(M, dtype=float)
            # filtered_magnitude 的下标 1+j 对应 edge_magnitude 的下标 j
            # (下标 0 是直流, 恒置 0), 故支撑集平移到 [last_point+1, current_point+1]。
            lo = min(max(last_point + 1, 1), M)
            hi = min(max(current_point + 2, lo), M)
            if hi > lo:
                filtered_magnitude[lo:hi] = magnitude[lo:hi]
            temp_spectrum = filtered_magnitude * np.exp(1j * phase[:M])

            full_spectrum = np.zeros(N, dtype=complex)
            full_spectrum[:M] = temp_spectrum  # 正频率半边 (复数)
            full_spectrum[N - M + 2:] = np.conj(temp_spectrum[1:M - 1][::-1])
            IMFs[idx] = np.fft.ifft(full_spectrum).real

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
