import numpy as np
from dataclasses import dataclass
from typing import List

from .EMD import EMD, EMDConfig

from .Define import Config, Decomposer, DecompositionResult, DEFAULT_NUMPY_TYPE
from .Base import Cache, set_allocable
from .Utils import Check, monotonic, mirror, find_peaks, Check_Time_and_Signal

"""
Config
"""
@dataclass(frozen=True, kw_only=True)
class CEEMDANConfig(Config):
    num_imf: int
    noise_amp: float
    M: int
    parallel: bool
    num_workers: int
    dtype: type | np.dtype | None
    EMD_config: EMDConfig


"""
Class and Function of decomposer
"""
class CEEMDAN(Decomposer):
    name = "CEEMDAN"

    def __init__(self, num_imf: int = -1, noise_amp: float = 0.1, M: int = 25, pre_decompose_noise: bool = False, parallel: bool = False, num_workers: int = 1, **kwargs):
        self.num_imf: int = num_imf
        self.noise_amp: float = noise_amp
        self.M: int = M
        self.pre_decompose_noise: bool = pre_decompose_noise
        self.parallel: bool = parallel
        self.num_workers: int = num_workers

        unknown = set(kwargs) - ...  # 让 AI 补充
        if unknown:
            raise TypeError(f"unexpected keyword arguments: {unknown}")

        self.dtype: type | np.dtype | None = kwargs.get("dtype", None)

        self.emd_num_imf: int = 1
        self.emd_pre_decompose_num_imf: int = self.num_imf if self.num_imf == -1 else self.num_imf - 1
        self.emd_nbsym: int = kwargs.get("emd_nbsym", 4)
        self.emd_internal_delta: float = kwargs.get("emd_internal_delta", 0.2)
        self.emd_internal_iter: int = kwargs.get("emd_internal_iter", 50)
        self.emd_res_energy: float = kwargs.get("emd_res_energy", 1e-6)
        self.emd_faster: bool = kwargs.get("emd_faster", True)
        self.emd_store_iter_info: bool = kwargs.get("emd_store_iter_info", False)
        self.emd_pre_num_imfs: int = kwargs.get("emd_pre_num_imfs", 10)
        self.emd_dtype: type | np.dtype | None = kwargs.get("emd_dtype", None)
        self.emd_safe: str = kwargs.get("emd_safe", "safe")
        self.emd_need_squeeze: bool = kwargs.get("emd_need_squeeze", True)
        self.emd_mirror_mod: str = kwargs.get("emd_mirror_mod", "extrema")
        self.emd_poly_order: int = kwargs.get("emd_poly_order", 2)
        self.emd_time_poly_order: int = kwargs.get("emd_time_poly_order", 2)
        self.emd_peaks_backend: str = kwargs.get("emd_peaks_backend", "numpy")

        self.param_check()

        self.EMD = EMD(
            num_imf=self.emd_num_imf,
            nbsym=self.emd_nbsym,
            internal_delta=self.emd_internal_delta,
            internal_iter=self.emd_internal_iter,
            res_energy=self.emd_res_energy,
            faster=self.emd_faster,
            store_iter_info=self.emd_store_iter_info,
            pre_num_imfs=self.emd_pre_num_imfs,
            dtype=self.emd_dtype,
            safe=self.emd_safe,
            need_squeeze=self.emd_need_squeeze,
            mirror_mod=self.emd_mirror_mod,
            poly_order=self.emd_poly_order,
            time_poly_order=self.emd_time_poly_order,
            peaks_backend=self.emd_peaks_backend
        )

        if self.pre_decompose_noise:
            self.pre_decompose_EMD = EMD(
                num_imf=self.emd_pre_decompose_num_imf,
                nbsym=self.emd_nbsym,
                internal_delta=self.emd_internal_delta,
                internal_iter=self.emd_internal_iter,
                res_energy=self.emd_res_energy,
                faster=self.emd_faster,
                store_iter_info=self.emd_store_iter_info,
                pre_num_imfs=self.emd_pre_num_imfs,
                dtype=self.emd_dtype,
                safe=self.emd_safe,
                need_squeeze=self.emd_need_squeeze,
                mirror_mod=self.emd_mirror_mod,
                poly_order=self.emd_poly_order,
                time_poly_order=self.emd_time_poly_order,
                peaks_backend=self.emd_peaks_backend
            )

        self.imfs: np.ndarray = None
        self.res:  np.ndarray = None

    def param_check(self) -> None:
        if not isinstance(self.num_imf, int) or isinstance(self.num_imf, bool):
            raise TypeError("num_imf must be an int")

        if self.num_imf != -1:
            if not (self.num_imf > 0):
                raise ValueError("num_imf must > 0")

        if self.noise_amp is not None:
            if not isinstance(self.noise_amp, (int, float)) or isinstance(self.noise_amp, bool):
                raise TypeError("noise_amp must be float or None")
            if not (self.noise_amp > 0):
                raise ValueError("noise_amp must > 0")

        if not isinstance(self.M, int) or isinstance(self.M, bool):
            raise TypeError("M must be an int")

        if not (self.M > 0):
            raise ValueError("M must > 0")

        if not isinstance(self.parallel, bool):
            raise TypeError("parallel must be bool")

        if not isinstance(self.num_workers, (int, float)) or isinstance(self.num_workers, bool):
            raise TypeError("num_workers must be an int or float")

        self.num_workers: int = set_allocable(self.num_workers)

        if self.dtype is not None and np.dtype(self.dtype).kind != "f":
            raise ValueError("the `dtype` must be a float numpy dtype or None.")

        if not isinstance(self.pre_decompose_noise, bool):
            raise TypeError("the `pre_decompose_noise` must be bool")

    def config_generate(self) -> CEEMDANConfig:
        return CEEMDANConfig(
            num_imf=self.num_imf,
            noise_amp=self.noise_amp,
            M=self.M,
            parallel=self.parallel,
            num_workers=self.num_workers,
            dtype=self.dtype,
            EMD_config=self.EMD.config_generate()
        )

    def decompose(self, S: List[int | float] | np.ndarray) -> DecompositionResult:
        S, N = Check_Time_and_Signal(S, T=None, ndim={1}, method=self.name, dtype=self.dtype, need_T=False)

        if not (
                S.dtype.kind == "f" and S.dtype.itemsize >= 4):  # int/bool 输入会在 avg_env /= 2 处直接 UFuncTypeError, f16 会让样条报非有限值
            S = Check.to_dtype(S, np.dtype(np.float64))

        # 每次调用都从干净状态开始: 否则第二次 decompose 会把上一次的累积结果当成初值
        self.imfs = None
        self.res = None

        # 自动档取 0.3·std(S): EEMD 的 Pareto 扫描显示弱音调捕获在 alpha≈0.3 处饱和; CEEMD 的互补噪声对消掉大部分残余, 同一强度下残余更小, 故沿用同一阈值
        noise_amp = self.noise_amp if self.noise_amp is not None else float(S.std() * 0.3)

        _, _, peak_max = find_peaks(S, mod="both", safe="safe", backend=self.emd_peaks_backend)
        if np.sum(peak_max) < 2 or np.sum(~peak_max) < 2:
            self.imfs = np.empty((0, N), dtype=S.dtype)
            self.res = S
            return DecompositionResult(
                IMFs=self.imfs,
                Res=self.res,
                info={"rows": [], "min row": 0, "noise_amp": noise_amp, "EMD info": []},
                config=self.config_generate()
            )

        if not self.parallel:
            if not self.pre_decompose_noise and self.num_imf == -1:
                ...

            else:
                noise_imfs: List[np.ndarray] = []

                if self.pre_decompose_noise:  # -1 or num
                    min_rows: int = None

                    for m in range(self.M):
                        noise = np.random.normal(loc=0, scale=noise_amp, size=S.size).astype(S.dtype, copy=False)
                        result = self.pre_decompose_EMD.decompose(noise)

                        imfs = result.IMFs
                        min_rows = imfs.shape[0] if min_rows is None else min(min_rows, imfs.shape[0])

                        noise_imfs.append(imfs)



                else:  # num
                    ...