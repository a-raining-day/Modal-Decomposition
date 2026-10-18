import numpy as np
from dataclasses import dataclass
from typing import List

from .Define import Config, Decomposer, DecompositionResult, DEFAULT_NUMPY_TYPE
from .Base import Cache
from .Utils import Check, monotonic, mirror, find_peaks, Check_Time_and_Signal

"""
Config
"""
@dataclass(frozen=True, kw_only=True)
class EMDConfig(Config):
    num_imf: int
    nbsym: int
    internal_iter: int
    internal_delta: float
    res_energy: float
    store_iter_info: bool
    pre_num_imfs: int
    dtype: type | np.dtype | None
    safe: str
    need_squeeze: bool
    mirror_mod: str
    poly_order: int
    time_poly_order: int
    peaks_backend: str
    faster: bool


"""
Class and Function of decomposer
"""
class EMD(Decomposer):
    name = "EMD"

    def __init__(self, num_imf: int = -1, nbsym: int = 4, internal_delta: float = 0.2, internal_iter: int = 50, res_energy: float = 1e-6, faster: bool = True, store_iter_info: bool = False, copy: bool = True, **kwargs):
        """

        :param num_imf:
        :param nbsym:
        :param internal_delta:
        :param internal_iter:
        :param res_energy:
        :param faster:
        :param store_iter_info:
        :param copy: 是否让 ``Check_Time_and_Signal`` 交回一份归本函数所有的信号副本。
                     默认 True: 残差可以直接原地 ``S -= imf``, 不必每轮分配等长数组,
                     调用方的输入也不会被改写。传 ``False`` 时不复制 (省一份 N 长数组,
                     且保留 memmap 支撑), 但 ``decompose`` 会**原地覆盖调用方的输入数组**。
        :param kwargs:
        """

        self.num_imf: int = num_imf
        self.nbsym: int = nbsym
        self.internal_iter: int = internal_iter
        self.internal_delta: float = internal_delta
        self.res_energy: float = res_energy
        self.faster: bool = faster
        self.store_iter_info: bool = store_iter_info
        self.copy: bool = copy

        unknown = set(kwargs) - {
            "pre_num_imfs", "dtype", "safe", "need_squeeze",
            "mirror_mod", "poly_order", "time_poly_order", "peaks_backend",
        }
        if unknown:
            raise TypeError(f"unexpected keyword arguments: {unknown}")

        self.pre_num_imfs: int = kwargs.get("pre_num_imfs", 10)

        self.dtype: type | np.dtype | None = kwargs.get("dtype", None)
        self.safe: str = kwargs.get("safe", "unknown")
        self.need_squeeze: bool = kwargs.get("need_squeeze", True)

        self.mirror_mod: str = kwargs.get("mirror_mod", "extrema")
        self.poly_order: int = kwargs.get("poly_order", 2)
        self.time_poly_order: int = kwargs.get("time_poly_order", 2)
        self.peaks_backend: str = kwargs.get("peaks_backend", "numpy")

        # self.config: EMDConfig = config  # 暂时先不考虑 config 输入, 有一些问题需要解决

        self.param_check()

        self.imfs: np.ndarray = None
        self.res:  np.ndarray = None

        self.temp_imf_arr: List[np.ndarray] = [] if self.store_iter_info else None
        self.iter_convergence: List[bool] = [] if self.store_iter_info else None

    def param_check(self) -> None:
        if not isinstance(self.num_imf, int) or isinstance(self.num_imf, bool):
            raise TypeError("the `num_imf` must be int.")

        if self.num_imf != -1:
            if not (self.num_imf > 0):
                raise TypeError("the `num_imf` must be positive.")

        if not isinstance(self.nbsym, int) or isinstance(self.nbsym, bool):
            raise TypeError("the `nbsym` must be int.")

        if not (self.nbsym >= 0):
            raise ValueError("the `nbsym` must >= 0.")

        if not isinstance(self.res_energy, float):
            raise TypeError("the `res_energy` must be float.")

        if not (self.res_energy >= 0):
            raise ValueError("the `res_energy` must >= 0.")

        if not isinstance(self.internal_iter, int) or isinstance(self.internal_iter, bool):
            raise TypeError("the `internal_iter` must be int.")

        if self.internal_iter != -1:
            if not (self.internal_iter > 0):
                raise ValueError("the `internal_iter` must > 0.")

        if not isinstance(self.internal_delta, float):
            raise TypeError("the `internal_delta` must be float.")

        if not (0 <= self.internal_delta < 1):
            raise ValueError("the `internal_delta` must be between 0 and 1.")

        if not isinstance(self.pre_num_imfs, int) or isinstance(self.pre_num_imfs, bool):
            raise TypeError("the `pre_num_imfs` must be int.")

        if not (self.pre_num_imfs >= 1):
            raise ValueError("the `pre_num_imfs` must >= 1.")

        if not isinstance(self.store_iter_info, bool):
            raise TypeError("the `store_iter_info` must be bool.")

        if not isinstance(self.copy, bool):
            raise TypeError("the `copy` must be bool.")

        if self.dtype is not None and np.dtype(self.dtype).kind != "f":
            raise ValueError("the `dtype` must be a float numpy dtype or None.")

        if self.safe not in ("safe", "suspicious", "unknown"):
            raise ValueError("the `safe` can only be ['safe', 'suspicious', 'unknown'].")

        if self.peaks_backend not in ("scipy", "numpy", "numba"):
            raise ValueError("the `peaks_backend` can only be ['scipy', 'numpy', 'numba'].")

        # "mirror" 模式返回的是 1-D 信号而非 (时间, 幅值, 极值掩码) 三元组, 本处无法使用
        if self.mirror_mod not in ("extrema", "polynomial", "parallel", "BLCSE"):
            raise ValueError("the `mirror_mod` can only be ['extrema', 'polynomial', 'parallel', 'BLCSE'].")

        if not isinstance(self.poly_order, int) or isinstance(self.poly_order, bool) or self.poly_order <= 0:
            raise ValueError("the `poly_order` must be a positive int.")

        if not isinstance(self.time_poly_order, int) or isinstance(self.time_poly_order, bool) or self.time_poly_order <= 0:
            raise ValueError("the `time_poly_order` must be a positive int.")

        # polynomial 模式每端只有 nbsym+1 个极值参与拟合, 阶数不得超过 nbsym
        if self.nbsym != 0 and self.mirror_mod == "polynomial":
            if self.poly_order > self.nbsym or self.time_poly_order > self.nbsym:
                raise ValueError("the `poly_order` and the `time_poly_order` must be <= `nbsym` when mirror_mod='polynomial'.")

        if not isinstance(self.faster, bool):
            raise TypeError("the `faster` must be bool.")

    def config_generate(self) -> EMDConfig:
        return EMDConfig(
            num_imf=self.num_imf,
            nbsym=self.nbsym,
            internal_iter=self.internal_iter,
            internal_delta=self.internal_delta,
            res_energy=self.res_energy,
            store_iter_info=self.store_iter_info,
            pre_num_imfs=self.pre_num_imfs,
            dtype=self.dtype,
            safe=self.safe,
            need_squeeze=self.need_squeeze,
            mirror_mod=self.mirror_mod,
            poly_order=self.poly_order,
            time_poly_order=self.time_poly_order,
            peaks_backend=self.peaks_backend,
            faster=self.faster,
        )

    def decompose(self, S: list | np.ndarray) -> DecompositionResult:
        si = Cache.import_module("scipy.interpolate")

        S, N = Check_Time_and_Signal(S, T=None, ndim={1}, method=self.name, dtype=self.dtype, need_T=False, safe=self.safe, copy=self.copy)

        if not (S.dtype.kind == "f" and S.dtype.itemsize >= 4):  # int/bool 输入会在 avg_env /= 2 处直接 UFuncTypeError, f16 会让样条报非有限值
            S = Check.to_dtype(S, np.dtype(np.float64))

        # 能量一律用 dot 而非 sum(x**2) 避免每次都要分配一份与信号等长的临时数组
        original_energy = np.dot(S, S)

        grid = None

        if self.num_imf == -1:
            initial_rows = self.pre_num_imfs
        else:
            initial_rows = self.num_imf
        self.imfs = np.empty((initial_rows, N), dtype=S.dtype)
        total_imfs = initial_rows
        num_imf = 0
        self.res = None


        iterations: List[int] = []
        final_sds: List[float] = []

        if self.store_iter_info:
            self.temp_imf_arr = []
            self.iter_convergence = []

        while True:
            # 停止判据前置: 残差已单调或已低于能量下限时不得再产出 IMF
            # (res_energy 按原始信号能量归一, 保证对幅值尺度不变)
            if monotonic(S, mod="monotonic", safe="safe") or np.dot(S, S) < self.res_energy * original_energy:
                self.res = S
                break

            # 内层筛分
            if grid is None:
                grid = np.arange(N, dtype=np.float64)
            h = S  # 可省略为 h = S
            last_h = None
            last_sd = float("nan")
            inner_iter = 0

            if self.store_iter_info:
                this_temp_imfs = []
                this_iter_convergences = []

            while True:
                inner_iter += 1
                if self.internal_iter != -1:
                    if inner_iter > self.internal_iter:
                        break

                # 找极值: 先按原始极值判定"残差是否还值得再筛", 再做端点延拓
                peak_x, peak_arr, peak_max = find_peaks(h, mod="both", safe="safe", backend=self.peaks_backend)
                if np.sum(peak_max) < 2 or np.sum(~peak_max) < 2:
                    break

                if self.nbsym == 0:
                    mirror_x, mirror_arr, mirror_max = peak_x, peak_arr, peak_max
                else:
                    try:
                        mirror_x, mirror_arr, mirror_max = mirror(h, safe="safe", nbsym=self.nbsym, mod=self.mirror_mod, peaks_backend=self.peaks_backend, poly_order=self.poly_order,time_poly_order=self.time_poly_order)
                    except ValueError as exc:
                        msg = str(exc)
                        if ("nbsym must be <=" in msg) or ("no extrema" in msg) or ("at least 2 extrema" in msg) or ("needs both the maxima" in msg):
                            mirror_x, mirror_arr, mirror_max = peak_x, peak_arr, peak_max
                        else:
                            raise

                max_spline = si.CubicSpline(mirror_x[mirror_max], mirror_arr[mirror_max])
                min_spline = si.CubicSpline(mirror_x[~mirror_max], mirror_arr[~mirror_max])
                max_env = np.asarray(max_spline(grid), dtype=h.dtype)
                min_env = np.asarray(min_spline(grid), dtype=h.dtype)
                # 求局部均值。此处不能像以前那样就地复用 max_env 的缓冲: faster=False 时
                # end() 要用原样的上/下包络做"包络异号"判定 (代价: 多一份 N 长数组)
                avg_env = np.add(max_env, min_env)
                avg_env /= 2

                h_new = h - avg_env

                if last_h is not None:
                    # 能量用 dot 而非 sum(x**2): 不产生与信号等长的平方临时数组
                    denominator = np.dot(last_h, last_h)
                    if denominator < 1e-12:
                        SD = 0.0
                    else:
                        delta = last_h - h_new
                        SD = np.dot(delta, delta) / denominator
                    last_sd = SD
                    # faster=True : Cauchy SD 收敛即放行 (高速档)
                    # faster=False: 还要通过 end() 的 IMF 必要条件 (对标 OLD / PyEMD)
                    if SD <= self.internal_delta and (self.faster or self.end(h_new, h, max_env, min_env)):
                        h = h_new
                        if self.store_iter_info:
                            this_temp_imfs.append(h_new)
                            this_iter_convergences.append(True)
                        break

                if self.store_iter_info:
                    this_temp_imfs.append(h_new)
                    this_iter_convergences.append(False)

                last_h = h_new
                h = h_new

            imf = h

            # 无进展保护: 本次没提取出有效分量 -> 残差不再变化, 外层循环必须终止
            #   * ``imf is S``       : 内层首次迭代就因极值不足退出, h 从未被改写,
            #                          残差与输入同一对象 (短/单峰信号), 不得当成 IMF;
            #   * 幅值可忽略          : BLCSE 等会伪造极值的模式绕过了极值守卫, 筛分塌缩成
            #                          零分量; 没有这一条就会退化成"零分量无限循环 +
            #                          self.imfs 无界增长"。
            if (imf is S) or max(float(imf.max()), -float(imf.min())) <= 1e-12 * max(np.sqrt(original_energy / N), np.finfo(float).tiny):
                self.res = S
                break

            iterations.append(inner_iter)
            final_sds.append(last_sd)

            if num_imf >= total_imfs:
                self.imfs = np.resize(self.imfs, (total_imfs + self.pre_num_imfs, N))
                total_imfs += self.pre_num_imfs
            self.imfs[num_imf, :] = imf
            num_imf += 1

            # 残差就地更新: S 是 Check_Time_and_Signal(copy=True) 交回的副本, 归本函数所有,
            # 每轮原地相减, 省下一次 N 长数组的分配/释放与随之而来的页错误。
            S -= imf

            if self.store_iter_info:
                self.temp_imf_arr.append(this_temp_imfs)
                self.iter_convergence.append(this_iter_convergences)

            if self.num_imf != -1 and num_imf >= self.num_imf:
                self.res = S
                break

        self.imfs = self.imfs[:num_imf, :]

        config = self.config_generate()
        info = {"iterations": iterations, "final_sd": final_sds}
        if self.store_iter_info:
            info["iter_temp_imf"] = self.temp_imf_arr
            info["iter_convergence"] = self.iter_convergence

        return DecompositionResult(
            IMFs=self.imfs,
            Res=self.res,
            info=info,
            config=config
        )

    def end(self, h: np.ndarray, h_old: np.ndarray, up: np.ndarray, dn: np.ndarray) -> bool:
        """
        ``faster=False`` 的额外放行判定 (仅供 ``decompose`` 在 SD 判据已通过后调用)。

        Cauchy SD 只说明"相邻两次筛分不再显著变化", 并不保证筛出来的东西是 IMF。
        本方法参照 OLD 0.3.0 与 PyEMD 1.9 的内层停筛条件, 再补三道 IMF 必要条件,
        全部通过才允许该阶收敛:

        1. **窄带平衡** ``|过零数 − 极值数| <= 1``
           —— OLD ``faster=False`` 的窄带门, 也等价于 PyEMD 的 ``f2 = |extNo − nzm| < 2``。
           过零用 PyEMD ``indzer`` 口径 (严格符号积 < 0, 零点段按段计), 极值用库内
           ``Utils.Peaks`` 规则, 与 OLD ``_narrowband_ok`` 逐字一致。
        2. **包络异号** ``up >= 0`` 且 ``dn <= 0``
           —— PyEMD ``check_imf`` 的首要条件 (局部极大必须为正、局部极小必须为负)。
        3. **收敛量级** ``svar = ||h_old − h||² / (max(h_old) − min(h_old)) < 0.001``
           或 ``std = Σ((h_old − h) / h)² < 0.2``
           —— PyEMD ``check_imf`` 的两个主判据 (其 ``energy_ratio`` 已由本函数的调用方
           用 Cauchy SD 承担, 故不重复)。

        刻意不含 PyEMD 的 ``sum(imf_new²) < 1e-10`` 绝对下限: 那是绝对能量阈值, 会像
        ``res_energy`` 一样破坏幅值尺度不变性; 全零/极小幅度分量由 ``decompose`` 外层
        的"无进展保护"拦截。

        Parameters
        ----------
        h : 当前候选 (已做完本轮 ``h <- h - mean``)
        h_old : 上一轮候选
        up, dn : 本轮的上 / 下包络 (必须是未就地改写过的原值)

        Returns
        -------
        bool
            True = 允许该阶收敛; False = 继续筛分直到 ``internal_iter``。
        """
        # --- 1) 窄带平衡: |过零 − 极值| <= 1 --------------------------------- #
        s1, s2 = h[:-1], h[1:]
        zc = int(np.sum(s1 * s2 < 0))
        if np.any(h == 0):
            indz = np.nonzero(h == 0)[0]
            if np.any(np.diff(indz) == 1):          # 存在零点段才做段合并 (PyEMD 同款条件)
                zero = h == 0
                dz = np.diff(np.concatenate(([0], zero, [0])))
                zc += int(np.nonzero(dz == 1)[0].size)

        m_idx, _ = find_peaks(h, mod="max", safe="safe", backend=self.peaks_backend)
        n_idx, _ = find_peaks(h, mod="min", safe="safe", backend=self.peaks_backend)
        if abs(zc - (m_idx.size + n_idx.size)) > 1:
            return False

        # --- 2) 包络异号 ------------------------------------------------------ #
        if np.any(up < 0) or np.any(dn > 0):
            return False

        # --- 3) PyEMD 的缩放方差 / 标准差 ------------------------------------- #
        delta = h_old - h
        span = float(np.max(h_old) - np.min(h_old))
        if span <= 0.0:                             # 上一步已是常量: 无从归一, 直接放行
            return True
        if float(np.dot(delta, delta)) / span < 0.001:
            return True
        with np.errstate(divide="ignore", invalid="ignore"):
            std = float(np.sum((delta / h) ** 2))
        return std < 0.2


def f_EMD(S: list | np.ndarray, *, num_imf: int = -1, nbsym: int = 4, internal_delta: float = 0.2, internal_iter: int = 50, res_energy: float = 1e-6, faster: bool = True, store_iter_info: bool = False, **kwargs):
    emd = EMD(
        num_imf=num_imf,
        nbsym=nbsym,
        internal_delta=internal_delta,
        internal_iter=internal_iter,
        res_energy=res_energy,
        faster=faster,
        store_iter_info=store_iter_info,
        **kwargs
    )

    return emd.decompose(S)

"""
other function for module
"""
