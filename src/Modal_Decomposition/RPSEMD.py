import numpy as np
from dataclasses import dataclass
from typing import List, Tuple

from .Define import Config, Decomposer, DecompositionResult, DEFAULT_NUMPY_TYPE, SAFE_MODS, SAFE_ARGS
from .Base import Cache
from .Utils import Check, monotonic, mirror, find_peaks, Check_Time_and_Signal, hilbert, HILBERT_MODS, HILBERT_MODS_ARGS

from .EMD import EMD, EMDConfig

"""
Config
"""
@dataclass(frozen=True, kw_only=True)
class RPSEMDConfig(Config):
    num_imf: int
    separation_ratio: float
    max_clusters: int
    sin_smaple_num: int
    res_energy: float
    standard: bool
    cluster_weight: float
    cluster_max_samples: int
    hilbert_mod: HILBERT_MODS | HILBERT_MODS_ARGS
    dtype: type | np.dtype | None
    EMD_config: EMDConfig

"""
Class and Function of decomposer
"""
class RPSEMD(Decomposer):
    name = "RPSEMD"

    def __init__(self, num_imf: int = -1, separation_ratio: float = 1.5, max_clusters: int = 4, sin_smaple_num: int = 8, res_energy: float = 1e-6, standard: bool = True, cluster_weight: float = 0.0, cluster_max_samples: int = 512, hilbert_mod: HILBERT_MODS | HILBERT_MODS_ARGS = "FFT", copy: bool = True, store_more_info: bool = False, **kwargs):
        self.num_imf: int = num_imf
        self.separation_ratio: float = separation_ratio
        self.max_clusters: int = max_clusters  # `-1` means while True | `None` means equal to S.size(length) | `int` means limited
        self.sin_smaple_num: int = sin_smaple_num
        self.res_energy: float = res_energy
        self.standard: bool = standard
        self.cluster_weight: float = cluster_weight  # 选簇时的能量权重门限: 权重 = 簇大小 × 平均幅值; 只有 >= cluster_weight × 总权重 的簇才参与"取最高频簇"。0 = 不过滤 (原行为)。
        self.cluster_max_samples: int = cluster_max_samples  # 聚类前的抽稀上限: linkage 是 O(N^2), 而聚类只用来找中心频率, 均匀抽稀即可。
        self.hilbert_mod: HILBERT_MODS | HILBERT_MODS_ARGS = hilbert_mod
        self.copy: bool = copy
        self.store_more_info: bool = store_more_info

        unknown = set(kwargs) - {
            "dtype", "safe",
            "emd_nbsym", "emd_internal_delta", "emd_internal_iter",
            "emd_res_energy", "emd_faster", "emd_store_iter_info", "emd_pre_num_imfs",
            "emd_dtype", "emd_safe", "emd_need_squeeze", "emd_mirror_mod",
            "emd_poly_order", "emd_time_poly_order", "emd_peaks_backend",
        }
        if unknown:
            raise TypeError(f"unexpected keyword arguments: {unknown}")

        self.dtype: type | np.dtype | None = kwargs.get("dtype", None)
        self.safe: SAFE_MODS | SAFE_ARGS = kwargs.get("safe", "unknown")

        self.emd_num_imf: int = 1
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

        self.imfs: np.ndarray = None
        self.res:  np.ndarray = None

    def param_check(self) -> None:
        if not isinstance(self.num_imf, int) or isinstance(self.num_imf, bool):
            raise TypeError("num_imf must be an int")

        if self.num_imf != -1:
            if not (self.num_imf > 0):
                raise ValueError("num_imf must >0")

        self.hilbert_mod = Check.validate_mod(self.hilbert_mod, HILBERT_MODS)

        if not isinstance(self.copy, bool):
            raise TypeError("copy must be bool")

        if self.dtype is not None and np.dtype(self.dtype).kind != "f":
            raise ValueError("the `dtype` must be a float numpy dtype or None.")

        self.safe = Check.validate_safe(self.safe)

        if not isinstance(self.separation_ratio, (int, float)) or isinstance(self.separation_ratio, bool):
            raise TypeError("separation_ratio must be int or float")

        if not (self.separation_ratio > 0):
            raise ValueError("the separation_ratio must be > 0, the best is `1.5` provided from paper")

        if not isinstance(self.standard, bool):
            raise TypeError("standard must be bool")

        if not isinstance(self.cluster_weight, (int, float)) or isinstance(self.cluster_weight, bool):
            raise TypeError("the `cluster_weight` must be int or float")

        if not (0 <= self.cluster_weight < 1):
            raise ValueError("the `cluster_weight` must be in [0, 1)")

        if not isinstance(self.cluster_max_samples, int) or isinstance(self.cluster_max_samples, bool):
            raise TypeError("the `cluster_max_samples` must be int")

        if self.cluster_max_samples < 0:
            raise ValueError("the `cluster_max_samples` must be >= 0")

        if self.max_clusters is not None:
            if not isinstance(self.max_clusters, int) or isinstance(self.max_clusters, bool):
                raise TypeError("max_clusters must be int or None")

            # -1 = 不设上限 (逐个试到满足分离条件); None = 取信号长度; 其余必须 >= 2
            if self.max_clusters != -1 and self.max_clusters < 2:
                raise ValueError("the `max_clusters` must be -1, None, or >= 2")

        if not isinstance(self.sin_smaple_num, int) or isinstance(self.sin_smaple_num, bool):
            raise TypeError("sin_smaple_num must be int")

        if not (self.sin_smaple_num > 0):
            raise ValueError("sin_smaple_num must be >0")

        if not isinstance(self.res_energy, float):
            raise TypeError("the res_energy must be float")

        if not (1 > self.res_energy >= 0):
            raise ValueError("the res_energy must >= 0 and < 1")

        if not isinstance(self.store_more_info, bool):
            raise TypeError("the store_more_info must be bool")

    def config_generate(self) -> Config:
        return RPSEMDConfig(
            num_imf=self.num_imf,
            separation_ratio=self.separation_ratio,
            max_clusters=self.max_clusters,
            sin_smaple_num=self.sin_smaple_num,
            res_energy=self.res_energy,
            standard=self.standard,
            cluster_weight=self.cluster_weight,
            cluster_max_samples=self.cluster_max_samples,
            hilbert_mod=self.hilbert_mod,
            dtype=self.dtype,
            EMD_config=self.EMD.config_generate()
        )


    def decompose(self, S: List[int | float] | np.ndarray) -> DecompositionResult:
        S, T, N = Check_Time_and_Signal(S, T=None, ndim={1}, method=self.name, dtype=self.dtype, need_T=True, safe=self.safe, copy=self.copy)

        if not (S.dtype.kind == "f" and S.dtype.itemsize >= 4):  # int/bool 会让 S -= float 直接报错; f16 会让内层 EMD 的样条出非有限值
            S = Check.to_dtype(S, np.dtype(np.float64))

        # 每次调用都从干净状态开始
        self.imfs = None
        self.res = None

        if self.num_imf != -1:
            self.imfs = np.empty((self.num_imf, N), dtype=S.dtype)

        else:
            self.imfs = []

        original_energy = np.dot(S, S)

        waves = np.empty((self.sin_smaple_num, N), dtype=S.dtype)

        cluster_infos: List[dict] = []
        each_center_amp: list = []
        each_center_freq: list = []

        if self.store_more_info:
            each_phases: list = []
            each_amp: list = []
            each_freq: list = []

        # `max_clusters=None` 表示取信号长度; 用局部变量, 不要改写实例状态
        max_clusters: int = S.size if self.max_clusters is None else self.max_clusters

        n_imf = 0
        while True:
            imf_result = self.EMD.decompose(S).IMFs
            if imf_result.shape[0] < 1:      # 内层 EMD 一阶也剥不出 -> 残差已不可分
                self.res = S
                break
            IMF = imf_result.squeeze()

            analytic_signal = hilbert(IMF, mod=self.hilbert_mod)

            amp, freq = self.get_instantaneous_features(analytic_signal)

            # 规范: 在**第一阶初步 IMF 的极值点**上取瞬时幅值/频率再去聚类, 而不是全样本。
            # 非极值点处的相位没有物理意义, 把它们纳入聚类只会把伪簇拉进来; 而且极值点
            # 数量远少于 N, linkage 的 O(N^2) 也就自然消失了。
            ext_x, _, _ = find_peaks(IMF, mod="both", safe="safe")
            if ext_x.size >= 4:
                amp_cluster, freq_cluster = amp[ext_x], freq[ext_x]
            else:                                  # 极值点太少, 退回全样本
                amp_cluster, freq_cluster = amp, freq

            # 兜底抽稀: 极值点仍很多时(长信号 + 高频分量)才生效
            if 0 < self.cluster_max_samples < amp_cluster.size:
                step = int(np.ceil(amp_cluster.size / self.cluster_max_samples))
                freq_cluster, amp_cluster = freq_cluster[::step], amp_cluster[::step]

            # 注意顺序: hierarchical_clustering(x, y) 里 x 是被聚类的**频率**、y 是**幅值**。
            # 传反了会把"平均幅值"当成中心频率(>Nyquist)、把"平均瞬时频率"当成正弦幅值。
            center_freq, center_amp, cluster_info = self.hierarchical_clustering(freq_cluster, amp_cluster, max_clusters=max_clusters, compare=self.separation_ratio, standard=self.standard, cluster_weight=self.cluster_weight)

            # 规范: θ_i = (2π/I)·i, i = 0..I-1。用 linspace(0, 2π, I) 会让末相位与首相位重复,
            # 于是 Σ cos(2πft+θ_i) ≠ 0, 辅助正弦无法在集合平均中**精确抵消**。
            phases = np.arange(self.sin_smaple_num) * (2.0 * np.pi / self.sin_smaple_num)
            for p in range(self.sin_smaple_num):
                waves[p, :] = center_amp * np.cos(2 * np.pi * center_freq * T + phases[p])

            cluster_infos.append(cluster_info)
            each_center_amp.append(center_amp)
            each_center_freq.append(center_freq)

            if self.store_more_info:
                each_phases.append(phases)
                each_amp.append(amp)
                each_freq.append(freq)

            row_imf: np.ndarray = None
            contrib: int = 0
            for p in range(self.sin_smaple_num):
                S_added = S + waves[p, :]

                imf_p = self.EMD.decompose(S_added).IMFs
                if imf_p.shape[0] < 1:
                    continue
                contrib += 1
                if row_imf is None:
                    row_imf = np.array(imf_p.squeeze(), copy=True)
                else:
                    row_imf += imf_p.squeeze()

            if row_imf is None:              # 所有相位都剥不出 IMF -> 残差已不可分
                self.res = S
                break

            row_imf /= contrib
            if self.num_imf == -1:
                self.imfs.append(row_imf)
            else:
                self.imfs[n_imf, :] = row_imf

            n_imf += 1
            S -= row_imf

            if self.num_imf != -1 and n_imf >= self.num_imf:
                self.res = S
                break

            # 停止判据: 残差单调 / 每方向极值不足 2 个 / 能量耗尽
            _, _, is_max = find_peaks(S, mod="both", safe="safe")
            if monotonic(S, mod="monotonic", safe="safe") \
                    or int(np.sum(is_max)) < 2 or int(np.sum(~is_max)) < 2 \
                    or np.dot(S, S) <= self.res_energy * original_energy:
                self.res = S
                break

        if self.num_imf == -1:
            self.imfs = np.asarray(self.imfs, dtype=S.dtype) if len(self.imfs) \
                else np.empty((0, N), dtype=S.dtype)
        else:
            # 提前 break (退化输入等) 时缓冲区里没写的行必须切掉, 否则返回未初始化的内存
            self.imfs = self.imfs[:n_imf, :]

        config = self.config_generate()
        info = {
            "cluster_infos": cluster_infos,
            "each_center_amp": each_center_amp,
            "each_center_freq": each_center_freq,
            "each_phases": each_phases if self.store_more_info else None,
            "each_amp": each_amp if self.store_more_info else None,
            "each_freq": each_freq if self.store_more_info else None,
            "rows": n_imf,
        }

        return DecompositionResult(
            IMFs=self.imfs,
            Res=self.res,
            info=info,
            config=config
        )


    @staticmethod
    def get_instantaneous_features(complex_signal: np.ndarray[..., np.complexfloating]) -> Tuple[np.ndarray, np.ndarray]:
        amp = np.abs(complex_signal)

        phase = np.unwrap(np.angle(complex_signal))
        freq = np.diff(phase) / (2.0 * np.pi)

        freq = np.append(freq, freq[-1])

        return amp, freq


    @staticmethod
    def hierarchical_clustering(x: np.ndarray, y: np.ndarray, *, max_clusters: int, compare: float, standard: bool = True, cluster_weight: float = 0.0) -> Tuple[np.floating[int | float], np.floating[int | float], dict]:
        sch = Cache.import_module("scipy.cluster.hierarchy")

        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)

        # standard=True 只在**聚类时**用 z 分数; 对外返回的中心频率必须换回物理量纲
        if standard:
            x_mean = float(x.mean())
            x_std = float(x.std()) if float(x.std()) > 0 else 1.0
            x_cluster = (x - x_mean) / x_std
        else:
            x_mean, x_std = 0.0, 1.0
            x_cluster = x

        # linkage 要 (n_samples, n_features); 一维数组会被当成 condensed distance matrix
        Z = sch.linkage(x_cluster.reshape(-1, 1), method="ward")

        best_n_clusters = None
        best_labels = None

        if max_clusters == -1:
            n_clusters = 2
            # 上限 = 样本数: 再往上 fcluster 也不可能分成更多簇, 否则会死循环
            while n_clusters <= max(2, x_cluster.size):
                labels = sch.fcluster(Z, n_clusters, criterion='maxclust')

                # 计算每个簇的中心频率和平均幅值
                cluster_info = {}
                for label in set(labels):
                    mask = labels == label
                    cluster_x = x_cluster[mask]
                    cluster_y = y[mask]

                    center_x = np.mean(cluster_x)
                    avg_y = np.mean(cluster_y)

                    cluster_info[label] = \
                    {
                        'center_x': center_x,
                        'avg_y': avg_y,
                        'size': np.sum(mask)
                    }

                # 检查所有簇对是否满足分离条件
                all_separated = True
                labels_list = list(cluster_info.keys())
                for i in range(len(labels_list)):
                    for j in range(i + 1, len(labels_list)):
                        x_i = cluster_info[labels_list[i]]['center_x']
                        x_j = cluster_info[labels_list[j]]['center_x']

                        lo, hi = min(x_i, x_j), max(x_i, x_j)
                        ratio = hi / lo if abs(lo) > 1e-12 else np.inf
                        if ratio <= compare:  # 不满足 >compare 的条件
                            all_separated = False
                            break
                    if not all_separated:
                        break

                if all_separated:
                    best_n_clusters = n_clusters
                    best_labels = labels
                    break

                n_clusters += 1

            if best_labels is None:          # 到上限仍不满足 -> 退化成 2 簇
                best_n_clusters = 2
                best_labels = sch.fcluster(Z, best_n_clusters, criterion='maxclust')

        else:
            for n_clusters in range(2, max_clusters):
                labels = sch.fcluster(Z, n_clusters, criterion='maxclust')

                # 计算每个簇的中心频率和平均幅值
                cluster_info = {}
                for label in set(labels):
                    mask = labels == label
                    cluster_x = x_cluster[mask]
                    cluster_y = y[mask]

                    center_x = np.mean(cluster_x)
                    avg_y = np.mean(cluster_y)

                    cluster_info[label] = \
                        {
                            'center_x': center_x,
                            'avg_y': avg_y,
                            'size': np.sum(mask)
                        }

                # 检查所有簇对是否满足分离条件
                all_separated = True
                labels_list = list(cluster_info.keys())
                for i in range(len(labels_list)):
                    for j in range(i + 1, len(labels_list)):
                        x_i = cluster_info[labels_list[i]]['center_x']
                        x_j = cluster_info[labels_list[j]]['center_x']

                        lo, hi = min(x_i, x_j), max(x_i, x_j)
                        ratio = hi / lo if abs(lo) > 1e-12 else np.inf
                        if ratio <= compare:  # 不满足 >compare 的条件
                            all_separated = False
                            break
                    if not all_separated:
                        break

                if all_separated:
                    best_n_clusters = n_clusters
                    best_labels = labels
                    break

            if best_n_clusters is None:
                best_n_clusters = 2
                best_labels = sch.fcluster(Z, best_n_clusters, criterion='maxclust')

        # 找到频率最高的簇（目标内在模态）。注意: 簇划分用的是 z 分数,
        # 但目标簇的"中心频率/平均幅值"必须回到**物理量纲** (x 本身)。
        #
        # cluster_weight > 0 时先按"能量权重"过滤: 权重 = 簇大小 × 平均幅值。Hilbert 变换的
        # 边界跳变会在 Nyquist 附近造出**样本少、幅值尖**的小簇, 它常常是最高频簇, 于是被
        # 当成目标频率 —— 过滤掉权重可忽略的簇即可避开。cluster_weight=0 时与原行为逐位相同。
        cluster_x = {}
        cluster_w = {}
        for label in set(best_labels):
            mask = best_labels == label
            cluster_x[label] = float(np.mean(x[mask]))
            cluster_w[label] = float(np.sum(mask)) * float(np.mean(y[mask]))

        if cluster_weight > 0:
            total_w = float(sum(cluster_w.values()))
            if total_w > 0:
                kept = {lab: f for lab, f in cluster_x.items()
                        if cluster_w[lab] >= cluster_weight * total_w}
                if kept:
                    cluster_x = kept

        target_label = max(cluster_x, key=cluster_x.get)
        target_mask = best_labels == target_label

        # 计算目标簇的参数
        center_x = np.mean(x[target_mask])  # 中心频率 (cycles/sample)
        avg_y    = np.mean(y[target_mask])  # 平均幅值

        return center_x, avg_y, {"matrix": Z, "best_labels": best_labels, "target_label": target_label}