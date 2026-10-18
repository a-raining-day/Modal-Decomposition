import numpy as np
from dataclasses import dataclass
from typing import List

from .EMD import EMD, EMDConfig

from .Define import Config, Decomposer, DecompositionResult, DEFAULT_NUMPY_TYPE
from .Base import Cache, set_allocable
from .Utils import Check, monotonic, mirror, find_peaks, Check_Time_and_Signal, PEAKS_BACKEND

"""
Config
"""
@dataclass(frozen=True, kw_only=True)
class CEEMDANConfig(Config):
    num_imf: int
    noise_amp: float
    M: int
    pre_decompose_noise: bool
    parallel: bool
    num_workers: int
    dtype: type | np.dtype | None
    EMD_config: EMDConfig
    pre_decompose_EMD_config: EMDConfig | None

"""
Class and Function of decomposer
"""
class CEEMDAN(Decomposer):
    name = "CEEMDAN"

    def __init__(self, num_imf: int = -1, noise_amp: float = 0.3, res_energy: float = 1e-6, M: int = 50, pre_decompose_noise: bool = False, parallel: bool = False, num_workers: int = 1, faster: bool = True, copy: bool = True, **kwargs):
        self.num_imf: int = num_imf
        self.noise_amp: float = noise_amp
        self.res_energy: float = res_energy
        self.M: int = M
        self.pre_decompose_noise: bool = pre_decompose_noise
        self.parallel: bool = parallel
        self.num_workers: int = num_workers
        self.faster: bool = faster
        self.copy: bool = copy

        unknown = set(kwargs) - {
            "dtype", "peaks_backend",
            "emd_nbsym", "emd_internal_delta", "emd_internal_iter",
            "emd_res_energy", "emd_faster", "emd_store_iter_info", "emd_pre_num_imfs",
            "emd_dtype", "emd_safe", "emd_need_squeeze", "emd_mirror_mod",
            "emd_poly_order", "emd_time_poly_order", "emd_peaks_backend",
        }
        if unknown:
            raise TypeError(f"unexpected keyword arguments: {unknown}")

        self.dtype: type | np.dtype | None = kwargs.get("dtype", None)
        self.peaks_backend: str = kwargs.get("peaks_backend", "numpy")   # 本类自己 find_peaks 用的后端

        # 内层 EMD 只取 1 阶 (CEEMDAN 每次只剥一阶); 噪声预分解要留下 k-1 阶
        self.emd_num_imf: int = 1
        self.emd_pre_decompose_num_imf: int = (
            self.num_imf if self.num_imf == -1 else max(1, self.num_imf - 1)
        )
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

        # 噪声预分解器: pre_decompose_noise=True 时需要; parallel=True 也需要 (worker 无状态,
        # 每阶的噪声 IMF 必须在阶段循环开始前就备好)。这里无条件构造, 只在用不到时闲着。
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

        if not isinstance(self.noise_amp, (int, float)) or isinstance(self.noise_amp, bool):
            raise TypeError("noise_amp must be float or int")

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

        if not isinstance(self.faster, bool):
            raise TypeError("the `faster` must be bool")

        if not isinstance(self.copy, bool):
            raise TypeError("the `copy` must be bool")

        if not isinstance(self.res_energy, float):
            raise TypeError("the `res_energy` must be float.")

        if not (1 > self.res_energy >= 0):
            raise ValueError("the `res_energy` must >= 0.")

        self.peaks_backend = Check.validate_mod(self.peaks_backend, PEAKS_BACKEND)

    def config_generate(self) -> CEEMDANConfig:
        return CEEMDANConfig(
            num_imf=self.num_imf,
            noise_amp=self.noise_amp,
            M=self.M,
            pre_decompose_noise=self.pre_decompose_noise,
            parallel=self.parallel,
            num_workers=self.num_workers,
            dtype=self.dtype,
            EMD_config=self.EMD.config_generate(),
            pre_decompose_EMD_config=self.pre_decompose_EMD.config_generate() if self.pre_decompose_noise else None
        )

    def decompose(self, S: List[int | float] | np.ndarray) -> DecompositionResult:
        """
        Complementary EEMD with Adaptive Noise (CEEMDAN)。

        第 1 阶      ``IMF_1 = (1/M) Σ_m E_1(S + w_m)``
        第 k (k>=2)  ``IMF_k = (1/M) Σ_m E_1(r_{k-1} + E_k(w_m))``,  ``r_k = r_{k-1} - IMF_k``

        ``E_1(·)`` 就是内层 ``EMD(num_imf=1)``, 每次只剥一阶; ``E_k(w_m)`` 是第 m 个噪声实现的
        第 k 阶 IMF。噪声口径与 EEMD/CEEMD 一致: ``noise_amp`` 是**幅值系数**, 实际噪声为
        ``noise_amp × N(0, std(S))``。因此第 1 阶加的是原始噪声、第 k 阶加的是噪声自己的第 k 阶
        IMF —— 两者都不再额外缩放 (原实现里的 ``noises[m] / noise_amp * S.std()`` 会把系数约掉,
        等于该参数失效, 且与第 k 阶不缩放的做法不自洽)。
        """
        S, N = Check_Time_and_Signal(S, T=None, ndim={1}, method=self.name, dtype=self.dtype, need_T=False, copy=self.copy)

        if not (S.dtype.kind == "f" and S.dtype.itemsize >= 4):  # int/bool 输入会在 avg_env /= 2 处直接 UFuncTypeError, f16 会让样条报非有限值
            S = Check.to_dtype(S, np.dtype(np.float64))

        # 每次调用都从干净状态开始: 否则第二次 decompose 会把上一次的累积结果当成初值
        self.imfs = None
        self.res  = None

        noise_std = self.noise_amp * S.std()          # 实际噪声标准差, 便于追溯

        # 退化输入 (常量/全零/线性趋势/单峰/过短): 信号本身极大或极小不足 2 个, 内层 EMD 一个 IMF
        # 也分不出来; 若照常加噪, 分出来的全是"我们加进去的噪声"。直接返回空 IMF 集, 与 EMD 对齐。
        _, _, peak_max = find_peaks(S, mod="both", safe="safe", backend=self.peaks_backend)
        if np.sum(peak_max) < 2 or np.sum(~peak_max) < 2:
            self.imfs = np.empty((0, N), dtype=S.dtype)
            self.res = S
            return DecompositionResult(
                IMFs=self.imfs,
                Res=self.res,
                info={"noise_std": noise_std,
                      "emd info": [], "noise emd info": [], "rows": []},
                config=self.config_generate()
            )

        original_energy = np.dot(S, S)

        # 噪声按输入精度生成: np.random.normal 是 float64, 不转会把 f32 工作数组整体升格
        noises: np.ndarray = np.random.normal(loc=0, scale=noise_std, size=(self.M, N)).astype(S.dtype, copy=False)

        EMD_info: List[dict] = []
        noisy_EMD_info: List[dict] = []
        rows: List[int] = []

        # num_imf=-1 与 EMD / EEMD / CEEMD 完全一致: **不设阶段上限**, 阶数由停止判据决定
        # (内层 EMD 剥不出 IMF / 残差单调 / 能量耗尽 / faster=False 的极值门)。
        # 显式 num_imf 时, 在阶数到点后 break。
        # 保底: 逐阶渐进的噪声序列是有限的 (每剥一阶都在缩小, 直到内层 EMD 一阶也剥不出),
        # 所以循环必然终止, 不需要人为数字。

        # 单次 IMF 的"无进展"门限: 与 EMD 的口径一致 (相对信号的均方根)
        scale_floor: float = 1e-12 * max(np.sqrt(original_energy / N), np.finfo(float).tiny)

        if not self.parallel:
            noise_imfs: np.ndarray = None       # 只有 pre_decompose_noise=True 时才有

            if self.pre_decompose_noise:
                stacked: List[np.ndarray] = []
                min_rows: int = None

                for m in range(self.M):
                    noise_result = self.pre_decompose_EMD.decompose(noises[m, :])
                    noisy_EMD_info.append(noise_result.info)

                    imfs_m = noise_result.IMFs
                    min_rows = imfs_m.shape[0] if min_rows is None else min(min_rows, imfs_m.shape[0])

                    stacked.append(imfs_m)

                if min_rows is None or min_rows < 1:
                    self.imfs = np.empty((0, N), dtype=S.dtype)
                    self.res = S
                    return DecompositionResult(
                        IMFs=self.imfs,
                        Res=self.res,
                        info={"noise_std": noise_std, "emd info": EMD_info, "noise emd info": noisy_EMD_info, "rows": rows},
                        config=self.config_generate()
                    )

                # [M, min rows, N] —— 各实现的阶数可能不同, 统一截到最小的那个;
                # 第 k 阶要用噪声的第 k-1 阶, 所以最多只能做到 min_rows + 1 阶
                noise_imfs = np.array(stacked)[:, :min_rows, :].astype(S.dtype, copy=False)

            # 显式 num_imf 时预分配, 峰值只有一份 (num_imf, N);
            # num_imf=-1 时阶数未知, 只能用 list 暂存, 最后一次性成阵
            imfs_buf: np.ndarray = np.empty((self.num_imf, N), dtype=S.dtype) if self.num_imf != -1 else None
            imfs: List[np.ndarray] = []
            n_stage: int = 0
            S_work: np.ndarray = S          # 逐步剥 IMF 的残差; S 是 Check_Time_and_Signal(copy=True) 给的副本, 可原地 -=

            while True:
                row_imf: np.ndarray = None
                contrib: int = 0

                for m in range(self.M):
                    if n_stage == 0:
                        # 第 1 阶: 直接加原始噪声
                        result = self.EMD.decompose(S_work + noises[m, :])
                        EMD_info.append(result.info)
                        imf_m = result.IMFs
                    elif self.pre_decompose_noise:
                        # 第 k 阶: 加噪声自己的第 k-1 阶 IMF
                        result = self.EMD.decompose(S_work + noise_imfs[m, n_stage - 1, :])
                        EMD_info.append(result.info)
                        imf_m = result.IMFs
                    else:
                        noise_result = self.EMD.decompose(noises[m, :])
                        noisy_EMD_info.append(noise_result.info)
                        imf_noise = noise_result.IMFs
                        noises[m, :] = noise_result.Res

                        if imf_noise.shape[0] < 1:
                            continue

                        result = self.EMD.decompose(S_work + imf_noise[0, :])
                        EMD_info.append(result.info)
                        imf_m = result.IMFs

                    if imf_m.shape[0] < 1:
                        continue                    # 该实现这一阶剥不出来, 跳过
                    contrib += 1
                    row_imf = imf_m if row_imf is None else row_imf + imf_m

                if row_imf is None:
                    break

                # 每阶恰好剥出 1 个 IMF: squeeze 成一维, 否则 S_work(1-D) - row_imf(1,N) 会被广播成 2-D
                imf_row = row_imf.squeeze() / self.M

                # 无进展就收尾 (内层 EMD 一阶也剥不出时 row_imf 会先变成 None)
                if np.max(np.abs(imf_row)) <= scale_floor:
                    break

                if imfs_buf is not None:
                    imfs_buf[n_stage, :] = imf_row
                else:
                    imfs.append(imf_row)
                rows.append(contrib)
                S_work -= imf_row
                n_stage += 1

                # 显式 num_imf 到点就停 (-1 时永不触发)
                if self.num_imf != -1 and n_stage >= self.num_imf:
                    break

                # 预分解噪声: 第 k 阶要用噪声的第 k-1 阶, 所以第 0..min_rows 阶都可用
                if self.pre_decompose_noise and n_stage > noise_imfs.shape[1]:
                    break

                stop = monotonic(S_work, mod="monotonic", safe="safe") or np.dot(S_work, S_work) < self.res_energy * original_energy
                if not self.faster and not stop:
                    _, _, pm_res = find_peaks(S_work, mod="both", safe="safe", backend=self.peaks_backend)
                    stop = int(np.sum(pm_res)) < 2 or int(np.sum(~pm_res)) < 2
                if stop:
                    break

            self.imfs = imfs_buf[:n_stage, :] if imfs_buf is not None else \
                (np.asarray(imfs, dtype=S.dtype) if imfs else np.empty((0, N), dtype=S.dtype))
            # 无论跑到 num_imf / 噪声阶数用完, 还是被各种停止判据 break, S_work 就是当前残差
            self.res = S_work

        else:
            import multiprocessing as mp

            # 噪声必须先整体预分解: 本分支每阶 spawn 一批 worker, 而 spawn 出来的进程不带状态,
            # 拿不到"上一阶留下的噪声残差", 所以只能一次性把每个噪声实现的前 k-1 阶 IMF 都备好。
            stacked: List[np.ndarray] = []
            min_rows: int = None

            for m in range(self.M):
                noise_result = self.pre_decompose_EMD.decompose(noises[m, :])
                noisy_EMD_info.append(noise_result.info)

                imfs_m = noise_result.IMFs
                min_rows = imfs_m.shape[0] if min_rows is None else min(min_rows, imfs_m.shape[0])

                stacked.append(imfs_m)

            if min_rows is None or min_rows < 1:
                self.imfs = np.empty((0, N), dtype=S.dtype)
                self.res = S
                return DecompositionResult(
                    IMFs=self.imfs,
                    Res=self.res,
                    info={"noise_std": noise_std, "emd info": EMD_info, "noise emd info": noisy_EMD_info, "rows": rows},
                    config=self.config_generate()
                )

            # [M, min rows, N] —— 第 k 阶要用噪声的第 k-1 阶, 所以最多只能做到 min_rows + 1 阶
            noise_imfs: np.ndarray = np.stack([row[:min_rows] for row in stacked]).astype(S.dtype, copy=False)

            # 试验按 M / num_workers 均分; 进程数不超过试验数 (否则进程会空转)
            num_workers: int = min(self.num_workers, self.M)
            base, extra = divmod(self.M, num_workers)
            tasks: List[dict] = []
            left = 0
            for p in range(num_workers):
                right = left + base + (1 if p < extra else 0)
                tasks.append({"index": p, "start": left, "end": right})
                left = right

            # 显式 num_imf 时预分配, 峰值只有一份 (num_imf, N);
            # num_imf=-1 时阶数未知, 只能用 list 暂存, 最后一次性成阵
            imfs_buf: np.ndarray = np.empty((self.num_imf, N), dtype=S.dtype) if self.num_imf != -1 else None
            imfs: List[np.ndarray] = []
            n_stage: int = 0
            S_work: np.ndarray = S

            # 与串行分支的差别只在"本阶由谁算": 相邻阶是串行依赖的 (第 k 阶要第 k-1 阶的全局
            # 平均残差), 所以不能像 EEMD/CEEMD 那样把整个分解交给一个 worker; 这里每阶 spawn
            # 一批 worker, 收齐本阶的 M 个 E_1 之和后再进入下一阶。
            from queue import Empty

            while True:
                # 第 1 阶给原始噪声, 其后给噪声自己的第 k-1 阶 IMF
                noise_block_all: np.ndarray = noises if n_stage == 0 else noise_imfs[:, n_stage - 1, :]

                queue = mp.Queue()
                parallel_process: List[mp.Process | None] = [None for _ in range(num_workers)]
                for task in tasks:
                    parallel_process[task["index"]] = mp.Process(
                        target=self.parallel_decompose,
                        args=(S_work, noise_block_all[task["start"]:task["end"], :], queue, task))
                    parallel_process[task["index"]].start()

                # 用"轮询 + 存活检查": 超时只是轮询间隔, 不对健康进程设截止时间, 所以耗时长的正常
                # 试验不会被误杀; 一旦发现 worker 已退出, 立刻清场并抛错。
                parallel_result_points_list: List = []
                worker_errors: List = []
                while len(parallel_result_points_list) + len(worker_errors) < num_workers:
                    try:
                        item = queue.get(timeout=0.5)
                    except Empty:
                        if any(p is not None and not p.is_alive() for p in parallel_process):
                            # 把已回传的记录先捞干净, 再统一报错
                            while True:
                                try:
                                    item = queue.get_nowait()
                                except Exception:
                                    break
                                if isinstance(item, tuple) and item and item[0] == "error":
                                    worker_errors.append(item)
                                else:
                                    parallel_result_points_list.append(item)
                            for p in parallel_process:
                                if p is not None and p.is_alive():
                                    p.terminate()
                            for p in parallel_process:
                                if p is not None:
                                    p.join()
                            codes = [p.exitcode for p in parallel_process if p is not None]
                            raise RuntimeError(
                                f"CEEMDAN: worker 异常退出, exitcode={codes}; 已回传的错误={worker_errors or '无'}") from None
                        continue
                    if isinstance(item, tuple) and item and item[0] == "error":
                        worker_errors.append(item)
                    else:
                        parallel_result_points_list.append(item)

                for p in range(num_workers):
                    parallel_process[p].join()
                queue.close()

                codes = [p.exitcode for p in parallel_process if p is not None]
                if worker_errors or any(c not in (0, None) for c in codes):
                    raise RuntimeError(f"CEEMDAN: worker 报错, exitcode={codes}; 错误={worker_errors or '无'}")

                parallel_result_points_list.sort(key=lambda it: it[0])

                row_imf: np.ndarray = None
                contrib: int = 0
                for item in parallel_result_points_list:
                    EMD_info.extend(item[3])
                    contrib += int(item[2])

                    acc = item[1]
                    if acc is None or acc.shape[0] < 1:
                        continue
                    if row_imf is None:
                        row_imf = acc
                    else:
                        min_row = min(row_imf.shape[0], acc.shape[0])
                        row_imf = row_imf[:min_row] + acc[:min_row]

                if row_imf is None:
                    break

                imf_row = row_imf.squeeze() / self.M

                if np.max(np.abs(imf_row)) <= scale_floor:
                    break

                if imfs_buf is not None:
                    imfs_buf[n_stage, :] = imf_row
                else:
                    imfs.append(imf_row)
                rows.append(contrib)
                S_work -= imf_row
                n_stage += 1

                # 显式 num_imf 到点就停 (-1 时永不触发)
                if self.num_imf != -1 and n_stage >= self.num_imf:
                    break
                # 本分支必须预分解噪声: 第 k 阶要用噪声的第 k-1 阶, 所以第 0..min_rows 阶都可用
                if n_stage > noise_imfs.shape[1]:
                    break

                stop = monotonic(S_work, mod="monotonic", safe="safe") or \
                    np.dot(S_work, S_work) < self.res_energy * original_energy
                if not self.faster and not stop:
                    _, _, pm_res = find_peaks(S_work, mod="both", safe="safe", backend=self.peaks_backend)
                    stop = int(np.sum(pm_res)) < 2 or int(np.sum(~pm_res)) < 2
                if stop:
                    break

            self.imfs = imfs_buf[:n_stage, :] if imfs_buf is not None else \
                (np.asarray(imfs, dtype=S.dtype) if imfs else np.empty((0, N), dtype=S.dtype))
            self.res = S_work

        config = self.config_generate()
        info = {
            "noise_std": noise_std,
            "emd info": EMD_info,
            "noise emd info": noisy_EMD_info,
            "rows": rows,
        }

        return DecompositionResult(
            IMFs=self.imfs,
            Res=self.res,
            info=info,
            config=config
        )

    def parallel_decompose(self, S, noise_block, pointer, task):
        """
        单个工作进程: 只负责本阶噪声块 ``noise_block`` 中自己那一段 (task["start"] .. task["end"])。

        CEEMDAN 的相邻阶是**串行依赖**的 (第 k 阶要第 k-1 阶的全局平均残差), 所以不能像 EEMD/CEEMD
        那样把一个 worker 派去跑完整个分解; 父进程每阶 spawn 一批 worker, 收齐本阶的 M 个 E_1 之和
        再进入下一阶。平均由父进程统一做。

        与 EEMD/CEEMD 的 ``parallel_decompose`` 一样, 把 (序号, 累积和, 有效数, EMD info) 通过
        ``pointer`` (mp.Queue) 交回父进程。
        """
        acc = None
        infos = []
        n_ok = 0

        # 任何异常都必须先回传一条错误记录再抛: 否则父进程只能靠"进程没了"来判断, 拿不到原因。
        # 回传之后 re-raise, 让子进程以非零 exitcode 结束, 父进程两条路都能捕获。
        try:
            for j in range(noise_block.shape[0]):
                result = self.EMD.decompose(S + noise_block[j, :])
                infos.append(result.info)

                imf_j = result.IMFs
                if imf_j.shape[0] < 1:
                    continue                    # 该实现这一阶剥不出来, 跳过
                n_ok += 1
                acc = imf_j if acc is None else acc + imf_j

            pointer.put((task["index"], acc, n_ok, infos))
        except BaseException as exc:
            pointer.put(("error", task["index"], f"{type(exc).__name__}: {exc}"))
            raise


def f_CEEMDAN(S: List[int | float] | np.ndarray, *, num_imf: int = -1, noise_amp: float = 0.3, res_energy: float = 1e-6, M: int = 50, pre_decompose_noise: bool = False, parallel: bool = False, num_workers: int = 1, faster: bool = True, **kwargs) -> DecompositionResult:
    ceemdan = CEEMDAN(
        num_imf=num_imf,
        noise_amp=noise_amp,
        res_energy=res_energy,
        M=M,
        pre_decompose_noise=pre_decompose_noise,
        parallel=parallel,
        num_workers=num_workers,
        faster=faster,
        **kwargs,
    )

    return ceemdan.decompose(S)
