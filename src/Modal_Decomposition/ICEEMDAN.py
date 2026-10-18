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
class ICEEMDANConfig(Config):
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
class ICEEMDAN(Decomposer):
    name = "ICEEMDAN"

    def __init__(self, num_imf: int = -1, noise_amp: float | int = 0.2, M: int = 100, res_energy: float = 1e-6, nbsym: int = 4, pre_decompose_noise: bool = False, parallel: bool = False, num_workers: int = 1, **kwargs) -> None:
        self.num_imf: int = num_imf
        self.noise_amp: float = noise_amp
        self.M: int = M
        self.res_energy: float = res_energy
        self.pre_decompose_noise: bool = pre_decompose_noise
        self.parallel: bool = parallel
        self.num_workers: int = num_workers
        self.nbsym: int = nbsym            # 局部均值算子 M(·) 的边界镜像极值个数

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
        self.peaks_backend: str = kwargs.get("peaks_backend", "numpy")

        # 内层 EMD 只取 1 阶; 第 n 阶要用噪声的第 n 阶 IMF, 所以预分解要备够 num_imf 阶
        self.emd_num_imf: int = 1
        self.emd_pre_decompose_num_imf: int = (
            self.num_imf if self.num_imf == -1 else max(1, self.num_imf)
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

        # 局部均值算子 M(x) = x - E_1(x): 用内层 EMD 把最高频筛到收敛再整体减掉,
        # 等价于"包络平均筛分到收敛", 而不是只做一次包络平均 (后者留下极值尺度的高频残渣)。
        # nbsym 只作用于这个算子 —— 边界镜像极值个数。
        self.local_mean_EMD = EMD(
            num_imf=1,
            nbsym=self.nbsym,
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

        if not isinstance(self.res_energy, float):
            raise TypeError("the `res_energy` must be float.")

        if not (self.res_energy >= 0):
            raise ValueError("the `res_energy` must >= 0.")

        self.peaks_backend = Check.validate_mod(self.peaks_backend, PEAKS_BACKEND)

        if not isinstance(self.nbsym, int) or isinstance(self.nbsym, bool):
            raise TypeError("the `nbsym` must be int.")

        if self.nbsym < 0:
            raise ValueError("the `nbsym` must >= 0.")

    def config_generate(self) -> ICEEMDANConfig:
        return ICEEMDANConfig(
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
        S, N = Check_Time_and_Signal(S, T=None, ndim={1}, method=self.name, dtype=self.dtype, need_T=False)

        if not (
                S.dtype.kind == "f" and S.dtype.itemsize >= 4):  # int/bool 输入会在 avg_env /= 2 处直接 UFuncTypeError, f16 会让样条报非有限值
            S = Check.to_dtype(S, np.dtype(np.float64))

        # 每次调用都从干净状态开始: 否则第二次 decompose 会把上一次的累积结果当成初值
        self.imfs = None
        self.res = None

        noise_std = self.noise_amp * S.std()  # 实际噪声标准差, 便于追溯

        # 退化输入 (常量/全零/线性趋势/单峰/过短): 信号本身极大或极小不足 2 个, 内层 EMD 一个 IMF
        # 也分不出来; 若照常加噪, 分出来的全是"我们加进去的噪声"。直接返回空 IMF 集, 与 EMD 对齐。
        _, _, peak_max = find_peaks(S, mod="both", safe="safe", backend=self.peaks_backend)
        if np.sum(peak_max) < 2 or np.sum(~peak_max) < 2:
            self.imfs = np.empty((0, N), dtype=S.dtype)
            self.res = S
            return DecompositionResult(
                IMFs=self.imfs,
                Res=self.res,
                info={"noise_std": noise_std,"emd info": [], "noise emd info": [], "rows": []},
                config=self.config_generate()
            )

        original_energy = np.dot(S, S)
        # 噪声池取 **单位方差**: 注入幅值由 β_k 决定 (见下), 若这里再乘 std(S) 会被 β_k 约掉
        noises: np.ndarray = np.random.normal(loc=0, scale=1.0, size=(self.M, N)).astype(S.dtype, copy=False)

        EMD_info: List[dict] = []
        noisy_EMD_info: List[dict] = []
        rows: List[int] = []

        if not self.parallel:
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

                # [M, min_rows, N] —— 第 n 阶要用噪声的第 n 阶 IMF, 所以最多只能做 min_rows 阶
                noise_imfs: np.ndarray = np.stack([row[:min_rows] for row in stacked]).astype(S.dtype, copy=False)
                del noises, stacked

                # 阶数在循环前就已知 (num_imf=-1 时就是噪声的阶数), 直接预分配
                max_imf: int = min_rows if self.num_imf == -1 else min(self.num_imf, min_rows)
                imfs_buf: np.ndarray = np.empty((max_imf, N), dtype=S.dtype)

                n_stage: int = 0
                while n_stage < max_imf:
                    # beta_n = noise_amp * std(r_{n-1}) / std(E_n(w)):
                    # 把注入幅值锚在**当前残差**的尺度上 (第 1 阶恰为 noise_amp*std(S))
                    beta: float = self.noise_amp * float(np.std(S)) / \
                        (float(np.std(noise_imfs[:, n_stage, :])) + 1e-12)

                    mid: np.ndarray = None
                    contrib: int = 0

                    for m in range(self.M):
                        S_noisy = S + beta * noise_imfs[m, n_stage, :]

                        im = self.local_mean_EMD.decompose(S_noisy).IMFs
                        if im.shape[0] < 1:
                            continue                    # 筛不出最高频, 这一实现这一阶跳过

                        lm = S_noisy - im[0]                     # M(x) = x - E_1(x)
                        contrib += 1
                        mid = lm if mid is None else mid + lm

                    if mid is None:
                        break

                    mid /= self.M                                # r_n = <M(r_{n-1} + beta_n E_n(w))>

                    imfs_buf[n_stage, :] = S - mid           # IMF_n = r_{n-1} - r_n
                    rows.append(contrib)
                    S = mid
                    n_stage += 1

                    if self.num_imf != -1 and n_stage >= self.num_imf:
                        break
                    if monotonic(S, mod="monotonic", safe="safe") or np.dot(S, S) < self.res_energy * original_energy:
                        break

                self.imfs = imfs_buf[:n_stage, :]
                self.res = S

            else:
                # 显式 num_imf 时预分配; num_imf=-1 时阶数未知, 只能用 list 暂存
                imfs: List[np.ndarray] = []
                imfs_buf: np.ndarray = np.empty((self.num_imf, N), dtype=S.dtype) if self.num_imf != -1 else None

                n_stage: int = 0
                while True:
                    # 先把本阶的噪声 IMF 块逐实现剥出来 (渐进剥噪声), 才能算 beta_n
                    block: List[np.ndarray] = []
                    for m in range(self.M):
                        noise_result = self.EMD.decompose(noises[m, :])
                        noisy_EMD_info.append(noise_result.info)

                        im_noise = noise_result.IMFs
                        noises[m, :] = noise_result.Res

                        if im_noise.shape[0] < 1:
                            continue                    # 噪声已剥不出 IMF, 这一实现这一阶跳过

                        block.append(im_noise[0, :])

                    if not block:
                        break

                    noise_block: np.ndarray = np.stack(block)
                    beta: float = self.noise_amp * float(np.std(S)) / \
                        (float(np.std(noise_block)) + 1e-12)

                    mid: np.ndarray = None
                    contrib: int = 0

                    for row in noise_block:
                        S_noisy = S + beta * row

                        im = self.local_mean_EMD.decompose(S_noisy).IMFs
                        if im.shape[0] < 1:
                            continue

                        lm = S_noisy - im[0]                     # M(x) = x - E_1(x)
                        contrib += 1
                        mid = lm if mid is None else mid + lm

                    if mid is None:
                        break

                    mid /= self.M

                    imf: np.ndarray = S - mid                   # IMF_n = r_{n-1} - r_n
                    if imfs_buf is not None:
                        imfs_buf[n_stage, :] = imf
                    else:
                        imfs.append(imf)
                    rows.append(contrib)
                    S = mid
                    n_stage += 1

                    # 显式 num_imf 到点就停 (-1 时永不触发)
                    if self.num_imf != -1 and n_stage >= self.num_imf:
                        break
                    if monotonic(S, mod="monotonic", safe="safe") or np.dot(S, S) < self.res_energy * original_energy:
                        break

                self.imfs = imfs_buf[:n_stage, :] if imfs_buf is not None else \
                    (np.asarray(imfs, dtype=S.dtype) if imfs else np.empty((0, N), dtype=S.dtype))
                self.res = S

        else:
            import multiprocessing as mp

            # ICEEMDAN 的相邻阶是**串行依赖**的 (第 n 阶要用第 n-1 阶的集平均局部均值当输入),
            # 所以能并行的只有"每一阶里对 M 个噪声实现求集平均"这一段。每阶 spawn 一批 worker,
            # 各算自己那一段的包络和, 父进程汇总后求平均 —— 与 EEMD/CEEMD 的 else 分支同形。
            #
            # spawn 出来的进程不带状态, 拿不到"上一阶留下的噪声残差", 因此噪声必须先整体预分解。
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

            # [M, min_rows, N] —— 第 n 阶要用噪声的第 n 阶 IMF, 所以最多只能做 min_rows 阶
            noise_imfs: np.ndarray = np.stack([row[:min_rows] for row in stacked]).astype(S.dtype, copy=False)
            del noises, stacked

            max_imf: int = min_rows if self.num_imf == -1 else min(self.num_imf, min_rows)
            imfs_buf: np.ndarray = np.empty((max_imf, N), dtype=S.dtype)

            # 试验按 M / num_workers 均分; 进程数不超过试验数 (否则进程会空转)
            num_workers: int = min(self.num_workers, self.M)
            base, extra = divmod(self.M, num_workers)
            tasks: List[dict] = []
            left = 0
            for p in range(num_workers):
                right = left + base + (1 if p < extra else 0)
                tasks.append({"index": p, "start": left, "end": right})
                left = right

            n_stage: int = 0

            from queue import Empty

            while n_stage < max_imf:
                # 第 n 阶用噪声的第 n 阶 IMF, 并按 beta_n 缩放到当前残差尺度后再下发给 worker
                noise_stage: np.ndarray = noise_imfs[:, n_stage, :]
                beta: float = self.noise_amp * float(np.std(S)) / (float(np.std(noise_stage)) + 1e-12)
                noise_block_all: np.ndarray = (beta * noise_stage).astype(S.dtype, copy=False)

                queue = mp.Queue()
                parallel_process: List[mp.Process | None] = [None for _ in range(num_workers)]
                for task in tasks:
                    parallel_process[task["index"]] = mp.Process(
                        target=self.parallel_decompose,
                        args=(S, noise_block_all[task["start"]:task["end"], :], queue, task))
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
                            # 有进程已退出。先 join 收尸: 子进程的 Queue.put 走后台 feeder 线程,
                            # 进程刚退出时结果可能还没 flush 进管道, 直接 get_nowait 会漏。
                            # ICEEMDAN 每阶只算两条包络, 毫秒级就结束, 这个窗口必然撞上。
                            for p in parallel_process:
                                if p is not None and not p.is_alive():
                                    p.join()

                            while True:                     # 把已到达的记录全部捞出来
                                try:
                                    item = queue.get_nowait()
                                except Exception:
                                    break
                                if isinstance(item, tuple) and item and item[0] == "error":
                                    worker_errors.append(item)
                                else:
                                    parallel_result_points_list.append(item)

                            codes = [p.exitcode for p in parallel_process if p is not None]
                            if any(c not in (0, None) for c in codes):
                                for p in parallel_process:
                                    if p is not None and p.is_alive():
                                        p.terminate()
                                for p in parallel_process:
                                    if p is not None:
                                        p.join()
                                raise RuntimeError(
                                    f"ICEEMDAN: worker 异常退出, exitcode={codes}; 已回传的错误={worker_errors or '无'}") from None

                            if len(parallel_result_points_list) + len(worker_errors) < num_workers \
                                    and all(p is not None and not p.is_alive() for p in parallel_process):
                                # 全部正常退出却仍收不齐 —— 不会再有新结果了, 直接报错免得空转
                                raise RuntimeError(
                                    f"ICEEMDAN: worker 全部退出但只收到 {len(parallel_result_points_list)}/{num_workers} 份结果")
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
                    raise RuntimeError(f"ICEEMDAN: worker 报错, exitcode={codes}; 错误={worker_errors or '无'}")

                parallel_result_points_list.sort(key=lambda it: it[0])

                mid: np.ndarray = None
                contrib: int = 0
                for item in parallel_result_points_list:
                    contrib += int(item[2])

                    acc = item[1]
                    if acc is None:
                        continue
                    mid = acc if mid is None else mid + acc

                if mid is None:
                    break

                mid /= self.M                                    # r_n = <M(r_{n-1} + beta_n E_n(w))>

                imfs_buf[n_stage, :] = S - mid                   # IMF_n = r_{n-1} - r_n
                rows.append(contrib)
                S = mid
                n_stage += 1

                if self.num_imf != -1 and n_stage >= self.num_imf:
                    break
                if monotonic(S, mod="monotonic", safe="safe") or np.dot(S, S) < self.res_energy * original_energy:
                    break

            self.imfs = imfs_buf[:n_stage, :]
            self.res = S

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

        ``noise_block`` 已由父进程按 ``beta_n`` 缩放好, 所以这里直接 ``S + noise_block[j]`` 就是
        ``r_{n-1} + beta_n E_n(w)``。每段算各自的局部均值 ``M(x) = x - E_1(x)`` 之和并回传,
        除以 ``M`` 求集平均由父进程统一做。相邻阶串行依赖, 所以父进程每阶 spawn 一批 worker,
        收齐本阶后再进入下一阶。

        与 EEMD/CEEMD 的 ``parallel_decompose`` 一样, 把 (序号, 局部均值和, 有效数) 通过
        ``pointer`` (mp.Queue) 交回父进程。
        """
        acc = None
        n_ok = 0

        # 任何异常都必须先回传一条错误记录再抛: 否则父进程只能靠"进程没了"来判断, 拿不到原因。
        try:
            for j in range(noise_block.shape[0]):
                S_noisy = S + noise_block[j, :]

                im = self.local_mean_EMD.decompose(S_noisy).IMFs
                if im.shape[0] < 1:
                    continue                    # 筛不出最高频, 这一实现这一阶跳过

                lm = S_noisy - im[0]            # M(x) = x - E_1(x)
                n_ok += 1
                acc = lm if acc is None else acc + lm

            pointer.put((task["index"], acc, n_ok))
        except BaseException as exc:
            pointer.put(("error", task["index"], f"{type(exc).__name__}: {exc}"))
            raise


def f_ICEEMDAN(S: List[int | float] | np.ndarray, *, num_imf: int = -1, noise_amp: float | int = 0.2, M: int = 100, res_energy: float = 1e-6, nbsym: int = 4, pre_decompose_noise: bool = False, parallel: bool = False, num_workers: int = 1, **kwargs) -> DecompositionResult:
    iceemdan = ICEEMDAN(
        num_imf=num_imf,
        noise_amp=noise_amp,
        M=M,
        res_energy=res_energy,
        nbsym=nbsym,
        pre_decompose_noise=pre_decompose_noise,
        parallel=parallel,
        num_workers=num_workers,
        **kwargs
    )

    return iceemdan.decompose(S)