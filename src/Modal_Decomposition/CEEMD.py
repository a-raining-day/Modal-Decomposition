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
class CEEMDConfig(Config):
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
class CEEMD(Decomposer):
    name = "CEEMD"

    def __init__(self, num_imf: int = -1, noise_amp: float = None, M: int = 25, parallel: bool = False, num_workers: int | float = 1, **kwargs) -> None:
        self.num_imf = num_imf
        self.noise_amp: float = noise_amp
        self.M: int = M
        self.parallel: bool = parallel
        self.num_workers: int | float = num_workers

        unknown = set(kwargs) - {
            "dtype",
            "emd_num_imf", "emd_nbsym", "emd_internal_delta", "emd_internal_iter",
            "emd_res_energy", "emd_faster", "emd_store_iter_info", "emd_pre_num_imfs",
            "emd_dtype", "emd_safe", "emd_need_squeeze", "emd_mirror_mod",
            "emd_poly_order", "emd_time_poly_order", "emd_peaks_backend",
        }
        if unknown:
            raise TypeError(f"unexpected keyword arguments: {unknown}")

        self.dtype: type | np.dtype | None = kwargs.get("dtype", None)

        self.emd_num_imf: int = kwargs.get("emd_num_imf", -1)
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

        if not isinstance(self.emd_num_imf, int) or isinstance(self.emd_num_imf, bool):
            raise TypeError("the `emd_num_imf` must be an int`")

        if self.emd_num_imf != -1:
            if not (self.emd_num_imf > 0):
                raise ValueError("the `emd_num_imf` must > 0")

        if self.num_imf == -1:
            self.emd_num_imf = -1
        else:
            if self.emd_num_imf != -1:
                self.emd_num_imf = self.num_imf if self.emd_num_imf < self.num_imf else self.emd_num_imf

    def config_generate(self) -> CEEMDConfig:
        return CEEMDConfig(
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

        if not (S.dtype.kind == "f" and S.dtype.itemsize >= 4):  # int/bool 输入会在 avg_env /= 2 处直接 UFuncTypeError, f16 会让样条报非有限值
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

        rows = []
        EMD_info = []

        if not self.parallel:
            for m in range(self.M):
                w = np.random.normal(loc=0, scale=noise_amp, size=S.size).astype(S.dtype, copy=False)  # 只生成一次
                S_plus_noisy = S + w
                S_minus_noisy = S - w

                plus_EMDResult = self.EMD.decompose(S_plus_noisy)
                minus_EMDResult = self.EMD.decompose(S_minus_noisy)

                IMFs_plus = plus_EMDResult.IMFs
                IMFs_minus = minus_EMDResult.IMFs

                if IMFs_plus.shape[0] == IMFs_minus.shape[0]:
                    IMFs = np.add(IMFs_plus, IMFs_minus)
                    IMFs /= 2
                else:
                    min_row_ensemble = min(IMFs_plus.shape[0], IMFs_minus.shape[0])
                    IMFs = np.add(IMFs_plus[:min_row_ensemble, :], IMFs_minus[:min_row_ensemble, :])
                    IMFs /= 2

                EMD_info.append((plus_EMDResult.info, minus_EMDResult.info))
                rows.append((IMFs_plus.shape[0], IMFs_minus.shape[0], IMFs.shape[0]))

                if self.imfs is None:
                    self.imfs = IMFs  # 没必要复制, 分解结果后续又不需要而且每轮还要被覆盖
                else:
                    min_row = min(self.imfs.shape[0], IMFs.shape[0])
                    self.imfs = self.imfs[:min_row]
                    self.imfs += IMFs[:min_row]

            self.imfs /= self.M
            self.res = S - self.imfs.sum(axis=0)

        else:
            import multiprocessing as mp

            # 试验按 M / num_workers 均分; 进程数不超过试验数 (否则进程会空转)
            num_workers = min(self.num_workers, self.M)
            trial_seeds = np.random.randint(0, 2 ** 31 - 1, size=self.M)  # 种子由父进程派生, 并行下也可复现
            base, extra = divmod(self.M, num_workers)
            tasks = []
            left = 0
            for p in range(num_workers):
                right = left + base + (1 if p < extra else 0)
                tasks.append({"index": p, "start": left, "end": right, "seeds": trial_seeds[left:right]})
                left = right

            # 回传通道: 每个进程 put 一条 (序号, 累积和, 试验数, 各行数, EMD info); 出错时 put 一条 ("error", 序号, 描述)
            queue = mp.Queue()
            parallel_process: List[mp.Process | None] = [None for _ in range(num_workers)]

            for task in tasks:
                parallel_process[task["index"]] = mp.Process(
                    target=self.parallel_decompose, args=(S, noise_amp, queue, task))
                parallel_process[task["index"]].start()

            # 用"轮询 + 存活检查": 超时只是轮询间隔, 不对健康进程设截止时间, 所以耗时长的正常试验不会被误杀; 一旦发现 worker 已退出, 立刻清场并抛错。
            from queue import Empty

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
                            f"CEEMD: worker 异常退出, exitcode={codes}; 已回传的错误={worker_errors or '无'}") from None
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
                raise RuntimeError(f"CEEMD: worker 报错, exitcode={codes}; 错误={worker_errors or '无'}")

            parallel_result_points_list.sort(key=lambda item: item[0])
            rows = [row for item in parallel_result_points_list for row in item[3]]
            min_row = min(item[1].shape[0] for item in parallel_result_points_list)

            self.imfs = np.array(parallel_result_points_list[0][1][:min_row], copy=True)
            EMD_info.extend(parallel_result_points_list[0][4])
            for item in parallel_result_points_list[1:]:
                self.imfs += item[1][:min_row]
                EMD_info.extend(item[4])

            self.imfs /= self.M  # 回传的是各进程累积和, 这里统一求全集成的算术平均
            self.res = S - self.imfs.sum(axis=0)

        config = self.config_generate()
        info = {"rows": rows, "min row": min(r[2] for r in rows),
                "noise_amp": noise_amp, "EMD info": EMD_info}

        return DecompositionResult(
            IMFs=self.imfs,
            Res=self.res,
            info=info,
            config=config
        )

    def parallel_decompose(self, S, noise_amp, pointer, task):
        """
        单个工作进程: 只负责自己那段试验 (task["start"] .. task["end"])。

        CEEMD 每次试验要分解一对互补信号 ``S + w`` 与 ``S - w`` (共用同一个 ``w``,
        这是互补性成立的前提), 再把两者逐阶平均; 因此本进程做 2*(end-start) 次内层 EMD。

        与串行分支一样先累积求和, 再把 (序号, 累积和, 试验数, 各行数, EMD info) 通过
        ``pointer`` (mp.Queue) 交回父进程; 平均由父进程统一做。
        """
        acc = None
        rows = []
        infos = []

        # 任何异常都必须先回传一条错误记录再抛: 否则父进程只能靠"进程没了"来判断, 拿不到原因。回传之后 re-raise, 让子进程以非零 exitcode 结束, 父进程两条路都能捕获。
        try:
            for m in range(task["start"], task["end"]):
                # spawn 启动的子进程 RNG 与父进程无关, 用父进程派生的种子保证可复现
                np.random.seed(int(task["seeds"][m - task["start"]]))

                w = np.random.normal(loc=0, scale=noise_amp, size=S.size).astype(S.dtype, copy=False)  # 只生成一次, 正负共用
                plus_EMDResult = self.EMD.decompose(S + w)
                minus_EMDResult = self.EMD.decompose(S - w)

                IMFs_plus: np.ndarray = plus_EMDResult.IMFs
                IMFs_minus: np.ndarray = minus_EMDResult.IMFs

                min_row_ensemble = min(IMFs_plus.shape[0], IMFs_minus.shape[0])
                IMFs = (IMFs_plus[:min_row_ensemble, :] + IMFs_minus[:min_row_ensemble, :]) / 2

                infos.append((plus_EMDResult.info, minus_EMDResult.info))
                rows.append((IMFs_plus.shape[0], IMFs_minus.shape[0], IMFs.shape[0]))

                if acc is None:
                    acc = np.array(IMFs, copy=True)
                else:
                    min_row = min(acc.shape[0], IMFs.shape[0])
                    acc = acc[:min_row] + IMFs[:min_row]

            pointer.put((task["index"], acc, len(rows), rows, infos))
        except BaseException as exc:
            pointer.put(("error", task["index"], f"{type(exc).__name__}: {exc}"))
            raise


def f_CEEMD(S: List[int | float] | np.ndarray, *, num_imf: int, noise_amp: float = None, M: int = 25, parallel: bool = False, num_workers: int | float = 1, **kwargs) -> DecompositionResult:
    ceemd = CEEMD(
        num_imf=num_imf,
        noise_amp=noise_amp,
        M=M,
        parallel=parallel,
        num_workers=num_workers,
        **kwargs
    )

    return ceemd.decompose(S)