"""
Ensemble Empirical Mode Decomposition —— 原生实现 (PyEMD-free)。

算法 (Wu & Huang 2009)
----------------------
1. 噪声幅度 ``scale = noise_width × |max(S) − min(S)|`` (与 PyEMD 同口径);
2. 做 ``trials`` 次独立试验: 每次在 ``S`` 上叠加白噪声, 用库内原生 ``EMD``
   引擎分解;
3. 同一阶的 ``trials`` 个结果逐点平均 —— 白噪声在平均中相互抵消, 留下比单次
   EMD 更稳健的分解。

自 0.3.0 起本模块即**原生实现** (原 ``EEMD_new``), 不再依赖 PyEMD;
``pyemd=True`` 保留一条到第三方 PyEMD 的过渡通道, 供逐点对拍使用。

残差语义 (相对旧 PyEMD 包装的修正)
----------------------------------
PyEMD 的 ``eemd()`` **不返回残差行**, 旧包装把**最后一行 IMF 当作残差**, 于是
少输出一阶 IMF 且 ``reconstruct()`` 不精确 (实测误差 0.27)。本实现按库内契约
取 ``Res = S − ΣIMFs``, 硬性保证 ``IMFs.sum(axis=0) + Res == S``。

逐轮即时截断
------------
各次试验的 IMF 阶数可能不同 (噪声改变极值计数)。本实现每收一轮就把已存结果截到
"当前最小阶数", 因此无需在最后再求和一遍, 峰值内存也随阶数下降而释放; 代价是
最低频那一阶被按最小公共阶数截掉。

并行策略
--------
``parallel=True`` 时用 ``multiprocessing.Pool`` 并行跑试验 —— EEMD 每次试验都在
**原始信号**上独立完成, 是 embarrassingly parallel。核数由 ``workers`` /
``cpu_ratio`` 二者之一决定, **两者互斥**:

* 都不给 → 默认 ``CPU_DEFAULT_RATIO`` × 可用核 (默认 **2/3**, 不占满核心);
* ``workers=-1`` → 用满可用核; ``workers=N`` → 恰好 N 个 (上限为可用核数);
* ``cpu_ratio=r`` → 可用核数 × ``r`` (向下取整, 至少 1)。

取值统一经 ``Utils.Memory.resolve_workers``, 与 ``Utils.FFT`` 的 ``workers``
共用同一策略。``parallel=False`` (默认) 时全部串行, 不产生任何子进程。

References
----------
10.1142/S1793536909000047
"""

import numpy as np
from dataclasses import dataclass
from typing import ClassVar, Literal

from .Base import Config, Decomposer, DecompositionResult
from ._Registry import register_class
from .EMD import EMD
from .Utils import Check_Time_and_Signal, resolve_seed, resolve_workers

__all__ = ["EEMD", "EEMDConfig"]

#: 包络后端 (与 ``EMD`` 的 canonical 取值一致; "linear" = np.interp 特例)。
_SPLINE_KINDS = ("CubicSpline", "PCHIP", "linear")

#: 内层 ``EMD`` 引擎接受的键 (供 ``**kwargs`` 校验; ``max_imf`` 由本类自行控制,
#: 不允许从 kwargs 传入以免与同名参数冲突)。
_ENGINE_KEYS = (
    "max_iter", "sd_thr", "find_peaks_mod", "faster",
)

#: 低于该总样本量 (trials × N) 时即使 ``parallel=True`` 也走串行 —— 子进程的
#: 启动/序列化开销会盖过收益 (单次 EMD 在几千点上只需毫秒级)。同一思路见
#: ``Utils.FFT.FFT_THREAD_MIN_ELEMS`` (小数组不起多线程)。
_PARALLEL_MIN_WORK = 1 << 16


@dataclass(frozen=True, kw_only=True)
class EEMDConfig(Config):
    """
    Effective parameters of an ``EEMD`` run.

    ``workers`` 记录**解析后**的实际进程数 (由 ``workers`` / ``cpu_ratio`` 或
    默认比例换算而来), 故一次运行用了多少核可直接从快照读出。
    ``backend`` 为 ``"native"`` 或 ``"pyemd"``。
    """
    trials: int
    noise_width: float
    max_imf: int
    seed: int | None
    parallel: bool
    workers: int
    spline_kind: str
    nbsym: int
    backend: str


def _trial_worker(payload):
    """
    单个试验的进程入口 (必须是模块级函数: ``multiprocessing`` 需要可 pickle,
    实例方法与闭包在 spawn 下无法序列化)。

    Parameters
    ----------
    payload : tuple
        ``(noisy_signal, spline_kind, nbsym, max_imf, engine_kwargs)``。

    Returns
    -------
    tuple
        ``(IMFs, iterations)`` —— ``IMFs`` 为 ``(k, N)`` 数组 (该轮阶数),
        ``iterations`` 为对应的筛分迭代次数列表。子进程内构造自己的 ``EMD``
        实例 (本包的可分解器对象不保证跨进程安全)。
    """
    noisy, spline_kind, nbsym, max_imf, engine_kwargs = payload

    result = EMD(
        spline_kind=spline_kind,
        nbsym=nbsym,
        max_imf=max_imf,
        **engine_kwargs,
    ).decompose(noisy)

    iterations = result.info.get("iterations", []) if result.info else []
    return result.IMFs, iterations


@register_class("EEMD")
class EEMD(Decomposer):
    """
    Ensemble Empirical Mode Decomposition (native, PyEMD-free).

    Averages the modes of ``trials`` independent native-EMD runs on white-noise
    perturbed copies of the signal. Use as ``Class.EEMD(**params).decompose(S,
    T)`` / ``Function.EEMD(S, T, **params)``; every result carries a frozen
    ``EEMDConfig`` snapshot.

    ``IMFs`` has shape ``(K, N)``; ``Res`` is the true remainder
    ``S − ΣIMFs`` (shape ``(N,)``), so ``reconstruct()`` reproduces the input
    exactly.

    ``pyemd=True`` restores the previous behaviour for the transition period:
    the call is delegated to the third-party PyEMD package (lazily imported
    through ``Base.Cache.cache.import_module``). PyEMD is being phased out —
    leave the flag at its default ``False`` to use the native engine.
    """

    name: ClassVar[str] = "EEMD"

    def __init__(
        self,
        trials: int = 100,
        noise_width: float = 0.05,
        max_imf: int = -1,
        seed: int | None = None,
        parallel: bool = False,
        spline_kind: Literal["CubicSpline", "PCHIP", "linear"] = "CubicSpline",
        nbsym: int = 2,
        rich_info: bool = False,
        workers: int | None = None,
        cpu_ratio: float | None = None,
        pyemd: bool = False,
        **kwargs,
    ):
        """
        Parameters
        ----------
        trials : int
            Number of ensemble realizations (>= 1).
        noise_width : float
            Amplitude of the added noise relative to the signal range:
            ``scale = noise_width * |max(S) - min(S)|`` (same convention as
            PyEMD). Must be >= 0.
        max_imf : int
            Maximum number of IMFs per trial; -1 decomposes completely.
        seed : int | None
            Local random seed. Overridden by a global seed when both are set.
        parallel : bool
            Run the trials through ``multiprocessing.Pool``. ``False``
            (default) is fully serial and spawns no subprocess.
        spline_kind : {"CubicSpline", "PCHIP", "linear"}
            Envelope backend of the internal native ``EMD``.
        nbsym : int
            Number of mirrored extrema at each boundary for the internal EMD.
        rich_info : bool
            When True, ``info`` additionally carries ``ensemble_std``
            (per-order pointwise standard deviation across trials, the
            ``ensemble_std`` semantics of PyEMD). Off by default because it
            costs one ``(trials, K, N)`` reduction.
        workers : int | None
            Process count when ``parallel`` is True. ``None`` = default policy
            (``CPU_DEFAULT_RATIO`` x available cores); ``-1`` = all cores;
            a positive int = exactly that many (capped at available cores).
            **Mutually exclusive with** ``cpu_ratio``.
        cpu_ratio : float | None
            Fraction of available cores to use (``(0, 1]``), floor, at least 1.
            **Mutually exclusive with** ``workers``.
        pyemd : bool
            ``False`` (default): use the native engine. ``True``: delegate to
            the third-party PyEMD package. Requires ``EMD-signal``; the option
            is transitional and will be removed when PyEMD support is dropped.
        **kwargs
            Extra parameters forwarded to the **internal native ``EMD``
            engine** (``max_iter`` / ``sd_thr`` / ``find_peaks_mod`` /
            ``faster``); 未识别的键会被拒绝, 不静默忽略。``pyemd=True`` 时这些
            键改传给 PyEMD 的 ``EMD``。默认取 ``EMD`` 自身的默认值。

        Raises
        ------
        ValueError
            On out-of-range ``trials`` / ``noise_width`` / ``max_imf`` /
            ``spline_kind`` / ``nbsym`` / ``rich_info``, on an unknown
            ``spline_kind``, when both ``workers`` and ``cpu_ratio`` are given,
            or on an unrecognized ``**kwargs`` key.
        """
        if isinstance(trials, bool) or not isinstance(trials, (int, np.integer)):
            raise ValueError(f"trials must be an int >= 1, got {trials!r}")
        if int(trials) < 1:
            raise ValueError(f"trials must be >= 1, got {trials!r}")

        if isinstance(noise_width, bool) or not isinstance(
            noise_width, (int, float, np.floating)
        ):
            raise ValueError(f"noise_width must be a number >= 0, got {noise_width!r}")
        if not np.isfinite(float(noise_width)) or float(noise_width) < 0.0:
            raise ValueError(f"noise_width must be finite and >= 0, got {noise_width!r}")

        if isinstance(max_imf, bool) or not isinstance(max_imf, (int, np.integer)):
            raise ValueError(f"max_imf must be -1 or >= 1, got {max_imf!r}")
        if int(max_imf) != -1 and int(max_imf) < 1:
            raise ValueError(f"max_imf must be -1 or >= 1, got {max_imf!r}")

        if not isinstance(parallel, bool):
            raise ValueError(f"parallel must be a bool, got {parallel!r}")
        if not isinstance(rich_info, bool):
            raise ValueError(f"rich_info must be a bool, got {rich_info!r}")
        if not isinstance(pyemd, bool):
            raise ValueError(f"pyemd must be a bool, got {pyemd!r}")

        if spline_kind not in _SPLINE_KINDS:
            raise ValueError(
                f"spline_kind must be one of {_SPLINE_KINDS}, got {spline_kind!r}"
            )
        if isinstance(nbsym, bool) or not isinstance(nbsym, (int, np.integer)):
            raise ValueError(f"nbsym must be an int >= 0, got {nbsym!r}")
        if int(nbsym) < 0:
            raise ValueError(f"nbsym must be >= 0, got {nbsym!r}")

        # **kwargs: 透传给内层引擎 —— 原生路径给库内 ``EMD``, ``pyemd=True`` 时给
        # PyEMD 的 ``EMD``。两条路径接受的键不同, 故这里只挡"两条都不认"的键;
        # 具体校验交给各自引擎 (EMD 的参数校验是完整的, 会在构造时立即报错)。
        unknown = sorted(set(kwargs) - set(_ENGINE_KEYS))
        if unknown:
            raise ValueError(
                f"unrecognized keyword argument(s) {unknown}; "
                f"the EMD engine accepts {list(_ENGINE_KEYS)} — "
                f"其余算法参数请用本类的同名形参显式传入"
            )

        self.trials = int(trials)
        self.noise_width = float(noise_width)
        self.max_imf = int(max_imf)
        self.seed = seed
        self.parallel = parallel
        self.spline_kind = spline_kind
        self.nbsym = int(nbsym)
        self.rich_info = rich_info
        self.pyemd = pyemd
        self.engine_kwargs = dict(kwargs)

        # workers / cpu_ratio 互斥, 在构造期解析并定死 (配置快照记录的即此值)。
        self.workers = resolve_workers(workers, cpu_ratio)

    def __call__(self, S, T=None) -> DecompositionResult:
        return self.decompose(S, T)

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    def _snapshot(self, effective_seed, backend: str) -> EEMDConfig:
        return EEMDConfig(
            trials=self.trials,
            noise_width=self.noise_width,
            max_imf=self.max_imf,
            seed=effective_seed,
            parallel=self.parallel,
            workers=self.workers,
            spline_kind=self.spline_kind,
            nbsym=self.nbsym,
            backend=backend,
        )

    def _use_pool(self, n_samples: int) -> bool:
        """
        Whether this run should actually go through ``multiprocessing.Pool``.

        ``parallel=True`` alone is not enough: a small workload (``trials`` x
        ``n_samples`` below ``_PARALLEL_MIN_WORK``) is faster serially, because
        the per-trial cost is milliseconds while spawning workers and pickling
        each noisy copy is not. The resolved ``workers`` ceiling is still
        snapshotted in the config regardless.
        """
        return (
            self.parallel
            and self.workers > 1
            and self.trials > 1
            and self.trials * int(n_samples) >= _PARALLEL_MIN_WORK
        )

    # ------------------------------------------------------------------ #
    # decompose
    # ------------------------------------------------------------------ #
    def decompose(self, S, T=None) -> DecompositionResult:
        """
        Decompose the signal into IMFs and a residual.

        Returns
        -------
        DecompositionResult
            ``IMFs`` (K, N); ``Res`` (N,) with ``S == IMFs.sum(axis=0) + Res``;
            ``info`` with ``iterations`` (per-order sifting counts across
            trials), ``n_trials`` (realizations that actually entered the
            average) and ``workers`` (processes actually used), plus
            ``ensemble_std`` when ``rich_info`` is True.
        """
        S, T, N = Check_Time_and_Signal(S, T, ndim={1}, method=self.name)

        S = np.asarray(S)
        effective_seed, _ = resolve_seed(self.seed, self.name)

        if self.pyemd:
            return self._decompose_pyemd(S, T, N, effective_seed)

        rng = np.random.default_rng(effective_seed)

        # 噪声幅度: 与 PyEMD 同口径 (相对信号峰峰值)。
        scale = self.noise_width * float(np.abs(np.max(S) - np.min(S)))

        noise = rng.normal(loc=0.0, scale=scale, size=(self.trials, S.size))
        noisy_signals = S[None, :] + noise

        payloads = [
            (noisy_signals[i], self.spline_kind, self.nbsym, self.max_imf,
             self.engine_kwargs)
            for i in range(self.trials)
        ]

        if self._use_pool(S.size):
            from multiprocessing import Pool

            chunksize = max(1, self.trials // (self.workers * 4))
            with Pool(processes=self.workers) as pool:
                raw_results = pool.map(_trial_worker, payloads, chunksize=chunksize)
            used_workers = self.workers
        else:
            raw_results = [_trial_worker(p) for p in payloads]
            used_workers = 1

        # 逐轮即时截断: 每收一轮就把已存堆叠截到当前最小阶数。
        # 轮数上限 = 各轮产出的最小阶数 (每阶都恰好有 trials 个真实样本,
        # 不做零填充); 末轮无需再截。
        all_modes = None          # (trials, k_min, N)
        all_iters = None          # (trials, k_min)
        n_modes = None

        for imfs, iters in raw_results:
            k = int(imfs.shape[0])
            if k == 0:
                continue

            iters_arr = np.zeros(k, dtype=np.int64)
            if iters:
                iters_arr[: min(k, len(iters))] = iters[:k]

            trial_modes = np.asarray(imfs, dtype=np.float64)
            trial_iters = iters_arr.reshape(1, k)

            if n_modes is None:
                n_modes = k
                all_modes = trial_modes.reshape(1, k, S.size)
                all_iters = trial_iters
                continue

            if k < n_modes:
                n_modes = k
                all_modes = all_modes[:, :k, :]
                all_iters = all_iters[:, :k]

            # 非连续轮次 (k > n_modes) 在此截断, 故 next 维度仍是 n_modes。
            all_modes = np.concatenate(
                (all_modes, trial_modes[:n_modes].reshape(1, n_modes, S.size)), axis=0
            )
            all_iters = np.concatenate((all_iters, trial_iters[:, :n_modes]), axis=0)

        valid_trials = 0 if all_modes is None else int(all_modes.shape[0])

        if n_modes is None or n_modes == 0:
            # 所有试验都没产出 IMF: 信号整体进残差, 精确重构仍成立。
            IMFs = np.empty((0, S.size), dtype=np.float64)
            Res = np.array(S, dtype=np.float64, copy=True)
            info = {
                "iterations": [],
                "n_trials": valid_trials,
                "workers": used_workers,
            }
        else:
            IMFs = np.mean(all_modes, axis=0).astype(np.float64, copy=False)

            info = {
                # 每阶 IMF 在 trials 次试验中的筛分迭代次数。
                "iterations": [all_iters[:, k].tolist() for k in range(n_modes)],
                # 实际参与平均的试验数 (若某轮零 IMF 会小于 trials)。
                "n_trials": valid_trials,
                # 实际使用的进程数 (小样本量下 parallel=True 也会退化为 1)。
                "workers": used_workers,
            }
            if self.rich_info:
                # PyEMD 的 ensemble_std 语义: 同阶各试验结果逐点标准差。
                info["ensemble_std"] = np.std(all_modes, axis=0)

            # 残差直接由总和反推 —— 逐轮已截断, 此处无须再求和一遍原始 trials。
            Res = (np.asarray(S, dtype=np.float64) - IMFs.sum(axis=0))

        return DecompositionResult(
            IMFs, Res, info, self._snapshot(effective_seed, "native")
        )

    # ------------------------------------------------------------------ #
    # 过渡分支: 交回第三方 PyEMD (旧的包装语义)
    # ------------------------------------------------------------------ #
    def _decompose_pyemd(self, S, T, N: int, effective_seed) -> DecompositionResult:
        """
        Delegate to the third-party PyEMD ``EEMD`` (``pyemd=True``).

        PyEMD is imported lazily through the process-wide import cache, so a
        missing ``EMD-signal`` only affects this branch.

        Note
        ----
        PyEMD 的 ``eemd()`` **不返回残差行**, 其真实残差在 ``decomposer.residue``。
        本分支按库内契约取 ``Res = S − ΣIMFs`` 而不是 ``result[-1]``, 因此
        即便走 PyEMD 也满足精确重构 —— 这是本库相对旧包装**有意保留的修正**。
        """
        from .Base.Cache import cache

        try:
            pyemd_module = cache.import_module(
                "PyEMD",
                description="PyEMD: 过渡期 EEMD/CEEMDAN 后端 (pyemd=True)",
            )
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise ImportError(
                "EEMD(pyemd=True) 需要第三方依赖 'EMD-signal' (导入名 'PyEMD'), "
                "但它未安装; 请先安装:\n"
                "    pip install EMD-signal\n"
                "或改用原生实现 (默认 pyemd=False, 不需要 PyEMD):\n"
                "    Class.EEMD(...).decompose(S)"
            ) from exc

        # PyEMD 的 EMD 引擎参数同样从 **kwargs 走 (它用自身的 spline_kind 拼写,
        # 故本类的 spline_kind 不转发, 只作配置快照记录)。
        ext_emd = pyemd_module.EMD(**self.engine_kwargs)
        decomposer = pyemd_module.EEMD(
            trials=self.trials,
            noise_width=self.noise_width,
            ext_EMD=ext_emd,
            parallel=self.parallel,
        )
        decomposer.noise_seed(effective_seed)

        stack = np.asarray(decomposer.eemd(S, T, max_imf=self.max_imf))

        # 契约修正: 不用 PyEMD 的最后一行当残差, 改由总和反推。
        IMFs = np.asarray(stack, dtype=np.float64)
        Res = np.asarray(S, dtype=np.float64) - IMFs.sum(axis=0)

        info = {
            "backend": "pyemd",
            "n_trials": self.trials,
            "workers": 1,
            # PyEMD 自己算好的残差 (旧包装丢弃的那个), 保留供对拍。
            "pyemd_residue": getattr(decomposer, "residue", None),
        }
        return DecompositionResult(
            IMFs, Res, info, self._snapshot(effective_seed, "pyemd")
        )
