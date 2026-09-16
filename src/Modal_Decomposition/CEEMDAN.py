"""
Complete Ensemble Empirical Mode Decomposition with Adaptive Noise
—— 原生实现 (PyEMD-free)。

CEEMDAN 的原生实现, 注册键 ``"CEEMDAN"`` (自 0.3.0 起, 原 ``CEEMDAN_new``)。
``pyemd=True`` 保留一条到第三方 PyEMD 的过渡通道 (``**kwargs`` 接收 PyEMD
专属参数), 供逐点对拍使用。

算法 (Torres 2011)
------------------
记 ``E_j(·)`` 为原生 ``EMD`` 引擎提取的第 ``j`` 阶 IMF, ``w_i`` 为第 ``i`` 条
白噪声实现, ``M = trials``, ``ε`` = 注噪幅度:

    c_1 = (1/M) Σ_i E_1( r_0 + ε·w_i )                         r_0 = S
    c_k = (1/M) Σ_i E_1( r_{k-1} + ε·E_k(w_i) )     (k ≥ 2)
    r_k = r_{k-1} − c_k

与 EEMD 的唯一区别在第 2 式: 每阶注入的是**噪声自身的第 k 阶 IMF** ``E_k(w_i)``,
而不是原始白噪声。EEMD 用同一份白噪声去打每一阶, 高频噪声会在低频阶上引入虚假
模态 (模态混叠); CEEMDAN 改成"本级频段的噪声", 于是每一阶的辅助扰动都落在该阶
将要提取的频段上, 模态混叠显著降低, 且所需 ``trials`` 远小于 EEMD。

``E_k(w_i)`` 由同一批噪声实现一次性**预分解**得到并跨阶复用 (见 ``info`` 的
``noise_imfs``); 各实现的可用阶数可能不同, 本实现按阶动态跳过阶数不足的实现
(见 ``info`` 的 ``trials_used``)。

与 PyEMD 版 (`pyemd=True`) 的有意差异
-------------------------------------
* **算法版本**: PyEMD 实现的是 Colominas 2014 改进版 (每阶只算**一组**局部均值并
  递归展开, 见 ``PyEMD/CEEMDAN.py`` 的 ``local_mean`` 分支), 而不是 Torres 2011
  原文。该改进版在本库已有原生实现 ``ICEEMDAN``, 故本实现走 Torres 2011 规范式,
  与 ``ICEEMDAN`` 形成真正的两种算法; 需要 PyEMD 语义时用 ``pyemd=True`` 分支
  逐位对拍。
* **注噪幅度可控**: PyEMD 的 ``noise_scale`` 在 ``beta_progress=True`` (其默认) 下
  被"按首阶 IMF 的 std 归一化"整项约掉 —— 实测 ``noise_scale`` 1.0 与 100.0 的
  输出差仅 4.4e-16, 真正的幅度旋钮是它未暴露的 ``epsilon=0.005``。本实现把幅度
  统一成单一参数 ``noise_width``, 逐点对拍时二者关系为 ``noise_width ≈ epsilon``。
* **原生引擎参数可控**: ``spline_kind`` / ``nbsym`` 显式暴露 (对齐 ``ICEEMDAN`` /
  ``RPSEMD`` / ``EEMD`` 的范式), 取代 PyEMD 专属的 ``ext_EMD`` /
  ``extrema_detection`` / ``noise_kind`` 口子; ``spline_kind`` 用 ``Utils.Spline``
  的 canonical 名, 不再是 PyEMD 的 ``"cubic"``/``"pchip"`` 拼写。
* **``range_thr`` / ``total_power_thr`` 在同一尺度上判定**: 递归整体在单位标准差的
  信号上进行 (与 PyEMD 相同做法), 故两个阈值对任意幅度的信号含义一致。
* **残差语义**: 按库内契约取 ``Res = S − ΣIMFs``, 硬性保证
  ``IMFs.sum(axis=0) + Res == S``。
* **``info`` 有诊断**: 每阶实际参与平均的 trial 数、各噪声实现的可用阶数、停机
  原因一并外露, 取代 PyEMD 分支的空 ``info``。

串行执行 (无并行空间)
--------------------
CEEMDAN 是**残差链迭代**, 整条链串行, 不存在可并行的维度:

* 各阶之间：``r_k = r_{k-1} − c_k`` 是链式依赖, 第 k 阶必须等第 k−1 阶算完;
* 同一阶内：全部 ``M`` 次试验共用**同一个** ``r_{k-1}``。它们看似互相独立, 但这
  一批只能在"第 k 阶的残差已经就位"之后才开始, 而下一批又要等这一批的均值
  ``c_k`` 定下来才算得出 ``r_k`` —— 批与批之间仍被链卡住, 并行只发生在批内部;
* 噪声池的预分解 ``E_k(w_i)`` 确实与主链无关, 但那是一次性 ``(M, N)`` 批量操作,
  摊到每阶后只剩查表。

因此在阶内强上进程池, 省下的只是单次 EMD (几千点上毫秒级) 的耗时, 而每阶都要
重开一次池并同步一轮, 开销直接盖过收益。本实现不做并行分支。

这与 ``EEMD`` 有本质区别: EEMD 的每次试验都在**原始信号**上独立完成
(embarrassingly parallel, ``parallel=True`` 有意义), CEEMDAN 没有这个性质。

References
----------
10.1109/ICASSP.2011.5947265
"""

import numpy as np
from dataclasses import dataclass
from typing import ClassVar, Literal

from .Base import Config, Decomposer, DecompositionResult
from ._Registry import register_class
from .EMD import EMD
from .Utils import Check_Time_and_Signal, is_monotonic, resolve_seed

__all__ = ["CEEMDAN", "CEEMDANConfig"]

#: 包络后端 (与 ``EMD`` 的 canonical 取值一致; "linear" = np.interp 特例)。
_SPLINE_KINDS = ("CubicSpline", "PCHIP", "linear")

#: 极值检测后端 (``Utils.Peaks.find_peaks`` 的 ``mod``）。
_PEAKS_MODS = ("scipy", "numpy", "numba")

#: ``pyemd=True`` 时透传给 ``PyEMD.CEEMDAN`` 的专属参数 (即旧包装的完整参数面;
#: ``trials`` / ``max_imf`` / ``seed`` 由本类同名形参映射, 不在此列)。
_PYEMD_KWARGS = (
    "noise_scale", "noise_kind", "extrema_detection", "parallel", "processes",
)


@dataclass(frozen=True, kw_only=True)
class CEEMDANConfig(Config):
    """
    Effective parameters of a ``CEEMDAN`` run.

    引擎参数 (``spline_kind`` / ``nbsym`` / ``max_iter`` / ``sd_thr`` /
    ``find_peaks_mod`` / ``faster``) 直接透传给内层 ``EMD``, 语义与其同名参数
    完全一致 (见 ``EMDConfig``)。
    """
    trials: int
    noise_width: float
    max_imf: int
    seed: int | None
    spline_kind: str
    nbsym: int
    max_iter: int
    sd_thr: float
    find_peaks_mod: str
    faster: bool
    range_thr: float
    total_power_thr: float


@register_class("CEEMDAN")
class CEEMDAN(Decomposer):
    """
    Complete Ensemble Empirical Mode Decomposition with Adaptive Noise
    (native, PyEMD-free).

    Stage ``k`` extracts its mode as the ensemble average of the first IMF of
    ``M`` perturbed copies of the residual, where the perturbation is the
    ``k``-th IMF of the noise realization itself (see the module docstring for
    the recursion). Use as ``Class.CEEMDAN(**params).decompose(S, T)`` /
    ``Function.CEEMDAN(S, T, **params)``; every result carries a frozen
    ``CEEMDANConfig`` snapshot.

    ``IMFs`` has shape ``(K, N)``; ``Res`` is the true remainder
    ``S − ΣIMFs`` (shape ``(N,)``), so ``reconstruct()`` reproduces the input
    exactly.
    """

    name: ClassVar[str] = "CEEMDAN"

    def __init__(
        self,
        trials: int = 100,
        noise_width: float = 0.005,
        max_imf: int = -1,
        seed: int | None = None,
        spline_kind: Literal["CubicSpline", "PCHIP", "linear"] = "CubicSpline",
        nbsym: int = 2,
        max_iter: int = 100,
        sd_thr: float = 0.01,
        find_peaks_mod: Literal["scipy", "numpy", "numba"] = "numpy",
        faster: bool = True,
        range_thr: float = 0.01,
        total_power_thr: float = 0.05,
        rich_info: bool = False,
        pyemd: bool = False,
        **kwargs,
    ):
        """
        Parameters
        ----------
        trials : int
            Number of ensemble realizations per stage (``M``, >= 1). CEEMDAN
            needs far fewer than EEMD (its noise is stage-matched), so 50-100
            is usually enough.
        noise_width : float
            Relative amplitude ``ε`` of the injected noise (>= 0). Stage ``k``
            perturbs the residual with ``ε · E_k(w_i)``. Defaults to the
            canonical Torres-2011 value ``0.005`` (the ``ε`` of the reference
            implementation); larger values trade residual purity for a
            stronger separation of close modes. ``0`` disables the noise
            assist entirely (the recursion degenerates to repeated
            ``max_imf=1`` sifting on the plain residual).
        max_imf : int
            Maximum number of IMFs; -1 decomposes completely.
        seed : int | None
            Local random seed. Overridden by a global seed when both are set.
        spline_kind : {"CubicSpline", "PCHIP", "linear"}
            Envelope backend of the internal native ``EMD``.
        nbsym : int
            Number of mirrored extrema at each boundary for the internal EMD.
        max_iter : int
            Sifting iterations per IMF of the internal EMD (>= 1). Same meaning
            as ``EMD(max_iter=...)``.
        sd_thr : float
            Cauchy SD stopping threshold ``Σm² / Σh_prev²`` of the internal EMD,
            in ``(0, 1]``. Same meaning as ``EMD(sd_thr=...)``.
        find_peaks_mod : {"scipy", "numpy", "numba"}
            Peak-detection backend of ``Utils.Peaks`` used by the internal EMD.
        faster : bool
            Stopping-policy branch of the internal EMD. Defaults to **True**
            (energy-Cauchy branch), which differs from a plain ``EMD`` run
            (``faster=False``, the quality branch). Rationale: CEEMDAN calls
            the engine ``trials`` times per stage purely to read *one* IMF off
            a noise-perturbed residual, and the ensemble average over
            ``trials`` realizations largely absorbs the per-realization row
            purity that the narrowband gate ``|zc − ext| ≤ 1`` would have
            bought. Measured on ``trials`` x median of 5 runs, ``faster=False``
            costs up to ~3x more (N=1024/trials=30: 1.28 s vs 0.38 s) and is
            never faster here. Set ``faster=False`` if you want the internal
            engine to match the plain ``EMD`` default exactly.
        range_thr : float
            Stop when the residual range ``max(r) − min(r)`` falls below this
            value (>= 0). The recursion runs on the unit-standard-deviation
            signal, so the threshold is independent of the signal's amplitude.
        total_power_thr : float
            Stop when the residual total power ``Σ|r|`` falls below this value
            (>= 0, same normalized scale as ``range_thr``).
        rich_info : bool
            When True, ``info`` additionally carries ``ensemble_std``
            (per-stage pointwise standard deviation across the trials that
            entered the average — the ``ensemble_std`` semantics of PyEMD).
            Off by default because it costs one extra ``(M, N)`` reduction per
            stage.
        pyemd : bool
            ``False`` (default): use the native engine. ``True``: delegate to
            the third-party PyEMD package (its Colominas-2014 variant).
            Requires ``EMD-signal``; the option is transitional and will be
            removed when PyEMD support is dropped.
        **kwargs
            PyEMD 专属参数, 仅在 ``pyemd=True`` 时透传给 ``PyEMD.CEEMDAN``;
            原生路径下出现这些键会报错 (它们对原生实现没有意义)。当前透传的是
            旧包装的全部参数面: ``noise_scale`` / ``noise_kind`` /
            ``extrema_detection`` / ``parallel`` / ``processes``。

            注: PyEMD 的 ``noise_scale`` 在其默认 ``beta_progress=True`` 下会被
            "按首阶 IMF 的 std 归一化"整项约掉 (实测 1.0 与 100.0 的输出差仅
            1 ulp), 真正的幅度旋钮是它未暴露的 ``epsilon`` (默认 ``0.005``)。
            故原生路径用 ``noise_width`` (默认 ``0.005`` ≈ ``epsilon``) 作为
            唯一的幅度参数。

        Raises
        ------
        ValueError
            On out-of-range ``trials`` / ``noise_width`` / ``max_imf`` /
            ``range_thr`` / ``total_power_thr`` / ``max_iter`` / ``sd_thr`` /
            ``spline_kind`` / ``nbsym`` / ``find_peaks_mod`` / ``faster`` /
            ``rich_info``, or on ``**kwargs`` keys while ``pyemd=False``.
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

        # --- 内层 EMD 引擎参数 (与 EMDConfig 同名同语义) -------------------- #
        if isinstance(max_iter, bool) or not isinstance(max_iter, (int, np.integer)):
            raise ValueError(f"max_iter must be an int >= 1, got {max_iter!r}")
        if int(max_iter) < 1:
            raise ValueError(f"max_iter must be >= 1, got {max_iter!r}")

        if isinstance(sd_thr, bool) or not isinstance(sd_thr, (int, float, np.floating)):
            raise ValueError(f"sd_thr must be a number in (0, 1], got {sd_thr!r}")
        if not np.isfinite(float(sd_thr)) or not (0.0 < float(sd_thr) <= 1.0):
            raise ValueError(f"sd_thr must be in (0, 1], got {sd_thr!r}")

        if find_peaks_mod not in _PEAKS_MODS:
            raise ValueError(
                f"find_peaks_mod must be one of {_PEAKS_MODS}, got {find_peaks_mod!r}"
            )
        if not isinstance(faster, bool):
            raise ValueError(f"faster must be a bool, got {faster!r}")

        for _name, _value in (
            ("range_thr", range_thr),
            ("total_power_thr", total_power_thr),
        ):
            if isinstance(_value, bool) or not isinstance(
                _value, (int, float, np.floating)
            ):
                raise ValueError(f"{_name} must be a number >= 0, got {_value!r}")
            if not np.isfinite(float(_value)) or float(_value) < 0.0:
                raise ValueError(f"{_name} must be finite and >= 0, got {_value!r}")

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

        # **kwargs 只服务 pyemd 分支 (PyEMD 专属参数)。原生路径下出现即报错 ——
        # 静默忽略正是这个库里已经被修过多次的坏味道。
        unknown = sorted(set(kwargs) - set(_PYEMD_KWARGS))
        if unknown:
            raise ValueError(
                f"unrecognized keyword argument(s) {unknown}; "
                f"pyemd=True accepts {list(_PYEMD_KWARGS)}"
            )
        if kwargs and not pyemd:
            raise ValueError(
                f"keyword argument(s) {sorted(kwargs)} are PyEMD-only and require "
                f"pyemd=True; the native engine takes its knobs as explicit "
                f"parameters (max_iter / sd_thr / find_peaks_mod / faster)"
            )

        self.trials = int(trials)
        self.noise_width = float(noise_width)
        self.max_imf = int(max_imf)
        self.seed = seed
        self.spline_kind = spline_kind
        self.nbsym = int(nbsym)
        self.max_iter = int(max_iter)
        self.sd_thr = float(sd_thr)
        self.find_peaks_mod = find_peaks_mod
        self.faster = faster
        self.range_thr = float(range_thr)
        self.total_power_thr = float(total_power_thr)
        self.rich_info = rich_info
        self.pyemd = pyemd
        self.pyemd_kwargs = dict(kwargs)

    def __call__(self, S, T=None) -> DecompositionResult:
        return self.decompose(S, T)

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    def _engine(self, max_imf: int) -> EMD:
        """
        Build the internal sifting engine with this run's effective EMD knobs.

        引擎参数全部由本类显式持有并透传 (``spline_kind`` / ``nbsym`` /
        ``max_iter`` / ``sd_thr`` / ``find_peaks_mod`` / ``faster``), 不依赖
        ``EMD`` 的默认值 —— 否则 CEEMDAN 的配置快照将无法解释内层行为。
        """
        return EMD(
            spline_kind=self.spline_kind,
            nbsym=self.nbsym,
            max_imf=max_imf,
            max_iter=self.max_iter,
            sd_thr=self.sd_thr,
            find_peaks_mod=self.find_peaks_mod,
            faster=self.faster,
        )

    def _snapshot(self, effective_seed) -> CEEMDANConfig:
        """Build the frozen config snapshot with the effective seed."""
        return CEEMDANConfig(
            trials=self.trials,
            noise_width=self.noise_width,
            max_imf=self.max_imf,
            seed=effective_seed,
            spline_kind=self.spline_kind,
            nbsym=self.nbsym,
            max_iter=self.max_iter,
            sd_thr=self.sd_thr,
            find_peaks_mod=self.find_peaks_mod,
            faster=self.faster,
            range_thr=self.range_thr,
            total_power_thr=self.total_power_thr,
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
            ``info`` with ``trials`` (realizations requested), ``trials_used``
            (per-stage count that actually entered the average — it has ``K+1``
            entries when the run stopped on a stage that produced no mode, and
            fewer than ``trials`` when some noise realization lacked that IMF
            order), ``noise_imfs`` (available IMF orders per noise realization),
            ``stop_reason`` and ``n_imfs``, plus ``ensemble_std`` (per-stage,
            stacked) when ``rich_info`` is True.
        """
        S, T, N = Check_Time_and_Signal(S, T, ndim={1}, method=self.name)

        if N < 4:
            raise ValueError(f"CEEMDAN: signal length {N} is too short (need >= 4)")

        effective_seed, _ = resolve_seed(self.seed, self.name)

        if self.pyemd:
            return self._decompose_pyemd(S, T, effective_seed)

        rng = np.random.default_rng(effective_seed)

        config = self._snapshot(effective_seed)

        # 递归在单位标准差的信号上进行 (与 PyEMD 同做法): ε 与两个阈值因此对任意
        # 幅度的信号含义一致; 求得的模态与残差最后统一乘回原尺度。
        S = np.asarray(S, dtype=np.float64)
        scale = float(np.std(S))
        if scale == 0.0:
            # 常量信号: 无 IMF, 全部进入残差 (精确重构仍成立)。
            return DecompositionResult(
                np.empty((0, N), dtype=np.float64),
                S.copy(),
                {
                    "trials": self.trials,
                    "trials_used": [],
                    "noise_imfs": [],
                    "stop_reason": "constant_signal",
                    "n_imfs": 0,
                },
                config,
            )

        work = S / scale

        # --- 噪声池: 一次生成 (M, N), 逐行预分解, 逐阶复用第 k 阶 IMF -------- #
        # 全程按行索引 ndarray, 不为 trial 包元组列表 —— 元组列表既多一份 Python
        # 对象开销, 也切断了大数组的连续内存访问。
        noise = rng.normal(loc=0.0, scale=1.0, size=(self.trials, N))
        # 单实例复用: EMD.decompose 不携带跨调用状态, 无需逐 trial 重建。
        noise_engine = self._engine(max_imf=-1)

        noise_imfs: list = []
        for w in noise:
            noise_imfs.append(noise_engine.decompose(w).IMFs)

        noise_orders = np.fromiter(
            (imfs.shape[0] for imfs in noise_imfs), dtype=np.int64, count=self.trials
        )
        max_order = int(noise_orders.max()) if noise_orders.size else 0

        # 逐阶挑出"阶数足够"的实现下标; E_k(w_i) 直接由 (下标, k) 索引取用,
        # 不再另建切片表。
        eligible = [np.flatnonzero(noise_orders > k) for k in range(max_order)]

        # 逐阶注入的信号是 ε·E_k(w_i): ε 是常数, 预先乘进噪声池, 主循环里就只剩
        # 一次加法 (M×K 次标量乘法不值得留在热路径上)。
        scaled_noise = [imfs * self.noise_width for imfs in noise_imfs]

        imfs_out: list = []
        trials_used: list = []
        ensemble_std: list = []
        residual = work.copy()
        stop_reason = "max_imf"
        first_engine = self._engine(max_imf=1)

        while self.max_imf == -1 or len(imfs_out) < self.max_imf:
            stage = len(imfs_out) + 1          # 1-based: 本阶提取第 stage 阶 IMF
            if stage > max_order:
                stop_reason = "noise_exhausted"
                break

            # 该阶注入"噪声自身的第 stage 阶 IMF", 而不是原始白噪声 —— 这正是
            # CEEMDAN 相对 EEMD 的关键改动 (stage-matched 辅助扰动)。
            idx = eligible[stage - 1]
            stacked = np.empty((idx.size, N), dtype=np.float64)
            perturbed = np.empty(N, dtype=np.float64)   # 复用: 每次只改内容
            used = 0
            for i in idx:
                # 原地写入, 省掉 M×K 次临时数组分配。
                np.add(residual, scaled_noise[i][stage - 1], out=perturbed)
                imfs = first_engine.decompose(perturbed).IMFs
                if imfs.shape[0] == 0:
                    continue
                stacked[used] = imfs[0]
                used += 1

            if used == 0:
                # 该阶全部扰动都取不到 IMF ⇒ 残差已经没有可提取的极值了
                # (同时覆盖"引擎在个别实现上失败", 那种情况用 valid_trials 记录)。
                stop_reason = "residual_no_extrema"
                break

            stacked = stacked[:used]
            mode = stacked.mean(axis=0)
            imfs_out.append(mode)
            trials_used.append(used)
            if self.rich_info:
                ensemble_std.append(stacked.std(axis=0))

            residual = residual - mode

            # --- 停机判据 (在归一化残差上判定) ----------------------------- #
            if is_monotonic(residual):
                stop_reason = "residual_monotonic"
                break
            if float(np.max(residual) - np.min(residual)) < self.range_thr:
                stop_reason = "range_thr"
                break
            if float(np.sum(np.abs(residual))) < self.total_power_thr:
                stop_reason = "total_power_thr"
                break
            # 残差是否还有极值: 用**一次** EMD 探测回答, 而不是让下一阶的 M 次
            # 试验去"发现"它 —— 后者在失败阶上要空耗 M 次完整 EMD。只要阶数
            # K < M 就是净赚 (集成法恒满足: trials 通常 50~100, K 是个位数)。
            # 实测 (trials=20, K=5): 有探测 125 次 EMD 调用
            # (噪声池 20 + 主循环 5×20 + 6 次探测), 无探测 138 次
            # (噪声池 20 + 失败阶再跑满 18 次) —— 且 K 越小优势越大。
            if first_engine.decompose(residual).IMFs.shape[0] == 0:
                stop_reason = "residual_no_extrema"
                break

        if imfs_out:
            IMFs = np.asarray(imfs_out, dtype=np.float64) * scale
        else:
            IMFs = np.empty((0, N), dtype=np.float64)

        # 残差由总和反推 —— 与 EEMD 同思路, 精确重构硬性成立。
        Res = S - IMFs.sum(axis=0)

        info = {
            "backend": "native",
            "trials": self.trials,
            # 每阶实际进入平均的 trial 数 (噪声实现可用阶数不足时会小于 trials)。
            "trials_used": trials_used,
            # 各噪声实现预分解出的可用 IMF 阶数。
            "noise_imfs": noise_orders.tolist(),
            "stop_reason": stop_reason,
            "n_imfs": len(imfs_out),
        }
        if self.rich_info and ensemble_std:
            # PyEMD 的 ensemble_std 语义: 同阶各试验结果逐点标准差 (按阶堆叠)。
            info["ensemble_std"] = np.asarray(ensemble_std, dtype=np.float64)

        return DecompositionResult(IMFs, Res, info, config)

    # ------------------------------------------------------------------ #
    # 过渡分支: 交回第三方 PyEMD (Colominas-2014 变体)
    # ------------------------------------------------------------------ #
    def _decompose_pyemd(self, S, T, effective_seed) -> DecompositionResult:
        """
        Delegate to the third-party PyEMD ``CEEMDAN`` (``pyemd=True``).

        PyEMD is imported lazily through the process-wide import cache, so a
        missing ``EMD-signal`` only affects this branch.

        ``**kwargs`` (``noise_scale`` / ``noise_kind`` / ``extrema_detection`` /
        ``parallel`` / ``processes``) 是 PyEMD 专属参数面, 在此原样透传;
        ``trials`` / ``max_imf`` / ``seed`` 由本类同名形参映射。
        PyEMD 的 ``ceemdan`` 返回 ``vstack(IMFs, residue)``, 即**最后一行本身
        就是真残差** (与它的 ``eemd`` 不同), 故直接沿用。
        """
        from .Base.Cache import cache

        try:
            pyemd_module = cache.import_module(
                "PyEMD",
                description="PyEMD: 过渡期 EEMD/CEEMDAN 后端 (pyemd=True)",
            )
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise ImportError(
                "CEEMDAN(pyemd=True) 需要第三方依赖 'EMD-signal' (导入名 'PyEMD'), "
                "但它未安装; 请先安装:\n"
                "    pip install EMD-signal\n"
                "或改用原生实现 (默认 pyemd=False, 不需要 PyEMD):\n"
                "    Class.CEEMDAN(...).decompose(S)"
            ) from exc

        kwargs = dict(self.pyemd_kwargs)
        # PyEMD.CEEMDAN 的 parallel 默认是 **True**, 会在每个模态阶上开一次
        # multiprocessing.Pool —— 本库不给它这个默认: 一是它与本库其余方法
        # (默认串行) 不一致, 二是 Pool 在受限环境 (含部分 Windows / 沙箱) 直接
        # PermissionError。需要并行时显式传 `pyemd=True, parallel=True`。
        kwargs.setdefault("parallel", False)
        if not kwargs.get("parallel", False):
            # parallel=False 时 PyEMD 的 processes 无意义且会告警, 一并清掉。
            kwargs.pop("processes", None)

        decomposer = pyemd_module.CEEMDAN(
            trials=self.trials,
            seed=effective_seed,
            spline_kind="cubic",          # PyEMD 用自身拼写; 原生路径用 canonical 名
            nbsym=self.nbsym,
            **kwargs,
        )

        stack = np.asarray(decomposer.ceemdan(S, T, self.max_imf))

        IMFs = np.asarray(stack[:-1, :], dtype=np.float64)
        Res = np.asarray(stack[-1, :], dtype=np.float64)

        info = {
            "backend": "pyemd",
            "trials": self.trials,
            "n_imfs": int(IMFs.shape[0]),
            "stop_reason": "pyemd",
        }
        return DecompositionResult(IMFs, Res, info, self._snapshot(effective_seed))
