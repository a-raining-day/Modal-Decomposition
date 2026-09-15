"""
Variational Mode Decomposition —— 原生实现 (自研 ADMM 内核)。

算法与实现
----------
按 Dragomiretskiy & Zosso (2014) 的频域 ADMM 交替求解 K 个带限模态, 沿用原始
MATLAB / vmdpy 的镜像延拓与单边谱约定; 中心频率与模态与 vmdpy 在相同初值/相同迭代
次数下逐点可比 (见 "与 vmdpy 的等价性"), 但**不依赖 vmdpy**。

存储引擎 (``engine``)
---------------------
- ``"auto"`` (默认): 由 ``Utils.Memory`` 的预算判定分流 —— 投影工作集在预算内用
  ``"ram"``, 否则用 ``"chunked"`` (必要时 ``out_of_core`` 落盘)。**大数组走这条路
  不会因一次性加载而崩**。
- ``"ram"``: 全内存单块扫描, 最快 (适合中小数组)。
- ``"chunked"``: 频轴分块扫描, 工作集 ≈ O(K·chunk); ``out_of_core=True`` 时
  镜像/频谱/模态谱/λ̂ 走临时 memmap, 常驻内存再降一档。
  分块不加速 (实测慢 15–25%), 其价值是**让超大输入可行**。

``engine="ram"`` 与 ``"chunked"`` 单块时的数值差异仅来自归约求和顺序 (≈1e-15)。
"""

import sys
from dataclasses import dataclass, replace
from typing import ClassVar, Literal

import numpy as np

from .Base import Config, Decomposer, DecompositionResult
from .Base.ConstDefine import (
    VMD_MIN_SAMPLES,
    VMD_UHAT_INFO_LIMIT,
    VMD_PEAK_INIT_LIMIT,
    VMD_CHUNK_WORK_BYTES,
)
from ._Registry import register_class
from .Utils import Check_Time_and_Signal, get_fft, get_mirror, get_peaks, resolve_seed
from .Utils.Chunk import chunked_fill, default_chunk_size, temp_memmap, drop_memmap
from .Utils.Memory import should_use_memmap

__all__ = ["VMD", "VMDConfig"]


@dataclass(frozen=True, kw_only=True)
class VMDConfig(Config):
    """
    Effective parameters of a VMD run.

    前 9 个字段与旧版包装器同名同义, 外加四个运行期开关:

    ``engine``
        频域扫描引擎 (``"ram"`` / ``"chunked"`` / ``"auto"``);
    ``chunk_size``
        频轴分块长度 (解析后的实际值);
    ``out_of_core``
        模态谱/频谱/镜像是否落盘 (解析后的实际值)。

    字段与 ``VMD`` (vmdpy 包装) 的对应: ``num_imf``↔``K``, ``n``↔``Niter``(500),
    ``epsilon``↔``tol``, ``init_mod``↔``init`` (uniform/random/zero ↔ 1/2/0),
    ``DC``/``alpha``/``tau`` 同名同义。
    """

    num_imf: int
    n: int
    fs: float
    alpha: float
    tau: float
    epsilon: float
    DC: bool
    init_mod: str
    seed: int | None
    engine: str
    chunk_size: int
    out_of_core: bool
    store_history: bool
    vmdpy: bool


@register_class("VMD")
class VMD(Decomposer):
    """
    Variational Mode Decomposition (native ADMM implementation).

    Decomposes a 1-D signal into ``K`` band-limited modes by solving the VMD
    variational problem in the frequency domain:

    1. mirror-extend the signal to ``T = 2N`` samples (boundary effect control);
    2. keep the positive-frequency half of the spectrum (``T/2 = N`` bins);
    3. alternate, per mode: Wiener-filter update of ``û_k`` around its center
       frequency ``ω_k``, then a power-weighted update of ``ω_k``; then the dual
       ascent step on ``λ̂``;
    4. stop when ``(1/T)·Σ_k‖Δû_k‖² ≤ epsilon`` or after ``n`` iterations;
    5. rebuild the real modes from the one-sided spectra and drop the mirror.

    Use as ``Class.VMD(**params).decompose(S, T)``; every result carries a
    frozen ``VMDConfig`` snapshot of the effective parameters. ``IMFs`` has
    shape (K,N); ``Res`` is the true remainder ``S − ΣIMFs`` (VMD keeps data
    fidelity as a *soft* constraint when ``tau=0``), so ``reconstruct()``
    reproduces the input exactly.

    Mode order
    ----------
    Row ``k`` of ``IMFs`` follows the ``k``-th initial center frequency (``ω``
    init is ascending), exactly like vmdpy/MATLAB: after free evolution the
    final ``ω`` may cross, so the rows are **not** guaranteed to be sorted by
    center frequency. ``info["omega"]`` is returned for callers that want to
    reorder (e.g. ``order = np.argsort(info["omega"])``).
    """

    name: ClassVar[str] = "VMD"

    def __init__(
        self,
        num_imf: int = 2,
        n: int = 500,
        fs: float = 1.0,
        alpha: float = 2000.0,
        tau: float = 0.0,
        epsilon: float = 1e-7,
        DC: bool = False,
        init_mod: Literal["uniform", "random", "zero", "peak"] = "uniform",
        seed: int | None = None,
        engine: Literal["ram", "chunked", "auto"] = "auto",
        chunk_size: int | None = None,
        out_of_core: bool | None = None,
        store_history: bool = False,
        vmdpy: bool = False,
        config: VMDConfig = None,
    ) -> None:
        """
        Parameters
        ----------
        num_imf : int
            Number of modes (``K``); must be ``>= 1`` and ``<= N//2``.
        n : int
            Maximum ADMM iterations (vmdpy's ``Niter``, default 500).
        fs : float
            Sampling frequency, used only to report ``omega_hz = omega * fs``;
            it does not change the decomposition (VMD is scale-invariant).
        alpha : float
            Bandwidth constraint / data-fidelity balancing parameter (> 0).
        tau : float
            Dual-ascent step (0 for noise slack, i.e. no Lagrangian update).
        epsilon : float
            Convergence tolerance of ``(1/T)·Σ_k‖Δû_k‖²``; 0 runs all ``n``
            iterations (used by the vmdpy parity check).
        DC : bool
            When true the first mode is kept at DC (``ω_0 = 0``). Must be a real
            ``bool`` (``1``/``0`` are rejected).
        init_mod : {"uniform", "random", "zero", "peak"}
            Center-frequency initialization; ``"uniform"``/``"random"``/``"zero"``
            match vmdpy ``init=1/2/0`` and ``"peak"`` starts from the strongest
            spectral peaks (``Utils.Peaks``, native-only).
        seed : int | None
            Local random seed (only affects ``init_mod="random"``). A non-None
            process-level seed set through ``Modal_Decomposition.set_seed``
            overrides it and emits a ``UserWarning``.
        engine : {"ram", "chunked", "auto"}
            Frequency-axis sweep engine. ``"auto"`` (default) defers to the global
            memory policy; ``"ram"`` keeps everything in memory (fastest, and by
            construction identical to a plain single-block sweep); ``"chunked"``
            sweeps the frequency axis in chunks (working set nearly independent
            of ``N``); ``"auto"`` picks ``"chunked"`` when the projected working
            set trips ``Utils.Memory.should_use_memmap``.
        chunk_size : int | None
            Frequency chunk length for ``engine="chunked"``. ``None`` uses
            ``Utils.Chunk.default_chunk_size`` (policy-aware). Values ``>= N``
            collapse to a single chunk.
        out_of_core : bool | None
            Store the mirror / spectrum / mode spectra in temporary ``np.memmap``
            files instead of RAM (only meaningful for ``engine="chunked"``).
            ``None`` decides per run through ``Utils.Memory.should_use_memmap``;
            ``True``/``False`` force it.
        store_history : bool
            Keep the per-iteration history: when True, ``info`` additionally
            carries ``u_hat_history`` (n_iter, K, N) and ``udiff_history``
            (n_iter,), costing ``n_iter·K·N`` complex elements.
            In the chunked engine the spectra are written **per chunk**, so enabling it
            does not add a full-spectrum copy pass. Default False.
        config : VMDConfig, optional
            Frozen parameter snapshot; when given it overrides every other
            argument. The returned snapshot is refreshed with the *effective*
            seed of each run (``self.config`` itself is never mutated).

        Raises
        ------
        ValueError
            On a non-integer/out-of-range ``num_imf`` or ``n``, a non-positive
            ``alpha``/``fs``, a negative ``tau``/``epsilon``, an invalid ``DC``,
            an unknown ``init_mod``/``engine``, or a ``chunk_size`` below 1.
        TypeError
            When ``config`` is given but is not a ``VMDConfig``.

        Notes
        -----
        没有并行开关: 见模块 docstring 的 "并行更新 (已评估, 未采用)" —— 模态间
        独立 (Jacobi) 更新在频带重叠时会发散, 分块方案也显著增加迭代数, 而多线程
        实测无收益。
        """
        self.config = config

        if isinstance(config, VMDConfig):
            self.num_imf = config.num_imf
            self.n = config.n
            self.fs = config.fs
            self.alpha = config.alpha
            self.tau = config.tau
            self.epsilon = config.epsilon
            self.DC = config.DC
            self.init_mod = config.init_mod
            self.seed = config.seed
            self.store_history = config.store_history
            self.engine = config.engine
            self.chunk_size = config.chunk_size
            self.out_of_core = config.out_of_core
            self.vmdpy = config.vmdpy
        elif config is None:
            self.num_imf = num_imf
            self.n = n
            self.fs = fs
            self.alpha = alpha
            self.tau = tau
            self.epsilon = epsilon
            self.DC = DC
            self.init_mod = init_mod
            self.seed = seed
            self.engine = engine
            self.chunk_size = chunk_size
            self.out_of_core = out_of_core
            self.store_history = store_history
            self.vmdpy = vmdpy
        else:
            raise TypeError(
                f"config must be a VMDConfig or None, got {type(config).__name__}"
            )

        # --- 参数校验 --------------------------------------------------- #
        if isinstance(self.num_imf, bool) or not isinstance(self.num_imf, (int, np.integer)):
            raise ValueError(f"num_imf must be an int >= 1, got {self.num_imf!r}")
        self.num_imf = int(self.num_imf)
        if self.num_imf < 1:
            raise ValueError(f"num_imf must be >= 1, got {self.num_imf!r}")

        if isinstance(self.n, bool) or not isinstance(self.n, (int, np.integer)):
            raise ValueError(f"n must be an int >= 1, got {self.n!r}")
        self.n = int(self.n)
        if self.n < 1:
            raise ValueError(f"n must be >= 1, got {self.n!r}")

        if isinstance(self.fs, bool) or not isinstance(self.fs, (int, float, np.floating)):
            raise ValueError(f"fs must be a number > 0, got {self.fs!r}")
        self.fs = float(self.fs)
        if not np.isfinite(self.fs) or self.fs <= 0.0:
            raise ValueError(f"fs must be positive and finite, got {self.fs!r}")

        if isinstance(self.alpha, bool) or not isinstance(self.alpha, (int, float, np.floating)):
            raise ValueError(f"alpha must be a number > 0, got {self.alpha!r}")
        self.alpha = float(self.alpha)
        if not np.isfinite(self.alpha) or self.alpha <= 0.0:
            raise ValueError(f"alpha must be positive and finite, got {self.alpha!r}")

        if isinstance(self.tau, bool) or not isinstance(self.tau, (int, float, np.floating)):
            raise ValueError(f"tau must be a number >= 0, got {self.tau!r}")
        self.tau = float(self.tau)
        if not np.isfinite(self.tau) or self.tau < 0.0:
            raise ValueError(f"tau must be non-negative and finite, got {self.tau!r}")

        if isinstance(self.epsilon, bool) or not isinstance(self.epsilon, (int, float, np.floating)):
            raise ValueError(f"epsilon must be a number >= 0, got {self.epsilon!r}")
        self.epsilon = float(self.epsilon)
        if not np.isfinite(self.epsilon) or self.epsilon < 0.0:
            raise ValueError(f"epsilon must be non-negative and finite, got {self.epsilon!r}")

        if not isinstance(self.DC, bool):
            raise ValueError(f"DC must be a bool, got {self.DC!r}")

        match self.init_mod.strip().lower() if isinstance(self.init_mod, str) else None:
            case "uniform" | "random" | "zero" | "peak" as name:
                self.init_mod = name
            case _:
                raise ValueError(
                    "init_mod must be one of ('uniform', 'random', 'zero', 'peak'), "
                    f"got {self.init_mod!r}"
                )

        match self.engine.strip().lower() if isinstance(self.engine, str) else None:
            case "ram" | "chunked" | "auto" as name:
                self.engine = name
            case _:
                raise ValueError(
                    "engine must be one of ('ram', 'chunked', 'auto'), "
                    f"got {self.engine!r}"
                )

        if self.chunk_size is not None:
            if isinstance(self.chunk_size, bool) or not isinstance(self.chunk_size, (int, np.integer)):
                raise ValueError(
                    f"chunk_size must be an int >= 1 or None, got {self.chunk_size!r}"
                )
            if int(self.chunk_size) < 1:
                raise ValueError(f"chunk_size must be >= 1, got {self.chunk_size!r}")
            self.chunk_size = int(self.chunk_size)

        if self.out_of_core is not None and not isinstance(self.out_of_core, bool):
            raise ValueError(
                f"out_of_core must be a bool or None, got {self.out_of_core!r}"
            )

        if not isinstance(self.store_history, bool):
            raise ValueError(f"store_history must be a bool, got {self.store_history!r}")

        if not isinstance(self.vmdpy, bool):
            raise ValueError(f"vmdpy must be a bool, got {self.vmdpy!r}")

        if not isinstance(config, VMDConfig):
            self.config = VMDConfig(
                num_imf=self.num_imf,
                n=self.n,
                fs=self.fs,
                alpha=self.alpha,
                tau=self.tau,
                epsilon=self.epsilon,
                DC=self.DC,
                init_mod=self.init_mod,
                seed=self.seed,
                engine=self.engine,
                chunk_size=self.chunk_size,
                out_of_core=self.out_of_core,
                store_history=self.store_history,
                vmdpy=self.vmdpy,
            )

        # 工具惰性取用: FFT 走 Utils.FFT 分发器 (后端由 Base.ConstDefine.FFT_BACKEND
        # 决定), 本模块不直接调 np.fft; vmdpy 分支的第三方库按需经 import cache 取。
        self.fft = get_fft()
        # 端点镜像统一走 Utils.Mirror (整条信号语义, 与极值镜像 mirror_extrema 区分)
        self.mirror = get_mirror().mirror_signal

    def decompose(self, S, T=None) -> DecompositionResult:
        """
        Decompose the signal into ``num_imf`` band-limited modes.

        Parameters
        ----------
        S : array-like
            1-D real signal (or a ``.npy`` path / long sequence — the input
            layer of ``Check_Time_and_Signal`` applies the global memory policy).
        T : array-like, optional
            Time axis; validated for length/duplicates/monotonicity but unused
            by the algorithm (VMD runs on the sample index).

        Returns
        -------
        DecompositionResult
            ``IMFs`` (K,N) modes in initialization order (see "Mode order" in
            the class docstring); ``Res`` = ``S − ΣIMFs``; ``info`` with

            ``omega`` (K,)
                final center frequencies, normalized cycles/sample (mirror axis);
            ``omega_hz`` (K,)
                the same frequencies in Hz (``omega * fs``);
            ``omega_history`` (n_iter, K)
                center frequency per completed iteration;
            ``u_hat`` (N, K)
                spectra of the returned modes (``fftshift(fft(mode))``, same
                convention as ``VMD``'s ``info["u_hat"]``);
            ``n_iter`` (int), ``converged`` (bool), ``init_mod`` (str),
            ``residual_ratio`` (float, ``‖Res‖/‖S‖``).
        """
        S, T, N = Check_Time_and_Signal(S, T, ndim={1}, method=self.name)

        if N < VMD_MIN_SAMPLES:
            raise ValueError(
                f"{self.name}: signal length must be >= {VMD_MIN_SAMPLES}, got {N}"
            )

        K = int(self.num_imf)
        if K > N // 2:
            raise ValueError(
                f"{self.name}: num_imf ({K}) must be <= N//2 ({N // 2}) so that "
                f"every mode keeps at least one spectral bin"
            )

        out_dtype = VMD._work_dtype(np.asarray(S).dtype)
        f = np.asarray(S, dtype=np.float64).reshape(-1)

        # --- vmdpy 分支: 交给可选的第三方实现 (不强制安装) ----------------- #
        if self.vmdpy:
            return self._decompose_vmdpy(S, f, N, K, out_dtype)

        effective_seed, _ = resolve_seed(self.seed, self.name)
        rng = np.random.default_rng(effective_seed)

        T_len = 2 * N

        # --- 引擎 / 分块粒度 / 落盘策略解析 (全部走 Utils 的既有策略) ------ #
        proj_ram = self._project_bytes(N, K, False, N)
        engine = self.engine
        if engine == "auto":
            engine = "chunked" if should_use_memmap(proj_ram["total"]) else "ram"

        if engine == "ram":
            chunk_size = N                      # 单块 == 不分块
            out_of_core = False
        else:
            if self.chunk_size is None:
                # 策略感知粒度: 目标"每块工作集 ≈ VMD_CHUNK_WORK_BYTES" (K 个模态各需
                # 旧值/新值/累加, 约 3 份复数), 再交给 Utils.Chunk 按内存策略收缩。
                # 注: 直接用 Chunk 的默认 8M 频点基数在 K 较大时会得到 GB 级工作集,
                # 故这里先按 K 折算基数。
                base = max(1, VMD_CHUNK_WORK_BYTES // (K * 16 * 3))
                chunk_size = int(default_chunk_size(N * 16, base=min(base, N)))
            else:
                chunk_size = int(self.chunk_size)
            chunk_size = max(1, min(chunk_size, N))
            out_of_core = (
                should_use_memmap(proj_ram["total"])
                if self.out_of_core is None
                else bool(self.out_of_core)
            )

        projection = self._project_bytes(N, K, out_of_core, chunk_size)

        # store_history=True: 逐次迭代的模态谱与收敛量 (代价 n·K·N 个复数)。
        u_hist = np.empty((self.n, K, N), dtype=np.complex128) if self.store_history else None
        udiff_hist = np.empty(self.n, dtype=np.float64) if self.store_history else None

        if engine == "ram":
            modes, omega, omega_history, n_iter, converged, init_used = self._engine_ram(
                f, N, K, rng, T_len, u_hist, udiff_hist
            )
        else:
            modes, omega, omega_history, n_iter, converged, init_used = self._engine_chunked(
                f, N, K, rng, T_len, chunk_size, out_of_core, u_hist, udiff_hist
            )

        IMFs = modes.astype(out_dtype, copy=False)
        Res = (f - modes.sum(axis=0)).astype(out_dtype, copy=False)

        norm_f = float(np.linalg.norm(f))
        residual_ratio = float(np.linalg.norm(Res)) / norm_f if norm_f > 0.0 else 0.0

        # u_hat 诊断谱是 K·N 复数: 大数组下不再附加计算 (见 VMD_UHAT_INFO_LIMIT)。
        if K * N * 16 <= VMD_UHAT_INFO_LIMIT:
            u_hat_info = self.fft.fftshift(self.fft.fft(modes, axis=1), axes=1).T
        else:
            u_hat_info = None

        info = {
            "omega": omega.copy(),
            "omega_hz": omega * self.fs,
            "omega_history": omega_history[:n_iter].copy(),
            "u_hat": u_hat_info,
            "fft_backend": self.fft.resolve_backend(None, int(2 * N * 8)),
            "n_iter": n_iter,
            "converged": converged,
            "init_mod": init_used,
            "residual_ratio": residual_ratio,
            "engine": engine,
            "chunk_size": chunk_size,
            "out_of_core": out_of_core,
            "projected_bytes": projection,
        }
        if u_hist is not None:
            info["u_hat_history"] = u_hist[:n_iter].copy()
            info["udiff_history"] = udiff_hist[:n_iter].copy()

        return DecompositionResult(
            IMFs,
            Res,
            info,
            replace(
                self.config,
                seed=effective_seed,
                engine=engine,
                chunk_size=chunk_size,
                out_of_core=out_of_core,
            ),
        )

    # ------------------------------------------------------------------ #
    # 可选分支: 用第三方 vmdpy 分解 (vmdpy=True)
    # ------------------------------------------------------------------ #
    def _decompose_vmdpy(self, S, f, N, K, out_dtype) -> DecompositionResult:
        """
        调用可选的第三方实现 ``vmdpy`` (经 import cache 惰性导入)。

        本分支用于**对照/复现**参考实现, 与原生分支的差异:
        - 迭代上限固定 500 (vmdpy 内部硬编码, ``n`` 不生效), 收敛判据为其原式;
        - 初值只支持 ``zero``/``uniform``/``random`` (``init_mod="peak"`` 仅原生支持);
        - **奇数长度不支持** (vmdpy 内部会丢末样本, 与本库返回契约冲突, 这里直接报错);
        - ``fs`` / ``seed`` / ``store_history`` / ``engine`` / ``chunk_size`` /
          ``out_of_core`` 在本分支不生效 (无迭代历史可存, 也不支持分块/外存);
        - 未安装 vmdpy 时抛 ``ImportError`` 并给出安装命令 —— 它是可选依赖, 不强制安装。
        """
        from .Base.Cache import cache

        if N % 2 != 0:
            raise ValueError(
                f"{self.name}(vmdpy=True): vmdpy 会丢弃末样本, 仅支持偶数长度; got N={N}"
            )
        match self.init_mod:                 # vmdpy 只认 0/1/2, 转换就地做
            case "zero":
                init_code = 0
            case "uniform":
                init_code = 1
            case "random":
                init_code = 2
            case _:
                raise ValueError(
                    f"{self.name}(vmdpy=True): init_mod={self.init_mod!r} 仅原生实现支持; "
                    "vmdpy 分支请用 ('zero', 'uniform', 'random')"
                )

        try:
            vmdpy = cache.import_module(
                "vmdpy",
                description="vmdpy: 可选第三方 VMD 实现 (VMD(vmdpy=True) 分支使用)",
            )
        except ImportError as exc:
            raise ImportError(
                "VMD(vmdpy=True) 需要可选的第三方实现 vmdpy; 请先安装:\n"
                "    %s -m pip install vmdpy" % sys.executable
            ) from exc

        # vmdpy.VMD(f, alpha, tau, K, DC, init, tol) -> (u, u_hat, omega_history)
        u, u_hat, omega_hist = vmdpy.VMD(
            f, self.alpha, self.tau, K, int(self.DC), init_code, self.epsilon
        )
        modes = np.asarray(u)
        omega_history = np.asarray(omega_hist)
        omega = omega_history[-1] if omega_history.ndim == 2 else omega_history
        n_iter = int(omega_history.shape[0]) if omega_history.ndim == 2 else 0

        IMFs = modes.astype(out_dtype, copy=False)
        Res = (f - modes.sum(axis=0)).astype(out_dtype, copy=False)
        norm_f = float(np.linalg.norm(f))
        info = {
            "impl": "vmdpy",
            "omega": np.asarray(omega).copy(),
            "omega_hz": np.asarray(omega) * self.fs,
            "omega_history": omega_history.copy(),
            "u_hat": np.asarray(u_hat),
            "fft_backend": None,            # vmdpy 内部自行调用 np.fft
            "n_iter": n_iter,
            "converged": None,              # vmdpy 不报告是否收敛
            "init_mod": self.init_mod,
            "residual_ratio": (float(np.linalg.norm(Res)) / norm_f) if norm_f > 0 else 0.0,
        }
        return DecompositionResult(
            IMFs, Res, info, replace(self.config, vmdpy=True)
        )

    # ------------------------------------------------------------------ #
    # 引擎 1: 全内存单块扫描 (与原单块实现逐位一致)
    # ------------------------------------------------------------------ #
    def _engine_ram(self, f, N, K, rng, T_len, u_hist=None, udiff_hist=None):
        """
        频域全内存扫描: 单块递推, 只保留"当前模态旧值"的 (N,) 暂存。

        ``u_hist`` / ``udiff_hist`` 非 None 时就地写入逐次迭代历史 (store_history)。

        Returns
        -------
        (modes, omega, omega_history, n_iter, converged, init_used)
            ``modes`` 为 float64 ``(K, N)`` 时域模态 (去镜像后)。
        """
        freqs = np.arange(N, dtype=np.float64) / T_len
        f_mirr = self.mirror(f)
        # fftshift(fft(f_mirr))[T//2:] 等价于 rfft 的前 N 个 bin (Nyquist 被丢弃)
        f_hat_plus = self.fft.rfft(f_mirr)[:N]
        del f_mirr

        init_used = self.init_mod
        omega = self._init_omega(self.init_mod, K, freqs, f_hat_plus, rng, N)
        if self.DC:
            omega[0] = 0.0

        # 模态谱按 (K, N) 行主序: 内循环全是连续扫描。
        # u_prev 整块 (K,N) 已去掉: 顺序更新只需当前模态的旧值 ⇒ 单个 (N,) 暂存。
        u_hat = np.zeros((K, N), dtype=np.complex128)
        lam_half = np.zeros(N, dtype=np.complex128)
        sum_all = np.zeros(N, dtype=np.complex128)
        old = np.empty(N, dtype=np.complex128)
        work = np.empty(N, dtype=np.complex128)
        resid = np.empty(N, dtype=np.complex128)
        denom = np.empty(N, dtype=np.float64)
        power = np.empty(N, dtype=np.float64)

        omega_history = np.empty((self.n, K), dtype=np.float64)
        n_iter = 0
        converged = False

        for it in range(self.n):
            udiff = 0.0

            for k in range(K):
                np.copyto(old, u_hat[k])            # 旧值 (替代整块 u_prev 的第 k 行)
                # sum_all = Σ_{i<k} û_i^{new} + Σ_{i>=k} û_i^{old},
                # 故 sum_all - û_k^{old} = Σ_{i≠k} (与 vmdpy 的 sum_uk 增量式等价)。
                np.subtract(sum_all, old, out=resid)
                np.subtract(f_hat_plus, resid, out=resid)
                resid -= lam_half

                # 维纳滤波器分母: 1 + α(f - ω_k)²
                np.subtract(freqs, omega[k], out=denom)
                np.square(denom, out=denom)
                denom *= self.alpha
                denom += 1.0

                np.divide(resid, denom, out=u_hat[k])

                # 中心频率: 正频半轴功率加权均值 (DC 模态锁定 0)。
                if not (self.DC and k == 0):
                    np.abs(u_hat[k], out=power)
                    np.square(power, out=power)
                    energy = float(power.sum())
                    if energy > 0.0:
                        omega[k] = float(np.dot(freqs, power)) / energy

                # 累加器 + 收敛量 ‖Δû_k‖²
                np.subtract(u_hat[k], old, out=work)
                sum_all += work
                udiff += float(np.vdot(work, work).real)

            # 对偶上升: 只保留 λ̂/2 一份 (λ̂/2 ← λ̂/2 + (τ(Σ_k û_k - f̂₊))/2)。
            # 注: 乘以 0.5 是 2 的幂次缩放, 与 IEEE 舍入可交换, 故与保留 λ̂ 的写法逐位相同。
            np.subtract(sum_all, f_hat_plus, out=work)
            work *= self.tau
            work *= 0.5
            lam_half += work

            omega_history[it] = omega
            if u_hist is not None:
                u_hist[it] = u_hat              # 本轮迭代后的模态谱
                udiff_hist[it] = udiff
            n_iter = it + 1

            if udiff / T_len <= self.epsilon + np.spacing(1.0):
                converged = True
                break

        half = np.zeros((K, N + 1), dtype=np.complex128)
        half[:, :N] = u_hat       # 单边谱不含 Nyquist 分量 (被单边化丢弃)
        full = self.fft.irfft(half, n=T_len, axis=1)      # (K, 2N)
        left = N // 2
        modes = np.ascontiguousarray(full[:, left:left + N])   # (K, N) float64
        return modes, omega, omega_history, n_iter, converged, init_used

    # ------------------------------------------------------------------ #
    # 引擎 2: 频轴分块扫描 (+ 可选外存)
    # ------------------------------------------------------------------ #
    def _engine_chunked(self, f, N, K, rng, T_len, chunk_size, out_of_core,
                        u_hist=None, udiff_hist=None):
        """
        频轴分块扫描: 工作集 ~ O(K·chunk) 而非 O(K·N); ``out_of_core=True`` 时
        镜像 / 单边谱 / 模态谱 / λ̂ 全部落在临时 memmap (磁盘)。

        与全内存路径的**唯一**数值差异是两个归约 (ω 的功率加权均值、收敛量
        ``Σ_k‖Δû_k‖²``) 的求和顺序 (逐块部分和再相加), 其余逐元素运算完全一致。

        ``u_hist`` / ``udiff_hist`` 非 None 时就地写入逐次迭代历史: 模态谱在分块循环
        内**逐块**写入, 因此开 store_history 不会多出一次全谱拷贝。

        Returns
        -------
        (modes, omega, omega_history, n_iter, converged, init_used)
            ``modes`` 为 float64 ``(K, N)``; ``out_of_core=True`` 时由外存模态谱
            逐模态逆变换得到 (每模态一份 (N+1,) 缓冲)。
        """
        # --- 1) 镜像延拓 (可落盘) ---------------------------------------- #
        if out_of_core:
            mirror = temp_memmap((T_len,), np.float64)
        else:
            mirror = np.empty(T_len, dtype=np.float64)
        self.mirror(f, out=mirror, chunk_size=chunk_size)

        # --- 2) 单边谱: 全数组变换, 无法分块; 这是外存的"内存地板" -------- #
        spec = self.fft.rfft(mirror)               # (N+1,) complex128, 常在内存
        if out_of_core:
            drop_memmap(mirror)
            spec_store = temp_memmap((N,), np.complex128)
            for a, b in self._spans(N, chunk_size):
                spec_store[a:b] = spec[a:b]
            f_hat_plus = spec_store
            del spec
        else:
            f_hat_plus = spec[:N]

        # --- 3) ω 初值 (peak 模式需要整谱幅度: 大数组下退化为 uniform) ------ #
        init_used = self.init_mod
        if self.init_mod == "peak" and N * 8 > VMD_PEAK_INIT_LIMIT:
            init_used = "uniform"
            omega = self._init_omega("uniform", K, None, None, rng, N)
        else:
            omega = self._init_omega(
                self.init_mod, K, np.arange(N, dtype=np.float64) / T_len,
                np.abs(f_hat_plus) if self.init_mod == "peak" else None, rng, N,
            )
        if self.DC:
            omega[0] = 0.0

        # --- 4) 状态区 (可落盘) ------------------------------------------ #
        if out_of_core:
            u_hat = temp_memmap((K, N), np.complex128)
            lam_half = temp_memmap((N,), np.complex128)
        else:
            u_hat = np.zeros((K, N), dtype=np.complex128)
            lam_half = np.zeros(N, dtype=np.complex128)

        omega_history = np.empty((self.n, K), dtype=np.float64)
        n_iter = 0
        converged = False

        # 每块的工作缓冲 (与 K 和 chunk 同阶, 与 N 无关)。
        denom = np.empty(chunk_size, dtype=np.float64)
        resid = np.empty(chunk_size, dtype=np.complex128)

        for it in range(self.n):
            wnum = np.zeros(K, dtype=np.float64)
            wden = np.zeros(K, dtype=np.float64)
            udiff = 0.0

            for a, b in self._spans(N, chunk_size):
                c = b - a
                # 读入本块 (memmap → RAM 拷贝; RAM 路径则是视图拷贝)
                old_k = np.array(u_hat[:, a:b])          # (K, c) 旧值
                acc = old_k.sum(axis=0)                  # Σ_i û_i^{old}
                F_c = np.asarray(f_hat_plus[a:b])
                lh_c = np.array(lam_half[a:b])
                fr_c = np.arange(a, b, dtype=np.float64) / T_len
                new_k = np.empty((K, c), dtype=np.complex128)

                for k in range(K):
                    np.subtract(acc, old_k[k], out=resid[:c])
                    np.subtract(F_c, resid[:c], out=resid[:c])
                    resid[:c] -= lh_c

                    np.subtract(fr_c, omega[k], out=denom[:c])
                    np.square(denom[:c], out=denom[:c])
                    denom[:c] *= self.alpha
                    denom[:c] += 1.0

                    np.divide(resid[:c], denom[:c], out=new_k[k])

                    if not (self.DC and k == 0):
                        p = new_k[k].real ** 2 + new_k[k].imag ** 2
                        wnum[k] += float(np.dot(fr_c, p))
                        wden[k] += float(p.sum())

                    np.subtract(new_k[k], old_k[k], out=resid[:c])
                    acc += resid[:c]
                    udiff += float(np.vdot(resid[:c], resid[:c]).real)

                u_hat[:, a:b] = new_k
                if u_hist is not None:
                    u_hist[it, :, a:b] = new_k       # 逐块写历史 (无额外整谱拷贝)

                # 对偶上升 (只保留 λ̂/2): λ̂/2 ← λ̂/2 + (τ(Σ_k û_k - f̂₊))/2
                np.subtract(acc, F_c, out=resid[:c])
                resid[:c] *= self.tau
                resid[:c] *= 0.5
                lh_c += resid[:c]
                lam_half[a:b] = lh_c

            # ω 用整谱部分和更新 (每次迭代每模态只用一次, 与全内存路径等价)
            for k in range(K):
                if self.DC and k == 0:
                    continue
                if wden[k] > 0.0:
                    omega[k] = wnum[k] / wden[k]

            omega_history[it] = omega
            if u_hist is not None:
                udiff_hist[it] = udiff
            n_iter = it + 1

            if udiff / T_len <= self.epsilon + np.spacing(1.0):
                converged = True
                break

        # --- 5) 逆变换: 逐模态 (每模态一份 (N+1,) 缓冲, 与 K 解耦) --------- #
        left = N // 2
        modes = np.empty((K, N), dtype=np.float64)
        half = np.empty(N + 1, dtype=np.complex128)
        for k in range(K):
            half[:N] = u_hat[k]
            half[N] = 0.0                    # 单边谱不含 Nyquist 分量
            full = self.fft.irfft(half, n=T_len)
            modes[k] = full[left:left + N]

        # 外存工作区用尽即释放 (关句柄 + 删文件), 不留给进程退出时的清理兜底。
        if out_of_core:
            drop_memmap(u_hat)
            drop_memmap(lam_half)
            drop_memmap(f_hat_plus)              # 即 spec_store

        return modes, omega, omega_history, n_iter, converged, init_used

    @staticmethod
    def _work_dtype(dtype) -> np.dtype:
        """float32/float64 保持原精度; 其余 (float16/int/bool) 提升为 float64。"""
        dtype = np.dtype(dtype)
        if dtype.kind == "f" and dtype.itemsize >= 4:
            return dtype
        return np.dtype(np.float64)

    @staticmethod
    def _project_bytes(N: int, K: int, out_of_core: bool, chunk_size: int) -> dict:
        """
        估算一次分解的主要内存项 (字节), 供 ``Utils.Memory`` 策略判定与实验报告引用。

        以 float64 / complex128 计 (``np.fft`` 的工作精度), 按引擎分别建模:

        * 内存内 (``out_of_core=False``): 镜像 + 单边谱 + 模态谱 + λ̂/2 + 累加器 + 频率轴
          + 逐模态暂存 + (K 个模态一起做的) 逆变换与其输出;
        * 外存 (``out_of_core=True``): 镜像/模态谱/λ̂ 落盘 ⇒ 常驻项只剩**单边谱**
          (``np.fft`` 无法分块, 这是"内存地板") + 结果本身 (K·N float64 的 IMFs 与 Res,
          返回契约要求它们在内存里) + 逐模态逆变换缓冲 + 单块工作集。

        返回各项与 ``total``; 这是量级估算 (未计 FFT 内部临时量), 用于决策与对照实测。
        """
        c16, f8 = 16, 8
        out_modes = K * N * f8                      # 返回的 IMFs (float64 时与工作数组同体)
        out_res = N * f8                            # 返回的 Res
        floor = (N + 1) * c16                       # 单边谱: rfft 输出, 无法分块

        if out_of_core:
            terms = {
                "input": 0,                         # 输入可保持 memmap
                "mirror": 0,                        # 落盘
                "spectrum": floor,
                "modes": 0,                         # 落盘
                "dual": 0,                          # 落盘
                "accum": 0,                         # 逐块
                "freqs": 0,                         # 逐块
                "scratch": 0,                       # 逐块
                "recon": floor + 2 * N * f8,        # 逐模态: (N+1) 复数 + 2N 实数
                "output": out_modes + out_res,
                "chunk_ws": K * chunk_size * c16 * 3 + chunk_size * (c16 + f8),
            }
        else:
            terms = {
                "input": N * f8,
                "mirror": 2 * N * f8,
                "spectrum": floor,
                "modes": K * N * c16,               # û 模态谱 (u_prev 整块已去掉)
                "dual": N * c16,                    # λ̂/2 (单份)
                "accum": N * c16,                   # Σ_i û_i
                "freqs": N * f8,
                "scratch": N * c16 * 3 + N * f8 * 2,   # 旧值/残差/增量 + denom/power
                "recon": K * floor + K * 2 * N * f8,   # K 个模态一起逆变换
                "output": out_modes + out_res,
                "chunk_ws": 0,
            }
        terms["total"] = int(sum(terms.values()))
        return terms

    @staticmethod
    def _spans(n: int, chunk_size: int):
        """把 ``[0, n)`` 切成 ``(start, stop)`` 半开区间 (分块粒度由调用方给定)。"""
        for start in range(0, n, chunk_size):
            yield start, min(start + chunk_size, n)

    @staticmethod
    def _init_omega(
        init_mod: str,
        K: int,
        freqs: np.ndarray,
        f_hat_plus: np.ndarray,
        rng: np.random.Generator,
        N: int,
    ) -> np.ndarray:
        """
        中心频率初值 ``ω`` (归一化循环频率, 升序)。

        Parameters
        ----------
        init_mod : {"uniform", "random", "zero", "peak"}
            ``"zero"``    — ``ω ≡ 0`` (vmdpy ``init=0``);
            ``"uniform"`` — ``ω_i = 0.5·i/K`` (vmdpy ``init=1``, 与 vmdpy/MATLAB 逐位一致;
                            注意该网格覆盖 ``[0, 0.5)``, 而正频半轴只到 ``0.25``);
            ``"random"``  — ``[1/N, 0.5]`` 对数均匀取 K 点后升序 (vmdpy ``init=2``);
            ``"peak"``    — 单边幅度谱最强 K 个峰 (``Utils.Peaks``, numpy 后端), 峰不足
                            时用谱内等间隔点补齐 —— 自研扩展。
        K : int
            模态数。
        freqs : np.ndarray
            正频半轴归一化频率 (``j/T``), 长度 N。
        f_hat_plus : np.ndarray
            镜像信号的单边谱 (长度 N), ``"peak"`` 模式据此找峰。
        rng : np.random.Generator
            局部随机数发生器 (``"random"`` 模式使用)。
        N : int
            原始信号长度 (随机初值下界 ``1/N`` 即原始记录的分辨率)。

        Returns
        -------
        np.ndarray
            长度 K、升序的初值频率 (可能含重复值, 例如 ``"zero"``)。
        """
        if init_mod == "zero":
            return np.zeros(K, dtype=np.float64)

        if init_mod == "uniform":
            return (0.5 / K) * np.arange(K, dtype=np.float64)

        if init_mod == "random":
            lo, hi = 1.0 / N, 0.5
            w = np.exp(np.log(lo) + (np.log(hi) - np.log(lo)) * rng.random(K))
            return np.sort(w)

        # "peak": Utils.Peaks 惰性取用 (每次 decompose 解析一次, 迭代内零开销)。
        find_peaks = get_peaks().find_peaks
        idx, props = find_peaks(np.abs(f_hat_plus), mod="numpy")

        if idx.size == 0:
            return (0.5 / K) * np.arange(K, dtype=np.float64)

        heights = np.asarray(props.get("peak_heights", np.abs(f_hat_plus)[idx]))
        strongest = np.argsort(heights, kind="stable")[::-1][:K]
        picked = np.sort(idx[strongest]).astype(np.float64) / (2.0 * N)

        if picked.size < K:
            # 峰不足 K 个: 其余初值均匀铺在正频半轴内, 再整体升序。
            tail = np.linspace(freqs[0], freqs[-1], K - picked.size + 2)[1:-1]
            return np.sort(np.concatenate([picked, tail]))

        return picked
