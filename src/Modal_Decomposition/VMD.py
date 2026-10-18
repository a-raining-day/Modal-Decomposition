"""
变分模态分解 (Variational Mode Decomposition) —— 原生实现 (自研 ADMM 内核)。

算法与实现
----------
按 Dragomiretskiy & Zosso (2014) 的频域 ADMM 交替求解 K 个带限模态, 沿用原始
MATLAB / vmdpy 的镜像延拓与单边谱约定: 相同初值、相同迭代次数下中心频率与模态与
vmdpy 逐点可比, 但不依赖 vmdpy。

存储引擎 (``engine``)
---------------------
- ``"auto"`` (默认): 由 ``Base.Memory`` 的预算判定分流 —— 投影工作集在预算内用
  ``"ram"``, 否则用 ``"chunked"`` (必要时 ``out_of_core`` 落盘)。大数组走这条路
  不会因为一次性加载而失败。
- ``"ram"``: 全内存单块扫描, 最快 (适合中小数组)。
- ``"chunked"``: 频轴分块扫描, 工作集 ≈ O(K·chunk); ``out_of_core=True`` 时
  镜像/频谱/模态谱/λ̂ 走临时 memmap, 常驻内存再降一档。
  分块不加速 (实测慢 15–25%), 其价值是让超大输入可行。

``engine="ram"`` 与 ``"chunked"`` 单块时的数值差异仅来自归约求和顺序 (≈1e-15)。

字符串形参的取值约定
--------------------
``init_mod`` / ``engine`` 按库内约定同时接受字符串与枚举成员 (``VMD_INIT_MODS`` /
``VMD_ENGINE_MODS``): 构造时由 ``Check.validate_mod`` 统一转成 Enum, 内部只按 Enum
分支; ``config_generate()`` 产出的 ``VMDConfig`` 与 ``info`` 里记录的仍是字符串
``.value``, 因此 ``config.to_dict()`` 的结果可直接 ``json.dumps``。
"""

import sys
from dataclasses import dataclass
from enum import Enum
from typing import ClassVar, Literal

import numpy as np

from .Define import (
    Config,
    Decomposer,
    DecompositionResult,
    DEFAULT_NUMPY_TYPE,
    VMD_CHUNK_WORK_BYTES,
    VMD_MIN_SAMPLES,
    VMD_PEAK_INIT_LIMIT,
    VMD_UHAT_INFO_LIMIT,
    check_output_memmap,
    finalize_output,
)
from .Base import (
    Cache,
    Chunk,
    Seed,
    drop_memmap,
    should_use_memmap,
    temp_memmap,
)
from .Utils import Check, Check_Time_and_Signal, find_peaks

__all__ = [
    "VMD",
    "VMDConfig",
    "VMD_INIT_MODS",
    "VMD_ENGINE_MODS",
    "VMD_INIT_MODS_ARGS",
    "VMD_ENGINE_MODS_ARGS",
    "f_VMD",
]

# 两个字符串形参的合法取值: Literal 别名负责静态检查 (模块全局, 供注解复用),
# Enum 负责运行期分支 —— 与 ``Utils._Hilbert`` 的 HILBERT_MODS(_ARGS) 同一约定。
VMD_INIT_MODS_ARGS = Literal["zero", "uniform", "random", "peak"]
VMD_ENGINE_MODS_ARGS = Literal["auto", "ram", "chunked"]

class VMD_INIT_MODS(Enum):
    Zero = "zero"
    Uniform = "uniform"
    Random = "random"
    Peak = "peak"

class VMD_ENGINE_MODS(Enum):
    Auto = "auto"
    RAM = "ram"
    Chunked = "chunked"


"""
Config
"""
@dataclass(frozen=True, kw_only=True)
class VMDConfig(Config):
    """
    一次 VMD 运行的有效参数快照 (只含标量)。

    字段与 ``VMD.__init__`` 的公开参数同名同义, 另加 ``dtype`` (返回 IMFs 的工作
    精度)。numpy 数组留在 ``DecompositionResult`` 里 —— 在 frozen dataclass 里放
    大数组会让 ``to_dict()`` 与相等比较又慢又容易出错。

    运行期才解析出来的存储策略 (``engine`` 的实际取值 / ``chunk_size`` /
    ``out_of_core``) 是单次运行的有效值, 记在 ``DecompositionResult.info`` 的
    同名键里, 不作为参数快照。

    字段与 vmdpy 的 ``VMD(f, alpha, tau, K, DC, init, tol)`` 对应: ``num_imf``↔
    ``K``, ``n``↔``Niter`` (500), ``epsilon``↔``tol``, ``init_mod``↔``init``
    (uniform/random/zero ↔ 1/2/0), ``DC``/``alpha``/``tau`` 同名同义。

    ``init_mod`` / ``engine`` 的注解是 ``Enum | Literal`` 联合: 调用方两种都收,
    而 ``config_generate()`` 填进来的是字符串 ``.value`` (不是 Enum 成员), 这样
    ``to_dict()`` 的返回值可以直接 ``json.dumps``。手工构造本 dataclass 时若直接
    传 Enum 成员, 值会原样保留 (dataclass 不做转换), 而 ``to_dict()`` 走
    ``dataclasses.asdict`` 不展开 Enum —— 那种快照需要调用方自行取 ``.value``。
    """

    num_imf: int
    n: int
    fs: float
    alpha: float
    tau: float
    epsilon: float
    DC: bool
    init_mod: VMD_INIT_MODS | VMD_INIT_MODS_ARGS
    seed: int | None
    engine: VMD_ENGINE_MODS | VMD_ENGINE_MODS_ARGS
    chunk_size: int | None
    out_of_core: bool | None
    store_history: bool
    vmdpy: bool
    dtype: type | np.dtype | None


"""
Class and Function of decomposer
"""
class VMD(Decomposer):
    """
    变分模态分解 (原生 ADMM 实现)。

    在频域求解 VMD 变分问题, 把一维信号分成 ``K`` 个带限模态:

    1. 镜像延拓到 ``T = 2N`` 个样本 (控制边界效应);
    2. 只保留频谱的正频半边 (``T/2 = N`` 个 bin);
    3. 逐模态交替: 在中心频率 ``ω_k`` 附近对 ``û_k`` 做维纳滤波更新, 再做功率加权
       的 ``ω_k`` 更新; 之后对 ``λ̂`` 做对偶上升;
    4. ``(1/T)·Σ_k‖Δû_k‖² ≤ epsilon`` 或迭代满 ``n`` 次时停止;
    5. 由单边谱重建实模态, 去掉镜像部分。

    用法 ``VMD(**params).decompose(S, T)``; 结果携带冻结的 ``VMDConfig`` 参数
    快照, ``IMFs`` 形状 (K, N)。

    无残差
    ------
    VMD 没有"残差"概念: 数据保真度是目标函数里的二次惩罚项 (``tau=0`` 时退化为
    软约束), 模态之和本身就是重构信号, 因此没有需要单独返回的剩余分量。
    ``DecompositionResult`` 允许没有残差的方法返回 ``Res=None``, 这里如实返回
    ``None``, 而不是人为构造一个 ``S − ΣIMFs``。需要了解模态对信号的拟合程度时
    读诊断量 ``info["residual_ratio"]`` (``‖S − ΣIMFs‖/‖S‖``)。

    模态顺序
    --------
    ``IMFs`` 的第 ``k`` 行跟随第 ``k`` 个初始中心频率 (``ω`` 初值升序), 与
    vmdpy/MATLAB 相同: 自由演化后最终 ``ω`` 可能交叉, 因此行序不保证按中心频率
    升序。需要重排时用 ``info["omega"]`` (例如
    ``order = np.argsort(info["omega"])``)。
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
        init_mod: VMD_INIT_MODS_ARGS | VMD_INIT_MODS = "uniform",
        seed: int | None = None,
        engine: VMD_ENGINE_MODS_ARGS | VMD_ENGINE_MODS = "auto",
        chunk_size: int | None = None,
        out_of_core: bool | None = None,
        store_history: bool = False,
        vmdpy: bool = False,
        dtype: type | np.dtype | None = None,
        config: VMDConfig | None = None,
        **kwargs,
    ) -> None:
        """
        Parameters
        ----------
        num_imf : int
            模态数 ``K``; 须满足 ``1 <= K <= N//2``。
        n : int
            ADMM 最大迭代次数 (即 vmdpy 的 ``Niter``, 默认 500)。
        fs : float
            采样频率, 只用于报告 ``omega_hz = omega * fs``; 不改变分解结果,
            因为 VMD 与频率尺度无关。
        alpha : float
            带宽约束与数据保真项之间的平衡参数 (> 0)。
        tau : float
            对偶上升步长 (取 0 表示允许噪声松弛, 即不更新拉格朗日乘子)。
        epsilon : float
            ``(1/T)·Σ_k‖Δû_k‖²`` 的收敛容差; 取 0 表示跑满 ``n`` 次迭代
            (与 vmdpy 对照时用)。
        DC : bool
            为真时第一个模态锁在直流 (``ω_0 = 0``)。必须是真正的 ``bool``
            (``1``/``0`` 被拒绝)。
        init_mod : {"uniform", "random", "zero", "peak"} | VMD_INIT_MODS
            中心频率初值; ``"uniform"``/``"random"``/``"zero"`` 分别对应 vmdpy 的
            ``init=1/2/0``, ``"peak"`` 从谱中最强的几个峰起步 (``Utils.Peaks``,
            仅原生实现支持)。字符串与 ``VMD_INIT_MODS`` 成员都接受, 统一规范化为
            Enum。
        seed : int | None
            局部随机种子 (只影响 ``init_mod="random"``)。``None`` 时退回进程级种子
            (``Base.Seed``); 给了非 None 的局部种子就以它为准, 不会被静默覆盖。
        engine : {"ram", "chunked", "auto"} | VMD_ENGINE_MODS
            频轴扫描引擎。``"auto"`` (默认) 交给全局内存策略; ``"ram"`` 全部放
            内存 (最快, 且与单块扫描逐位相同); ``"chunked"`` 频轴分块扫描 (工作集
            几乎与 ``N`` 无关); ``"auto"`` 在投影工作集触发
            ``Base.Memory.should_use_memmap`` 时选 ``"chunked"``。字符串与
            ``VMD_ENGINE_MODS`` 成员都接受, 统一规范化为 Enum。解析后的引擎
            (``"auto"`` → ``"ram"``/``"chunked"``) 以字符串记在 ``info["engine"]``,
            不写进参数快照。
        chunk_size : int | None
            ``engine="chunked"`` 的频轴分块长度。``None`` 时按
            ``VMD_CHUNK_WORK_BYTES`` 与全局内存策略推导; 取值 ``>= N`` 退化为单块。
        out_of_core : bool | None
            把镜像 / 频谱 / 模态谱放进临时 ``np.memmap`` 文件而不是内存 (只对
            ``engine="chunked"`` 有意义)。``None`` 时每次运行由
            ``Base.Memory.should_use_memmap`` 决定; ``True``/``False`` 强制指定。
        store_history : bool
            是否保留逐次迭代历史: 为真时 ``info`` 额外带 ``u_hat_history``
            (n_iter, K, N) 与 ``udiff_history`` (n_iter,), 代价是 ``n_iter·K·N``
            个复数。分块引擎里模态谱本来就是逐块写入的, 因此开启它不会多出一次
            整谱拷贝。默认 False。
        dtype : type | np.dtype | None
            返回 ``IMFs`` 的工作精度。``None`` 跟随输入精度
            (``float32``/``float64``); 整型 / bool / ``float16`` 输入升到
            float64。给了就必须是浮点 numpy dtype。
        config : VMDConfig, optional
            冻结的参数快照; 给了它就覆盖其余全部参数 (包括 ``dtype``)。运行结果
            另带一份由 :meth:`config_generate` 新建的快照。
        **kwargs
            ``safe`` (校验等级, 默认 ``"unknown"``) 与 ``copy`` (是否复制调用方
            输入, 默认 ``True``); 其余未知关键字直接报 ``TypeError``。

        Raises
        ------
        ValueError
            ``num_imf``/``n`` 不是整数或越界, ``alpha``/``fs`` 非正, ``tau``/
            ``epsilon`` 为负, ``DC`` 非法, ``init_mod``/``engine`` 未知,
            ``chunk_size`` 小于 1, 或 ``dtype`` 不是浮点。
        TypeError
            给了 ``config`` 但不是 ``VMDConfig``、关键字未知, 或 ``seed`` 非法。

        Notes
        -----
        没有并行开关: 模态间独立 (Jacobi) 更新在频带重叠时会发散, 分块并行也显著
        增加迭代次数, 而多线程实测无收益。
        """
        unknown = set(kwargs) - {"safe", "copy"}
        if unknown:
            raise TypeError(f"unexpected keyword arguments: {unknown}")

        self.safe: str = kwargs.get("safe", "unknown")
        self.copy: bool = kwargs.get("copy", True)

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
            self.engine = config.engine
            self.chunk_size = config.chunk_size
            self.out_of_core = config.out_of_core
            self.store_history = config.store_history
            self.vmdpy = config.vmdpy
            self.dtype = config.dtype
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
            self.dtype = dtype
        else:
            raise TypeError(
                f"config must be a VMDConfig or None, got {type(config).__name__}"
            )

        # 运行期状态: 每次 decompose 开头重置。VMD 没有残差, ``res`` 恒为 None。
        self.imfs: np.ndarray | None = None
        self.res: np.ndarray | None = None

        self.param_check()

    def param_check(self) -> None:
        """
        校验并规范化全部构造参数。

        非法值在这里就抛错, 而不是延后到 ``decompose``。除尺寸与取值范围外, 还
        检查 ``dtype`` / ``seed`` / ``safe`` / ``copy`` 的类型。
        """
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

        # 两个字符串形参按库内约定统一转成 Enum (实例上此后只存 Enum, 内部分支只比 Enum)。
        self.init_mod = Check.validate_mod(self.init_mod, VMD_INIT_MODS)

        self.engine = Check.validate_mod(self.engine, VMD_ENGINE_MODS)

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

        if self.seed is not None:
            if isinstance(self.seed, bool) or not isinstance(self.seed, (int, np.integer)):
                raise TypeError(f"seed must be an int >= 0 or None, got {self.seed!r}")
            if int(self.seed) < 0:
                raise ValueError(f"seed must be non-negative, got {self.seed!r}")
            self.seed = int(self.seed)

        # dtype 只允许浮点: 返回的 IMFs 是实数模态, 整型/复数精度没有意义。
        if self.dtype is not None and np.dtype(self.dtype).kind != "f":
            raise ValueError("the `dtype` must be a float numpy dtype or None.")

        self.safe = Check.validate_safe(self.safe)

        if not isinstance(self.copy, bool):
            raise TypeError("the `copy` must be bool")

    def config_generate(self) -> VMDConfig:
        """
        构造本次参数快照 (只含标量构造参数; 运行期解析出的有效存储策略见 ``info``)。

        ``init_mod`` / ``engine`` 存 Enum 成员, 与 RPSEMD 的 ``hilbert_mod`` 口径一致
        (``Config.to_dict()`` 是 ``dataclasses.asdict``, 不展开 Enum)。
        """
        return VMDConfig(
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
            dtype=self.dtype,
        )

    def decompose(self, S, T=None, output_memmap: bool = False) -> DecompositionResult:
        """
        把信号分解成 ``num_imf`` 个带限模态。

        Parameters
        ----------
        S : array-like
            一维实信号 (也可以是 ``.npy`` 路径或长序列 —— ``Check_Time_and_Signal``
            的输入层会套用全局内存策略)。
        T : array-like, optional
            只为签名兼容而接受, 实际忽略: VMD 运行在样本序号上, 时间轴既不校验也
            不使用 (``need_T=False`` 还额外省掉一份 ``arange(N)`` 临时数组)。

        Returns
        -------
        DecompositionResult
            ``IMFs`` (K,N), 模态按初值顺序排列 (见类 docstring 的"模态顺序");
            ``Res`` 恒为 None (VMD 没有残差概念); ``info`` 含

            ``omega`` (K,)
                最终中心频率, 归一化循环频率 (镜像轴);
            ``omega_hz`` (K,)
                同一组频率的 Hz 值 (``omega * fs``);
            ``omega_history`` (n_iter, K)
                每次完成迭代时的中心频率;
            ``u_hat`` (N, K)
                返回模态的频谱 (``fftshift(fft(mode))``, 与 vmdpy 的
                ``info["u_hat"]`` 同一约定); 超过 ``VMD_UHAT_INFO_LIMIT`` 时
                为 None;
            ``n_iter`` (int), ``converged`` (bool), ``init_mod`` (str),
            ``seed`` (int, 本次运行实际使用的种子);
            ``residual_ratio`` (float, ``‖S − ΣIMFs‖/‖S‖``) —— 纯诊断量, 不作为
            ``Res`` 返回;
            ``engine``/``chunk_size``/``out_of_core`` (本次运行解析出的存储策略,
            ``engine`` 记解析后的 ``.value`` 字符串, 如 ``"auto"`` → ``"ram"``),
            ``projected_bytes`` (dict), ``fft_backend`` (str)。
        """
        # 形参校验先于任何计算: 传错类型应立即失败, 而不是跑完整个分解再报错
        check_output_memmap(output_memmap)
        # 每次调用都从干净状态开始 (实例上不累积任何运行期数组)。
        self.imfs = None
        self.res = None

        # VMD 只用样本序号: need_T=False 省掉 arange(N) 那份与信号等长的临时数组。
        S, N = Check_Time_and_Signal(
            S,
            T=None,
            ndim={1},
            method=self.name,
            dtype=self.dtype,
            need_T=False,
            safe=self.safe,
            copy=self.copy,
        )

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

        # bool/int 无法参与频域浮点运算, f16 精度不足: 统一提升到 float64;
        # float32/float64 原样保留, 返回的 IMFs 与输入同精度。
        if not (S.dtype.kind == "f" and S.dtype.itemsize >= 4):
            S = Check.to_dtype(S, np.dtype(np.float64))

        out_dtype = VMD._work_dtype(S.dtype)
        f = np.asarray(S, dtype=np.float64).reshape(-1)

        # --- vmdpy 分支: 交给可选的第三方实现 (不强制安装) ----------------- #
        if self.vmdpy:
            return self._decompose_vmdpy(S, f, N, K, out_dtype, output_memmap)

        # 局部种子优先; 未给时退回进程级全局种子 (Base.Seed)。
        effective_seed = Seed.get() if self.seed is None else int(self.seed)
        rng = np.random.default_rng(effective_seed)

        T_len = 2 * N

        # --- 引擎 / 分块粒度 / 落盘策略解析 (全部走 Base 的既有策略) ------ #
        proj_ram = self._project_bytes(N, K, False, N)
        engine = self.engine
        if engine is VMD_ENGINE_MODS.Auto:
            engine = (
                VMD_ENGINE_MODS.Chunked
                if should_use_memmap(proj_ram["total"])
                else VMD_ENGINE_MODS.RAM
            )

        if engine is VMD_ENGINE_MODS.RAM:
            chunk_size = N                      # 单块 == 不分块
            out_of_core = False
        else:
            if self.chunk_size is None:
                # 策略感知粒度: 目标"每块工作集 ≈ VMD_CHUNK_WORK_BYTES" (K 个模态各需
                # 旧值/新值/累加, 约 3 份复数), 再交给 Base.Chunk 按内存策略收缩。
                # 直接用 Chunk 的默认 8M 频点基数在 K 较大时会得到 GB 级工作集,
                # 故这里先按 K 折算基数。
                base = max(1, VMD_CHUNK_WORK_BYTES // (K * 16 * 3))
                chunk_size = min(base, int(Chunk.chunk_size(N * 16)), N)
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

        if engine is VMD_ENGINE_MODS.RAM:
            modes, omega, omega_history, n_iter, converged, init_used = self._engine_ram(
                f, N, K, rng, T_len, u_hist, udiff_hist
            )
        else:
            modes, omega, omega_history, n_iter, converged, init_used = self._engine_chunked(
                f, N, K, rng, T_len, chunk_size, out_of_core, u_hist, udiff_hist
            )

        IMFs = modes.astype(out_dtype, copy=False)

        # VMD 没有残差概念: 模态和就是重构信号, Res 如实返回 None (不硬造一个)。
        norm_f = float(np.linalg.norm(f))
        residual_ratio = (
            float(np.linalg.norm(f - modes.sum(axis=0))) / norm_f if norm_f > 0.0 else 0.0
        )

        # u_hat 诊断谱要 K·N 个复数: 超过 VMD_UHAT_INFO_LIMIT 时不生成, 记 None。
        if K * N * 16 <= VMD_UHAT_INFO_LIMIT:
            u_hat_info = np.fft.fftshift(np.fft.fft(modes, axis=1), axes=1).T
        else:
            u_hat_info = None

        info = {
            "omega": omega.copy(),
            "omega_hz": omega * self.fs,
            "omega_history": omega_history[:n_iter].copy(),
            "u_hat": u_hat_info,
            "fft_backend": "numpy",
            "n_iter": n_iter,
            "converged": converged,
            "init_mod": init_used.value,
            "seed": effective_seed,
            "residual_ratio": residual_ratio,
            "engine": engine.value,
            "chunk_size": chunk_size,
            "out_of_core": out_of_core,
            "projected_bytes": projection,
        }
        if u_hist is not None:
            info["u_hat_history"] = u_hist[:n_iter].copy()
            info["udiff_history"] = udiff_hist[:n_iter].copy()

        self.imfs = IMFs

        return finalize_output(
            DecompositionResult(
                IMFs=IMFs,
                Res=None,
                info=info,
                config=self.config_generate(),
            ),
            output_memmap,
        )

    # ------------------------------------------------------------------ #
    # 可选分支: 用第三方 vmdpy 分解 (vmdpy=True)
    # ------------------------------------------------------------------ #
    def _decompose_vmdpy(self, S, f, N, K, out_dtype, output_memmap: bool = False) -> DecompositionResult:
        """
        调用可选的第三方实现 ``vmdpy`` (经 ``Base.Cache`` 惰性导入)。

        本分支用于对照 / 复现参考实现, 与原生分支的差异:
        - 迭代上限固定 500 (vmdpy 内部硬编码, ``n`` 不生效), 收敛判据为其原式;
        - 初值只支持 ``zero``/``uniform``/``random`` (``init_mod="peak"`` 仅原生支持);
        - 奇数长度不支持 (vmdpy 内部会丢末样本, 与本库返回契约冲突, 这里直接报错);
        - ``fs`` / ``seed`` / ``store_history`` / ``engine`` / ``chunk_size`` /
          ``out_of_core`` 在本分支不生效 (无迭代历史可存, 也不支持分块/外存);
        - 未安装 vmdpy 时抛 ``ImportError`` 并给出安装命令 —— 它是可选依赖, 不强制安装。
        """
        if N % 2 != 0:
            raise ValueError(
                f"{self.name}(vmdpy=True): vmdpy 会丢弃末样本, 仅支持偶数长度; got N={N}"
            )
        match self.init_mod:                 # vmdpy 只认 0/1/2, 转换就地做
            case VMD_INIT_MODS.Zero:
                init_code = 0
            case VMD_INIT_MODS.Uniform:
                init_code = 1
            case VMD_INIT_MODS.Random:
                init_code = 2
            case _:
                raise ValueError(
                    f"{self.name}(vmdpy=True): init_mod={self.init_mod!r} 仅原生实现支持; "
                    "vmdpy 分支请用 ('zero', 'uniform', 'random')"
                )

        # Cache.import_module 在模块缺失时返回 None (不抛 ImportError), 这里转成
        # 带安装命令的 ImportError, 让调用方拿到可直接执行的安装命令。
        vmdpy = Cache.import_module(
            "vmdpy",
            description="vmdpy: 可选第三方 VMD 实现 (VMD(vmdpy=True) 分支使用)",
        )
        if vmdpy is None:
            raise ImportError(
                "VMD(vmdpy=True) 需要可选的第三方实现 vmdpy; 请先安装:\n"
                "    %s -m pip install vmdpy" % sys.executable
            )

        # vmdpy.VMD(f, alpha, tau, K, DC, init, tol) -> (u, u_hat, omega_history)
        u, u_hat, omega_hist = vmdpy.VMD(
            f, self.alpha, self.tau, K, int(self.DC), init_code, self.epsilon
        )
        modes = np.asarray(u)
        omega_history = np.asarray(omega_hist)
        omega = omega_history[-1] if omega_history.ndim == 2 else omega_history
        n_iter = int(omega_history.shape[0]) if omega_history.ndim == 2 else 0

        IMFs = modes.astype(out_dtype, copy=False)
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
            "init_mod": self.init_mod.value,
            "seed": self.seed,
            "residual_ratio": (float(np.linalg.norm(f - modes.sum(axis=0))) / norm_f)
            if norm_f > 0.0 else 0.0,
        }

        self.imfs = IMFs

        return finalize_output(
            DecompositionResult(
                IMFs=IMFs,
                Res=None,
                info=info,
                config=self.config_generate(),
            ),
            output_memmap,
        )

    # ------------------------------------------------------------------ #
    # 引擎 1: 全内存单块扫描
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
        f_mirr = self._mirror_signal(f)
        # fftshift(fft(f_mirr))[T//2:] 等价于 rfft 的前 N 个 bin (Nyquist 被丢弃)
        f_hat_plus = np.fft.rfft(f_mirr)[:N]
        del f_mirr

        init_used = self.init_mod
        omega = self._init_omega(self.init_mod, K, freqs, f_hat_plus, rng, N)
        if self.DC:
            omega[0] = 0.0

        # 模态谱按 (K, N) 行主序存放, 内循环全是连续扫描。
        # 顺序更新只需要当前模态的旧值, 故只留一个 (N,) 暂存。
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
                np.copyto(old, u_hat[k])            # 第 k 个模态的旧值
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
            # 乘以 0.5 是 2 的幂次缩放, 与 IEEE 舍入可交换, 故与保留 λ̂ 的写法逐位相同。
            np.subtract(sum_all, f_hat_plus, out=work)
            work *= self.tau
            work *= 0.5
            lam_half += work

            omega_history[it] = omega
            if u_hist is not None:
                u_hist[it] = u_hat              # 该次迭代后的模态谱
                udiff_hist[it] = udiff
            n_iter = it + 1

            if udiff / T_len <= self.epsilon + np.spacing(1.0):
                converged = True
                break

        half = np.zeros((K, N + 1), dtype=np.complex128)
        half[:, :N] = u_hat       # 单边谱不含 Nyquist 分量 (被单边化丢弃)
        full = np.fft.irfft(half, n=T_len, axis=1)      # (K, 2N)
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

        与全内存路径的唯一数值差异是两个归约 (ω 的功率加权均值、收敛量
        ``Σ_k‖Δû_k‖²``) 的求和顺序 (逐块部分和再相加), 其余逐元素运算完全一致。

        ``u_hist`` / ``udiff_hist`` 非 None 时就地写入逐次迭代历史: 模态谱在分块循环
        内逐块写入, 因此开 store_history 不会多出一次全谱拷贝。

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
        self._mirror_signal(f, out=mirror, chunk_size=chunk_size)

        # --- 2) 单边谱: 全数组变换, 无法分块; 这是外存的"内存地板" -------- #
        spec = np.fft.rfft(mirror)                     # (N+1,) complex128, 常在内存
        if out_of_core:
            # force=True: 先关句柄再删文件 (Windows 下映射未关时 os.remove 会 PermissionError,
            # 文件要留到进程退出才清)。此后再不读 mirror。
            drop_memmap(mirror, force=True)
            spec_store = temp_memmap((N,), np.complex128)
            for a, b in self._spans(N, chunk_size):
                spec_store[a:b] = spec[a:b]
            f_hat_plus = spec_store
            del spec
        else:
            f_hat_plus = spec[:N]

        # --- 3) ω 初值 (peak 模式需要整谱幅度: 大数组下退化为 uniform) ------ #
        init_used = self.init_mod
        if init_used is VMD_INIT_MODS.Peak and N * 8 > VMD_PEAK_INIT_LIMIT:
            init_used = VMD_INIT_MODS.Uniform
            omega = self._init_omega(VMD_INIT_MODS.Uniform, K, None, None, rng, N)
        else:
            omega = self._init_omega(
                init_used, K, np.arange(N, dtype=np.float64) / T_len,
                np.abs(f_hat_plus) if init_used is VMD_INIT_MODS.Peak else None, rng, N,
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
            full = np.fft.irfft(half, n=T_len)
            modes[k] = full[left:left + N]

        # 外存工作区用尽即释放 (关句柄 + 删文件), 不留给进程退出时的清理兜底。
        # 三块缓冲此后都不再被读取, 故 force=True 关闭映射后立即删文件。
        if out_of_core:
            drop_memmap(u_hat, force=True)
            drop_memmap(lam_half, force=True)
            drop_memmap(f_hat_plus, force=True)     # 即 spec_store

        return modes, omega, omega_history, n_iter, converged, init_used

    # ------------------------------------------------------------------ #
    # 端点镜像 (VMD / vmdpy 的整条信号语义)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _mirror_signal(f, out=None, chunk_size=None):
        """
        整条信号的端点镜像延拓 ``[flip(S[:L]), S, flip(S[N-R:])]``。

        ``L = N//2``、``R = N - N//2`` ⇒ 总长恰为 ``2N`` (奇数 N 也不丢样本),
        与 vmdpy 的 ``fMirr`` 逐点一致; 反射是 "symmetric" 约定 (重复边界样本,
        ``x[-k] = S[k-1]``), 不能用 ``np.pad(mode="reflect")`` 代替。

        Parameters
        ----------
        f : np.ndarray | np.memmap
            1-D 信号。
        out : np.ndarray | np.memmap, optional
            目标缓冲 (外存路径下即临时 memmap); 长度须为 ``2N``。
        chunk_size : int | None
            分块搬运粒度 (与 ``out`` 搭配限制工作集); None 时一次写入。

        Returns
        -------
        np.ndarray
            长度 ``2N`` 的镜像信号 (``out`` 给定时即 ``out``)。
        """
        n = f.size
        left = n // 2
        right = n - left
        total = left + n + right

        if out is None:
            out = np.empty(total, dtype=f.dtype)
        elif out.shape[0] != total:
            raise ValueError(f"out length {out.shape[0]} != 2N = {total}")

        step = max(1, int(chunk_size)) if chunk_size else max(1, total)

        # 分块搬运按目标区间对齐取源 (写成 src[a:b][::-1] 只在单块时正确)。
        for a in range(0, left, step):
            b = min(a + step, left)
            out[a:b] = f[left - b:left - a][::-1]
        for a in range(0, right, step):
            b = min(a + step, right)
            out[left + n + a:left + n + b] = f[n - b:n - a][::-1]

        out[left:left + n] = f
        if hasattr(out, "flush"):
            out.flush()
        return out

    @staticmethod
    def _work_dtype(dtype) -> np.dtype:
        """float32/float64 保持原精度; 其余 (float16/int/bool) 提升为 float64。"""
        dtype = np.dtype(dtype)
        if dtype.kind == "f" and dtype.itemsize >= 4:
            return dtype
        return np.dtype(DEFAULT_NUMPY_TYPE)

    @staticmethod
    def _project_bytes(N: int, K: int, out_of_core: bool, chunk_size: int) -> dict:
        """
        估算一次分解的主要内存项 (字节), 供 ``Base.Memory`` 策略判定与外部对照使用。

        以 float64 / complex128 计 (``np.fft`` 的工作精度), 按引擎分别建模:

        * 内存内 (``out_of_core=False``): 镜像 + 单边谱 + 模态谱 + λ̂/2 + 累加器 + 频率轴
          + 逐模态暂存 + (K 个模态一起做的) 逆变换与其输出;
        * 外存 (``out_of_core=True``): 镜像/模态谱/λ̂ 落盘 ⇒ 常驻项只剩单边谱
          (``np.fft`` 无法分块, 这是"内存地板") + 结果本身 (K·N float64 的 IMFs,
          返回契约要求它在内存里) + 逐模态逆变换缓冲 + 单块工作集。

        返回各项与 ``total``; 这是量级估算 (未计 FFT 内部临时量), 用于决策与对照实测。
        """
        c16, f8 = 16, 8
        out_modes = K * N * f8                      # 返回的 IMFs (float64 时与工作数组同体)
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
                "output": out_modes,
                "chunk_ws": K * chunk_size * c16 * 3 + chunk_size * (c16 + f8),
            }
        else:
            terms = {
                "input": N * f8,
                "mirror": 2 * N * f8,
                "spectrum": floor,
                "modes": K * N * c16,               # û 模态谱
                "dual": N * c16,                    # λ̂/2 (单份)
                "accum": N * c16,                   # Σ_i û_i
                "freqs": N * f8,
                "scratch": N * c16 * 3 + N * f8 * 2,   # 旧值/残差/增量 + denom/power
                "recon": K * floor + K * 2 * N * f8,   # K 个模态一起逆变换
                "output": out_modes,
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
        init_mod: VMD_INIT_MODS,
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
        init_mod : VMD_INIT_MODS
            ``VMD_INIT_MODS.Zero``    — ``ω ≡ 0`` (vmdpy ``init=0``);
            ``VMD_INIT_MODS.Uniform`` — ``ω_i = 0.5·i/K`` (vmdpy ``init=1``, 与
                            vmdpy/MATLAB 逐位一致; 该网格覆盖 ``[0, 0.5)``, 而
                            正频半轴只到 ``0.25``);
            ``VMD_INIT_MODS.Random``  — ``[1/N, 0.5]`` 对数均匀取 K 点后升序
                            (vmdpy ``init=2``);
            ``VMD_INIT_MODS.Peak``    — 单边幅度谱最强 K 个峰 (``Utils.Peaks``,
                            numpy 后端), 峰不足时用谱内等间隔点补齐 —— 自研扩展。
        K : int
            模态数。
        freqs : np.ndarray
            正频半轴归一化频率 (``j/T``), 长度 N。
        f_hat_plus : np.ndarray
            镜像信号的单边谱 (长度 N), ``VMD_INIT_MODS.Peak`` 据此找峰。
        rng : np.random.Generator
            局部随机数发生器 (``VMD_INIT_MODS.Random`` 使用)。
        N : int
            原始信号长度 (随机初值下界 ``1/N`` 即原始记录的分辨率)。

        Returns
        -------
        np.ndarray
            长度 K、升序的初值频率 (可能含重复值, 例如 ``VMD_INIT_MODS.Zero``)。
        """
        if init_mod is VMD_INIT_MODS.Zero:
            return np.zeros(K, dtype=np.float64)

        if init_mod is VMD_INIT_MODS.Uniform:
            return (0.5 / K) * np.arange(K, dtype=np.float64)

        if init_mod is VMD_INIT_MODS.Random:
            lo, hi = 1.0 / N, 0.5
            w = np.exp(np.log(lo) + (np.log(hi) - np.log(lo)) * rng.random(K))
            return np.sort(w)

        # VMD_INIT_MODS.Peak: Utils.Peaks 每次 decompose 解析一次, 迭代内零开销。
        idx, heights = find_peaks(np.abs(f_hat_plus), mod="max", backend="numpy")

        if idx.size == 0:
            return (0.5 / K) * np.arange(K, dtype=np.float64)

        heights = np.asarray(heights)
        strongest = np.argsort(heights, kind="stable")[::-1][:K]
        picked = np.sort(idx[strongest]).astype(np.float64) / (2.0 * N)

        if picked.size < K:
            # 峰不足 K 个: 其余初值均匀铺在正频半轴内, 再整体升序。
            tail = np.linspace(freqs[0], freqs[-1], K - picked.size + 2)[1:-1]
            return np.sort(np.concatenate([picked, tail]))

        return picked


"""
other function for module
"""
def f_VMD(S: list | np.ndarray, T=None, output_memmap: bool = False, *, num_imf: int = 2, n: int = 500, fs: float = 1.0, alpha: float = 2000.0, tau: float = 0.0, epsilon: float = 1e-07, DC: bool = False, init_mod: VMD_INIT_MODS_ARGS | VMD_INIT_MODS = "uniform", seed: int | None = None, engine: VMD_ENGINE_MODS_ARGS | VMD_ENGINE_MODS = "auto", chunk_size: int | None = None, out_of_core: bool | None = None, store_history: bool = False, vmdpy: bool = False, dtype: type | np.dtype | None = None, config: VMDConfig | None = None, **kwargs) -> DecompositionResult:
    vmd = VMD(
        num_imf=num_imf,
        n=n,
        fs=fs,
        alpha=alpha,
        tau=tau,
        epsilon=epsilon,
        DC=DC,
        init_mod=init_mod,
        seed=seed,
        engine=engine,
        chunk_size=chunk_size,
        out_of_core=out_of_core,
        store_history=store_history,
        vmdpy=vmdpy,
        dtype=dtype,
        config=config,
        **kwargs
    )

    return vmd.decompose(S, T, output_memmap=output_memmap)
