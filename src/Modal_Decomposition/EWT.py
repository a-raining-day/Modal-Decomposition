"""
Empirical Wavelet Transform —— 本库默认实现 (自研经验小波滤波器组)。

注册键 ``"EWT"`` 由本文件提供 (``Class.EWT`` / ``Function.EWT``)。

频域支撑上的自适应滤波器组由 ``Operator`` 层提供: ``Daubechies_polynomial``
给出单位分解光滑阶跃 β(x) (β(x)+β(1−x)=1), ``Empirical_wavelet`` 用 β 构造
带过渡带 γ 的经验小波传递函数 ``H(w0, w1)`` (``w0=0`` 时退化为低通尺度函数)。

滤波器组、边界检测、变换与逆变换全部在本库内完成, **不依赖 ewtpy**;
依赖 ewtpy 的适配层在 ``EWTpy.py``, 注册键 ``"EWTpy"``。与 Gilles 2013
原文/ewtpy 的对应关系:
``Σ_k H_k²(ω) ≡ 1`` (能量型紧框架, 逐点验证误差 < 1e-16); 而幅度型
``Σ_k H_k(ω)`` 在过渡带内最大到 √2, 故单次滤波所得各带之和并不等于原信号 ——
本实现把该差额全部放进 ``Res = S − ΣIMFs``, 于是 ``reconstruct()`` 仍精确。

单位约定
--------
``boundaries`` / ``frequency`` 一律是**频率** (Hz, 0…fs/2, 取自 ``rfftfreq``),
不是频点下标; ``band_factor`` 是过渡带相对宽度 (0<γ<1), 过渡带为
``[(1−γ)w, (1+γ)w]``; 默认 ``band_factor="auto"`` 时 γ 由边界间距反推
(见 :meth:`EWT._resolve_band_factor`), 以保证相邻过渡带不交叠、``Σ H_k² ≡ 1``
逐点成立 —— 显式传 float 则用固定宽度。

参数类型约定
------------
字符串型分支参数一律采用项目既有的 ``Literal`` + ``Enum`` 约定 (同
``Utils._Hilbert`` / ``Utils.Mirror`` / ``SVMD``): 模块全局先定义
``EWT_*_MODS_ARGS = Literal[...]``, 紧跟 ``EWT_*_MODS(Enum)``; 形参注解写
``EWT_*_MODS_ARGS | EWT_*_MODS``, 内部一律经 :meth:`Utils.Check.validate_mod`
归一成 Enum 成员后再用 ``match`` 分支 (不直接比较字符串)。

"数字或字符串" 混合语义的参数只有 ``band_factor``: 数值即过渡带宽度, 原样
保留为 float; 只有字符串形态 (``"auto"``) 走 ``EWT_BAND_FACTOR_MODS``。

References
----------
10.48550/arXiv.2304.06274
"""

import numpy as np
from dataclasses import dataclass
from enum import Enum
from typing import ClassVar, Literal, Sequence

from .Define import (
    BIG_ARRAY,
    DEFAULT_NUMPY_TYPE,
    EWT_CHUNK_WORK_BYTES,
    EWT_SLEPIAN_MAX_SAMPLES,
    Config,
    Decomposer,
    DecompositionResult,
    check_output_memmap,
    finalize_output,
)
from .Base import Cache
from .Utils import Check, Check_Time_and_Signal, find_peaks, mirror, slepian
from .Operator import empirical_wavelet_arr

__all__ = [
    "EWT",
    "EWTConfig",
    "EWT_BOUNDARY_MODS",
    "EWT_BOUNDARY_MODS_ARGS",
    "EWT_BAND_FACTOR_MODS",
    "EWT_BAND_FACTOR_MODS_ARGS",
    "EWT_PRE_DEAL_MODS",
    "EWT_PRE_DEAL_MODS_ARGS",
    "EWT_NO_DC_TYPE_MODS",
    "EWT_NO_DC_TYPE_MODS_ARGS",
    "EWT_WINDOW_KIND_MODS",
    "EWT_WINDOW_KIND_MODS_ARGS",
    "EWT_SLEPIAN_MODS",
    "EWT_SLEPIAN_MODS_ARGS",
    "EWT_SMOOTH_KIND_MODS",
    "EWT_SMOOTH_KIND_MODS_ARGS",
    "EWT_MAX_MIN_FALLBACK_MODS",
    "EWT_MAX_MIN_FALLBACK_MODS_ARGS",
    "EWT_ENVELOPE_SPLINE_MODS",
    "EWT_ENVELOPE_SPLINE_MODS_ARGS",
    "f_EWT",
]

#: ``boundary_mod``: 谱边界检测策略 (四套规则见 :meth:`EWT.boundary`)。
EWT_BOUNDARY_MODS_ARGS = Literal["maximum", "max-min", "scale-space", "envelope"]
#: ``band_factor``: 只有字符串 ``"auto"`` 走 Enum; 数值即过渡带相对宽度 γ。
EWT_BAND_FACTOR_MODS_ARGS = Literal["auto"]
#: ``pre_deal``: 预处理分支名 (可传单个名或序列; 按 canonical 顺序施加)。
EWT_PRE_DEAL_MODS_ARGS = Literal["no-dc", "no-trend", "window", "Slepian-Optimize", "Smooth"]
#: ``no_dc_type``: ``scipy.signal.detrend`` 的 ``type``。
EWT_NO_DC_TYPE_MODS_ARGS = Literal["linear", "constant"]
#: ``window_kind``: 时域窗型 (``"hanning"`` 是 ``"hann"`` 的别名, 归一后再校验)。
EWT_WINDOW_KIND_MODS_ARGS = Literal["hann", "hamming"]
#: ``slepian_mod``: Slepian 后端 (``None`` = ``ConstDefine.SLEPIAN_BACKEND``)。
EWT_SLEPIAN_MODS_ARGS = Literal["auto", "numpy", "scipy", "C"]
#: ``smooth_kind``: 频域平滑窗型 (``Smooth`` 分支)。
EWT_SMOOTH_KIND_MODS_ARGS = Literal["box", "gaussian"]
#: ``max_min_fallback``: ``max-min`` 段内无极小时的边界回退方式。
EWT_MAX_MIN_FALLBACK_MODS_ARGS = Literal["midpoint", "geometric"]
#: ``envelope_spline``: 包络样条后端 (成员值即 ``_SPLINE_CTORS`` 的键)。
EWT_ENVELOPE_SPLINE_MODS_ARGS = Literal["UnivariateSpline", "CubicSpline", "PCHIP", "Akima"]

class EWT_BOUNDARY_MODS(Enum):
    Maximum = "maximum"
    MaxMin = "max-min"
    ScaleSpace = "scale-space"
    Envelope = "envelope"

class EWT_BAND_FACTOR_MODS(Enum):
    Auto = "auto"

class EWT_PRE_DEAL_MODS(Enum):
    NoDC = "no-dc"
    NoTrend = "no-trend"
    Window = "window"
    SlepianOptimize = "Slepian-Optimize"
    Smooth = "Smooth"

class EWT_NO_DC_TYPE_MODS(Enum):
    Linear = "linear"
    Constant = "constant"

class EWT_WINDOW_KIND_MODS(Enum):
    Hann = "hann"
    Hamming = "hamming"

class EWT_SLEPIAN_MODS(Enum):
    Auto = "auto"
    Numpy = "numpy"
    Scipy = "scipy"
    C = "C"

class EWT_SMOOTH_KIND_MODS(Enum):
    Box = "box"
    Gaussian = "gaussian"

class EWT_MAX_MIN_FALLBACK_MODS(Enum):
    Midpoint = "midpoint"
    Geometric = "geometric"

class EWT_ENVELOPE_SPLINE_MODS(Enum):
    UnivariateSpline = "UnivariateSpline"
    CubicSpline = "CubicSpline"
    PCHIP = "PCHIP"
    Akima = "Akima"

"""
Config
"""
@dataclass(frozen=True, kw_only=True)
class EWTConfig(Config):
    """
    一次 EWT 运行的有效参数。

    ``band_factor`` 记录**实际使用**的 γ (``"auto"`` 在 ``decompose`` 中反推后
    落定为 float), ``fs`` 记录本次 ``decompose`` 的采样频率; 分支参数以标量字典
    快照保存 (只放标量, 不放 numpy 数组; 其中的 ``EWT_*_MODS`` 成员是 Enum)。
    """
    num_imfs: int
    band_factor: float | EWT_BAND_FACTOR_MODS
    boundary_mod: EWT_BOUNDARY_MODS
    pre_deal: tuple
    mirror: bool
    fs: float
    dtype: type | np.dtype | None
    pre_deal_kwargs: dict
    boundary_kwargs: dict

    def to_dict(self) -> dict:
        """
        运行参数的纯 dict 快照。

        覆写基类 ``Config.to_dict`` (基类用的是 ``dataclasses.asdict``, 会把
        ``Enum`` 成员原样留下, 于是 ``json.dumps`` 会报 "not JSON serializable")。
        这里递归把每个 ``Enum`` 换成它的 ``.value``, 嵌套在 ``pre_deal_kwargs`` /
        ``boundary_kwargs`` 里的分支 mod 因此也能直接 ``json.dumps``。
        """
        def plain(value):
            if isinstance(value, Enum):
                return value.value
            if isinstance(value, dict):
                return {k: plain(v) for k, v in value.items()}
            if isinstance(value, tuple):
                return tuple(plain(v) for v in value)
            if isinstance(value, list):
                return [plain(v) for v in value]
            return value

        return {k: plain(v) for k, v in super().to_dict().items()}

"""
Class and Function of decomposer
"""
class EWT(Decomposer):
    name: ClassVar[str] = "EWT"

    pre_deal_mod_constraint = ("None", "no-DC", "no-trend", "window", "Slepian-Optimize", "Smooth")
    pre_deal_mod = {"None", "no-DC", "no-trend", "window", "Slepian-Optimize", "Smooth"}
    boundary_mod_constraint = EWT_BOUNDARY_MODS_ARGS
    boundary_mods = {m.value for m in EWT_BOUNDARY_MODS}

    def __init__(
        self,
        num_imfs: int = 8,
        band_factor: float | EWT_BAND_FACTOR_MODS_ARGS | EWT_BAND_FACTOR_MODS = "auto",
        boundary_mod: EWT_BOUNDARY_MODS_ARGS | EWT_BOUNDARY_MODS = "maximum",
        pre_deal: EWT_PRE_DEAL_MODS_ARGS | Sequence[EWT_PRE_DEAL_MODS_ARGS] | None = None,
        mirror: bool = True,
        dtype: type | np.dtype | None = None,
        **kwargs
    ):
        """
        构造 EWT 分解器: 校验并归一全部分支参数, 非法值在实例化时立即报错。

        Parameters
        ----------
        num_imfs : int
            模态数 (滤波器组的频带数), 必须 >= 1。对边界策略它是**上限**:
            ``"maximum"`` / ``"max-min"`` / ``"scale-space"`` 恒给出 ``num_imfs``
            个频带, ``"envelope"`` 可能更少 (模式数由显著峰数自动确定)。
        band_factor : float | EWT_BAND_FACTOR_MODS_ARGS | EWT_BAND_FACTOR_MODS
            每个经验小波的过渡带相对宽度 γ: 过渡带位于边界 ``w`` 的
            ``[(1−γ)w, (1+γ)w]`` 区间。取值是 **数字或字符串** 的联合:

            * ``"auto"`` (``EWT_BAND_FACTOR_MODS.Auto``, 默认) —— **按检测到的
              边界间距反推**(见 :meth:`_resolve_band_factor`): ``γ`` 取"相邻过渡带
              恰好不交叠"的上确界, 于是滤波器组满足紧框架 ``Σ_i H_i² ≡ 1``,
              幅度带和误差最小 (与 ewtpy 的自动 gamma 同法)。
            * ``float`` (``0 < γ < 1``) —— 固定过渡带宽度; γ 偏大时相邻过渡带
              叠加 (``Σ H² > 1``, 同一频段被重复计入), 偏小则频带间出现重构空档。
              数值**不**走 Enum, 参数校验后原样归一成 ``float``。
        boundary_mod : EWT_BOUNDARY_MODS_ARGS | EWT_BOUNDARY_MODS
            谱边界检测策略 (由 :meth:`boundary` 集成, 四套规则见其文档)。字符串
            或 ``EWT_BOUNDARY_MODS`` 成员皆可, 实例上一律存 Enum 成员。
        pre_deal : EWT_PRE_DEAL_MODS_ARGS | Sequence[EWT_PRE_DEAL_MODS_ARGS] | None
            预处理分支选择 (单个名或序列; ``None``/``"None"`` = 不预处理)。
            取值见 ``EWT_PRE_DEAL_MODS_ARGS``, 也接受对应的 ``EWT_PRE_DEAL_MODS``
            成员 (内部一律经 ``Check.validate_mod`` 归一成 canonical 名字后存进
            ``self.pre_deal_mods``; 属性不叫 ``self.pre_deal``, 以免遮蔽同名的
            公共方法 :meth:`pre_deal`)。
            分支按 canonical 顺序施加: 时域 ``("no-dc", "no-trend", "window")``
            在 ``Check_Time_and_Signal`` 之后作用于信号, 频域
            ``("Slepian-Optimize", "Smooth")`` 在 ``fft`` 之后作用于
            **边界检测用的幅度谱** (滤波仍用原复数谱)。``mirror`` 不属于预处理
            (它由独立参数控制, 在时域预处理之后、``fft`` 之前施加)。
            ``"Smooth"`` (幅度谱平滑, 默认**不开启**) 用于抑制伪峰、让
            ``maximum`` 也能给出贴合的边界, 代价是噪声下抖动变大。
        mirror : bool
            ``True`` (默认): 频域滤波在**端点镜像延拓**后的信号上进行
            (``Utils.mirror`` 的 ``mod="mirror"``, 每端镜像 ``N//2`` 个样本,
            长度 ``2N``, 轴长与原信号共享 ``0…fs/2``), 逆变换后取中间 ``N`` 点 ——
            同 ewtpy 的 ``fMirr`` 处理, 抑制频域滤波的环形卷积边界效应。
            ``False``: 直接在原信号的单边谱上滤波 (等价于对该长度信号做循环
            卷积)。两种取值下**边界检测都在原信号谱上**做, 故 ``boundaries``
            相同; 滤波结果与代价不同 (镜像多一次 ``rfft`` 且轴长翻倍)。
        dtype : type | np.dtype | None
            返回 IMF 的工作 dtype。None 保留输入精度; int/bool/f16 输入会被提升
            到 float64。
        **kwargs
            分支参数 (未传入即用默认值, 见 ``PRE_DEAL_DEFAULTS`` /
            ``BOUNDARY_DEFAULTS``; 字符串取值一律经 ``Check.validate_mod``
            归一成对应的 ``EWT_*_MODS`` 成员):
            时域 ``no_dc_type`` (``EWT_NO_DC_TYPE_MODS``) / ``no_dc_bp`` /
            ``no_trend_win`` / ``window_kind`` (``EWT_WINDOW_KIND_MODS``);
            频域 ``slepian_nw`` / ``slepian_k`` / ``slepian_mod``
            (``EWT_SLEPIAN_MODS``) / ``slepian_max_samples`` /
            ``smooth_kind`` (``EWT_SMOOTH_KIND_MODS``) / ``smooth_width``;
            边界 ``peak_distance`` /
            ``max_min_fallback`` (``EWT_MAX_MIN_FALLBACK_MODS``) /
            ``scale_space_scales`` /
            ``scale_space_sigma_min`` / ``scale_space_sigma_max`` /
            ``scale_space_otsu_bins`` / ``scale_space_min_persist`` /
            ``envelope_spline`` (``EWT_ENVELOPE_SPLINE_MODS``) /
            ``envelope_noise_pct`` / ``envelope_snr_db`` /
            ``envelope_prominence``; 以及库级 ``safe`` / ``copy``.
            未识别的键报 ``TypeError``。
        """
        self.num_imfs: int = num_imfs
        self.band_factor: float | EWT_BAND_FACTOR_MODS_ARGS | EWT_BAND_FACTOR_MODS = band_factor
        self.boundary_mod: EWT_BOUNDARY_MODS_ARGS | EWT_BOUNDARY_MODS = boundary_mod
        self.mirror: bool = mirror
        self.dtype: type | np.dtype | None = dtype

        unknown = set(kwargs) - _KWARGS_ALLOWED
        if unknown:
            raise TypeError(f"unexpected keyword arguments: {unknown}")

        self.safe = kwargs.get("safe", "unknown")
        self.copy: bool = kwargs.get("copy", True)

        # 属性名用 pre_deal_mods, 以免遮蔽同名的公共方法 ``pre_deal()``。
        self.pre_deal_mods = _canon_pre_deal_mods(pre_deal)
        self.pre_deal_time = tuple(m for m in self.pre_deal_mods if m in PRE_DEAL_MOD_TIME)
        self.pre_deal_freq = tuple(m for m in self.pre_deal_mods if m in PRE_DEAL_MOD_FREQ)

        # --- 分支参数: kwargs 逐名覆盖默认值 -------------------------------- #
        self.pre_deal_kwargs = {k: kwargs.get(k, v) for k, v in PRE_DEAL_DEFAULTS.items()}
        self.boundary_kwargs = {k: kwargs.get(k, v) for k, v in BOUNDARY_DEFAULTS.items()}
        self.kwargs = kwargs

        # --- 运行期状态: 每次 decompose 开头复位 (缓冲按形状跨调用复用) ----- #
        self.frequency = None
        self.empirical_wave = None
        self.modal = None
        self._trend = None              # no-trend 分支构造出的趋势 (保留进 result)
        self._slepian_degraded = False  # Slepian 分支是否因规模上限而降级
        self._band_factor_effective = None   # "auto" 解析后的实际 γ
        self.fs: float = 1.0                 # 最近一次 decompose 的采样频率

        self.param_check()

    def param_check(self) -> None:
        """校验构造参数 (非法值在实例化时立即报错, 不留到 decompose)。"""
        if not isinstance(self.num_imfs, int) or isinstance(self.num_imfs, bool) or self.num_imfs < 1:
            raise ValueError(f"num_imfs must be an int >= 1, got {self.num_imfs!r}")

        # band_factor 是 "数字或字符串" 的联合: 数值原样保留为 float, 只有字符串
        # 形态 ("auto") 走 Enum; 实例上最终只可能是 float 或 EWT_BAND_FACTOR_MODS.Auto。
        if isinstance(self.band_factor, str):
            self.band_factor = Check.validate_mod(self.band_factor, EWT_BAND_FACTOR_MODS)
        elif not isinstance(self.band_factor, EWT_BAND_FACTOR_MODS):
            if not (0.0 < float(self.band_factor) < 1.0):
                raise ValueError(f"band_factor must be in (0, 1), got {self.band_factor!r}")
            self.band_factor = float(self.band_factor)      # 归一成 float (np.float 等)

        self.boundary_mod = Check.validate_mod(self.boundary_mod, EWT_BOUNDARY_MODS)

        if not isinstance(self.mirror, bool):
            raise ValueError(f"mirror must be a bool, got {self.mirror!r}")

        if self.dtype is not None and np.dtype(self.dtype).kind != "f":
            raise ValueError("the `dtype` must be a float numpy dtype or None.")

        if not isinstance(self.copy, bool):
            raise TypeError("the `copy` must be bool")

        self.safe = Check.validate_safe(self.safe)

        self._validate_branch_kwargs()

    def _validate_branch_kwargs(self) -> None:
        """校验分支参数取值, 并把字符串归一成 ``EWT_*_MODS`` 成员。

        非法值立即报错 (不留到 decompose); 归一后就地写回
        ``self.pre_deal_kwargs`` / ``self.boundary_kwargs``, 于是内部只会见到
        Enum 成员 (或数值), 不再出现字符串分支名。
        """
        kw, bw = self.pre_deal_kwargs, self.boundary_kwargs

        kw["no_dc_type"] = Check.validate_mod(kw["no_dc_type"], EWT_NO_DC_TYPE_MODS)
        if kw["no_trend_win"] is not None and int(kw["no_trend_win"]) < 1:
            raise ValueError(f"no_trend_win must be >= 1 or None, got {kw['no_trend_win']!r}")
        kw["window_kind"] = _canon_window_kind(kw["window_kind"])
        if float(kw["slepian_nw"]) <= 0.0:
            raise ValueError(f"slepian_nw must be > 0, got {kw['slepian_nw']!r}")
        if kw["slepian_k"] is not None and int(kw["slepian_k"]) < 1:
            raise ValueError(f"slepian_k must be >= 1 or None, got {kw['slepian_k']!r}")
        if kw["slepian_mod"] is not None:       # None = ConstDefine.SLEPIAN_BACKEND
            kw["slepian_mod"] = Check.validate_mod(kw["slepian_mod"], EWT_SLEPIAN_MODS)
        kw["smooth_kind"] = _canon_mod_lower(kw["smooth_kind"], EWT_SMOOTH_KIND_MODS)
        if float(kw["smooth_width"]) < 1.0:
            raise ValueError(f"smooth_width must be >= 1, got {kw['smooth_width']!r}")

        bw["max_min_fallback"] = Check.validate_mod(
            bw["max_min_fallback"], EWT_MAX_MIN_FALLBACK_MODS
        )
        if int(bw["scale_space_scales"]) < 2:
            raise ValueError(f"scale_space_scales must be >= 2, got {bw['scale_space_scales']!r}")
        if float(bw["scale_space_sigma_min"]) <= 0.0 or float(bw["scale_space_sigma_max"]) <= 0.0:
            raise ValueError("scale_space sigma bounds must be > 0")
        if int(bw["scale_space_otsu_bins"]) < 2:
            raise ValueError(f"scale_space_otsu_bins must be >= 2, got {bw['scale_space_otsu_bins']!r}")
        bw["envelope_spline"] = Check.validate_mod(
            bw["envelope_spline"], EWT_ENVELOPE_SPLINE_MODS
        )
        if not (0.0 <= float(bw["envelope_noise_pct"]) <= 100.0):
            raise ValueError(f"envelope_noise_pct must be in [0, 100], got {bw['envelope_noise_pct']!r}")
        if float(bw["envelope_snr_db"]) < 0.0:
            raise ValueError(f"envelope_snr_db must be >= 0, got {bw['envelope_snr_db']!r}")
        if not (0.0 <= float(bw["envelope_prominence"]) <= 1.0):
            raise ValueError(
                f"envelope_prominence must be in [0, 1], got {bw['envelope_prominence']!r}"
            )

    def config_generate(self) -> EWTConfig:
        """本次运行的有效参数快照 (``band_factor`` 是实际使用的 γ)。"""
        band_factor = self.band_factor \
            if self._band_factor_effective is None else float(self._band_factor_effective)

        return EWTConfig(
            num_imfs=self.num_imfs,
            band_factor=band_factor,
            boundary_mod=self.boundary_mod,
            pre_deal=tuple(self.pre_deal_mods),
            mirror=self.mirror,
            fs=float(self.fs),
            dtype=self.dtype,
            pre_deal_kwargs=dict(self.pre_deal_kwargs),
            boundary_kwargs=dict(self.boundary_kwargs),
        )

    def decompose(self, S, T=None, fs: float = 1.0, output_memmap: bool = False) -> DecompositionResult:
        """
        把信号分解为 ``num_imfs`` 个带限模态。

        Parameters
        ----------
        S : array-like
            信号 (入口 ``Check_Time_and_Signal`` 负责应用全局内存策略)。
        T : array-like, optional
            时间轴; 只校验长度 / 重复 / 单调性, 算法本身不使用。
        fs : float
            采样频率, 只用于把频率轴与报告的边界表示成 Hz (变换本身与尺度无关)。

        Returns
        -------
        DecompositionResult
            ``IMFs``: ``(num_imfs, N)`` 实值频带, 按频率升序, 工作 dtype 由
            ``dtype`` 参数决定 (缺省即输入精度); ``Res`` = ``S − ΣIMFs`` (故
            ``reconstruct()`` 能精确还原 ``S``); ``info`` 含 ``mfb`` (滤波器组,
            ``(num_imfs, N//2+1)``; 分块路径下为 None, 见下)、``boundaries``
            (``num_imfs+1`` 个, Hz)、``boundary_mod`` (字符串形态, 即 Enum 的
            ``.value``)、``residual_ratio`` (``‖Res‖ / ‖S‖``)、``chunked``、
            ``chunk_size`` 与 ``mirror``。

        镜像 (``mirror`` 参数)
        ---------------------
        边界检测一律在原信号单边谱上做, 故 ``boundaries`` 与 ``mirror`` 取值无关;
        ``mirror=True`` (默认) 时滤波改在端点镜像延拓信号 (每端 ``N//2`` 个样本,
        长度 ``2N``) 上做, 逆变换后取中间 ``N`` 点, 以抑制频域滤波的环形卷积
        边界效应 (ewtpy 的 ``fMirr`` 同法); ``mirror=False`` 时即对原长度信号做
        循环卷积。

        滤波的两条路径
        -------------
        滤波器组自身的体积 (``num_imfs × (N//2+1) × 8`` 字节) 达到
        ``Define.BIG_ARRAY`` 时不再一次性物化整张滤波器组, 改为分块 +
        ``Operator`` 算子逐块求值, 并逐模态落定 ——
        每模态只保留一条单边带谱 ``(N//2+1,)`` 复数组, 工作集与 ``num_imfs``
        解耦 (``info["chunked"]`` 为 True, ``info["mfb"]`` 为 None); 未达阈值
        时走一次性物化的向量化路径, 两条路径逐元素结果一致。
        """
        # 形参校验先于任何计算: 传错类型应立即失败, 而不是跑完整个分解再报错
        check_output_memmap(output_memmap)
        S, N = Check_Time_and_Signal(
            S, T=T, ndim={1}, method=self.name, dtype=self.dtype,
            need_T=False, safe=self.safe, copy=self.copy,
        )

        # int/bool 输入无法承载滤波结果, f16 会让算子/样条出非有限值: 统一抬到 f64
        if not (S.dtype.kind == "f" and S.dtype.itemsize >= 4):
            S = Check.to_dtype(S, np.dtype(np.float64))

        # 每次调用都从干净状态开始 (缓冲 self.modal / self.empirical_wave 按形状复用)
        self._trend = None
        self._slepian_degraded = False
        self._band_factor_effective = None
        self.fs = float(fs)

        S_in = S                                # 原始输入: Res / reconstruct 的基准

        # --- 时域预处理: Check_Time_and_Signal 之后立即施加 ----------------- #
        S = self._time_domain_pre_deal(S)

        # 边界检测始终在**预处理后信号的原始谱**上做 (与 ewtpy 一致: 镜像只作用
        # 于滤波阶段, 不改变 boundaries)。
        F_orig = np.fft.rfft(S)                         # (N//2+1,) 单边谱
        freq_orig = np.fft.rfftfreq(S.size, 1.0 / fs)   # 0…fs/2, 含 Nyquist
        self.frequency = freq_orig

        # --- 频域预处理: fft 之后立即施加 (产出边界检测用的幅度谱) ---------- #
        amp = self._freq_domain_pre_deal(S, F_orig)

        boundary = self.boundary(amp, freq_orig, mod=self.boundary_mod)
        n_bands = int(boundary.size) - 1        # "envelope" 分支可少于 num_imfs

        # mirror=True: 滤波在端点镜像延拓信号上进行 (Utils.mirror 的
        # mod="mirror", 每端 N//2 个样本 ⇒ 长度 2N), 逆变换后取中间 N 点 ——
        # 同 ewtpy 的 fMirr 处理, 抑制频域滤波的环形卷积边界效应。镜像轴与原轴
        # 共享 0…fs/2 频率范围, 故 boundary 通用。
        if self.mirror:
            left = N // 2                       # 每端镜像的样本数 (与 Utils.mirror 一致)
            f_mirr = mirror(S, mod="mirror", nbsym=left, safe="safe")
            axis_len = int(f_mirr.size)
            F = np.fft.rfft(f_mirr)
            frequency = np.fft.rfftfreq(axis_len, 1.0 / fs)
        else:
            axis_len = N
            F = F_orig
            frequency = freq_orig
            left = 0

        n_bins = axis_len // 2 + 1
        if n_bins < n_bands + 1:
            raise ValueError(
                f"{self.name}: signal length {N} is too short for "
                f"{n_bands} bands (需要 axis_len//2+1 >= {n_bands + 1})"
            )

        # γ: 显式 float 原样使用; "auto" 按边界间距反推 (下称"自动 gamma")。
        band_factor, bf_auto = self._resolve_band_factor(boundary, frequency)
        self._band_factor_effective = band_factor   # 落进 config_generate()

        # 最高带的滤波上边界: 算子把边界 w 的过渡带放在 [(1−γ)w, (1+γ)w], 若最高带
        # 仍以 Nyquist 为上边界, 它就会在 Nyquist 处再放一条过渡带 —— 而 Nyquist
        # 之上没有任何频带能补偿这半条过渡带 (单边谱里不存在), 于是 Σ H² 在该处
        # 掉到 0.5 并且频谱高端被系统性压低。改为 w1 = Nyquist/(1−γ), 过渡带正好
        # 从 Nyquist 起跳 ⇒ 最高带一直通到谱尾, 全轴满足紧框架 Σ H² ≡ 1。
        # (只改滤波用的这份副本; info["boundaries"] 仍是检测到的真实边界。)
        filter_boundary = np.array(boundary, dtype=np.float64, copy=True)
        filter_boundary[-1] = float(frequency[-1]) / (1.0 - band_factor)

        # 跨调用长度守卫: 模态缓冲按本次 (n_bands, N) 复用/重建。
        if self.modal is None or self.modal.shape != (n_bands, N):
            self.modal = np.zeros((n_bands, N), dtype=DEFAULT_NUMPY_TYPE)

        # 大数组判据: 滤波器组体积 (n_bands × n_bins, float64) 达到 Define.BIG_ARRAY
        # 时不物化整张滤波器组, 改为分块 + 算子逐块求值, 并逐模态落定 (工作集与
        # 频带数解耦, 只留一条带谱)。
        filter_bytes = n_bands * n_bins * np.dtype(np.float64).itemsize
        chunked = filter_bytes >= BIG_ARRAY

        if chunked:
            # 块长按"目标单块工作集"反推: 块内除算子输出外还有
            # _OPERATOR_TMP_ELEMS 个 float64 临时量, 故
            # 块长 = EWT_CHUNK_WORK_BYTES / (临时量数 × 8 字节), 再夹到 [1, n_bins]。
            elem_bytes = _OPERATOR_TMP_ELEMS * np.dtype(np.float64).itemsize
            chunk_size = max(1, min(n_bins, EWT_CHUNK_WORK_BYTES // elem_bytes))
            self.empirical_wave = None              # 大数组下不保留整张滤波器组
            for row in range(n_bands):
                band_hat = np.empty(n_bins, dtype=np.complex128)
                with np.errstate(divide="ignore", invalid="ignore"):
                    for start in range(0, n_bins, chunk_size):
                        stop = min(start + chunk_size, n_bins)
                        band_hat[start:stop] = empirical_wavelet_arr(
                            frequency[start:stop],
                            band_factor=band_factor,
                            w0=filter_boundary[row],
                            w1=filter_boundary[row + 1],
                        ) * F[start:stop]
                self.modal[row, :] = np.fft.irfft(band_hat, n=axis_len)[left:left + N]
        else:
            chunk_size = None
            if self.empirical_wave is None or self.empirical_wave.shape != (n_bands, n_bins):
                self.empirical_wave = np.zeros((n_bands, n_bins), dtype=DEFAULT_NUMPY_TYPE)

            # 逐带求传递函数: 滤波器直接建立在频率轴上 (不是下标轴), 且 |x| 使
            # 单边谱天然对称, 无需再补负频镜像。
            # 算子内部用 np.select 一次性展开所有候选分支 (含退化区间 0/0), 会产生
            # 被丢弃的除零/无效值告警; 这里就地屏蔽 (结果不受影响, 只是消除噪声)。
            with np.errstate(divide="ignore", invalid="ignore"):
                for row in range(n_bands):
                    self.empirical_wave[row, :] = empirical_wavelet_arr(
                        frequency,
                        band_factor=band_factor,
                        w0=filter_boundary[row],
                        w1=filter_boundary[row + 1],
                    )

            freq_modal = self.empirical_wave * F
            full = np.fft.irfft(freq_modal, n=axis_len, axis=1)
            # 镜像路径下取中间 N 点 (复制成连续数组, 避免留下切片视图)
            self.modal = np.ascontiguousarray(full[:, left:left + N])

        # 输出 dtype 跟随工作精度 (f32 输入 ⇒ f32 模态), 内部一律按 f64 计算。
        IMFs = np.asarray(self.modal, dtype=S.dtype)

        # 残差以**原始输入**为基准: 预处理(去趋势/加窗/镜像)带来的差异一并落入
        # Res, 于是 reconstruct() 始终等于用户传入的 S; no-trend 的趋势也因此
        # 保留在结果中 (Res 内 + info["trend"] 可单独取用)。
        Res = S_in - IMFs.sum(axis=0)

        norm_s = float(np.linalg.norm(S_in))
        residual_ratio = float(np.linalg.norm(Res)) / norm_s if norm_s > 0.0 else 0.0

        # 另给一个"分解侧"残差比: 相对**预处理后**信号。两个比值配合可区分
        # "预处理移除了多少" 与 "滤波器组没覆盖多少"。
        norm_pre = float(np.linalg.norm(S))
        resid_pre = S - IMFs.sum(axis=0)
        residual_ratio_pre = float(np.linalg.norm(resid_pre)) / norm_pre if norm_pre > 0.0 else 0.0

        return finalize_output(
            DecompositionResult(
                IMFs,
                Res,
                {
                    "mfb": self.empirical_wave,
                    "boundaries": boundary,
                    # info 里给字符串形态 (Enum 的 .value): 调用方与序列化只认字符串。
                    "boundary_mod": self.boundary_mod.value,
                    "n_bands": n_bands,
                    "band_factor": band_factor,         # 实际使用的 γ (auto 时已反推)
                    "band_factor_auto": bf_auto,
                    "pre_deal": tuple(self.pre_deal_mods),
                    "pre_deal_kwargs": dict(self.pre_deal_kwargs),
                    "boundary_kwargs": dict(self.boundary_kwargs),
                    "trend": self._trend,
                    "slepian_degraded": self._slepian_degraded,
                    "residual_ratio": residual_ratio,
                    "residual_ratio_preprocessed": residual_ratio_pre,
                    "chunked": chunked,
                    "chunk_size": chunk_size,
                    "mirror": self.mirror,
                },
                self.config_generate(),
            ),
            output_memmap,
        )

    # ------------------------------------------------------------------ #
    # 预处理: 时域 (check 之后) / 频域 (fft 之后)
    # ------------------------------------------------------------------ #
    def _time_domain_pre_deal(self, S: np.ndarray, mods=None, **kwargs) -> np.ndarray:
        """
        时域预处理: 在 ``Check_Time_and_Signal`` 之后立即作用于信号。

        分支 (canonical 顺序 ``no-dc`` → ``no-trend`` → ``window``):

        * ``no-dc``    —— 去除趋势 (``scipy.signal.detrend``, ``no_dc_type``
          默认 ``"linear"``, ``no_dc_bp`` 默认 0); 去掉的趋势**不保留**。
        * ``no-trend`` —— 移动平均构造趋势并减去 (``no_trend_win`` 默认自动:
          约 ``N//20`` 的奇数, 端点复制填充); 趋势保留在
          ``info["trend"]``, 并随 ``Res`` 一起回到结果中
          (``Res = S_原始 − ΣIMFs`` ⇒ ``reconstruct()`` 仍等于原始输入)。
        * ``window``   —— 加窗 (``window_kind`` ∈ ``EWT_WINDOW_KIND_MODS``;
          ``np.hanning`` / ``np.hamming``, 对称窗); 加窗损失同样并入 ``Res``。

        Parameters
        ----------
        S : np.ndarray
            时域信号 (check 之后, mirror 之前)。
        mods : EWT_PRE_DEAL_MODS_ARGS | EWT_PRE_DEAL_MODS | Sequence[...] | None
            显式指定分支 (按 canonical 顺序重排, 字符串经 ``Check.validate_mod``
            归一); None 时用 ``self.pre_deal_time``。
        **kwargs
            逐名覆盖该分支参数 (未给出用 ``self.pre_deal_kwargs``)。

        Returns
        -------
        np.ndarray
            处理后的信号 (未选任何分支时原样返回, 不复制)。
        """
        kw = dict(self.pre_deal_kwargs)
        kw.update(kwargs)
        if mods is None:
            mods = self.pre_deal_time
        else:
            mods = tuple(m for m in PRE_DEAL_MOD_TIME if m in _canon_pre_deal_mods(mods))

        self._trend = None
        for mod in mods:
            match Check.validate_mod(mod, EWT_PRE_DEAL_MODS):
                case EWT_PRE_DEAL_MODS.NoDC:
                    sp = Cache.import_module(
                        "scipy.signal",
                        description="scipy.signal: detrend (供 EWT no-dc 分支使用)",
                    )
                    if sp is None:
                        raise ImportError(
                            "EWT pre_deal='no-dc' needs scipy: python -m pip install scipy"
                        )
                    detrend_type = Check.validate_mod(kw["no_dc_type"], EWT_NO_DC_TYPE_MODS)
                    S = np.asarray(
                        sp.detrend(S, type=detrend_type.value, bp=int(kw["no_dc_bp"])),
                        dtype=DEFAULT_NUMPY_TYPE,
                    )
                case EWT_PRE_DEAL_MODS.NoTrend:
                    trend = _moving_average(S, kw["no_trend_win"])
                    S = S - trend
                    self._trend = trend
                case EWT_PRE_DEAL_MODS.Window:
                    S = S * _window_vec(S.size, kw["window_kind"])
        return S

    def _freq_domain_pre_deal(self, S: np.ndarray, F: np.ndarray, mods=None, **kwargs) -> np.ndarray:
        """
        频域预处理: 在 ``fft`` 之后立即作用于谱, 产出**边界检测用的幅度谱**。

        分支 (按 ``PRE_DEAL_MOD_FREQ`` 的顺序施加: ``Slepian-Optimize`` → ``Smooth``):

        * ``Slepian-Optimize`` —— Slepian 多锥谱估计: 经 ``Utils.slepian``
          统一入口取 ``(K, N)`` 的 Slepian 序列 (``slepian_nw`` 默认 3.0,
          ``slepian_k`` 默认 ``2*NW-1``), 各锥按其集中比加权平均:
          ``amp = sqrt(Σ_k ratio_k·|rfft(S·v_k)|² / Σk ratio_k)``。
          后端由 ``slepian_mod`` 决定 (None = ``ConstDefine.SLEPIAN_BACKEND``;
          显式取值见 ``EWT_SLEPIAN_MODS``);
          **小 N (NW >= N/2, 默认参数下 N <= 6) 由入口自动换后端并 warning**,
          本类不再直接调 scipy。
          ``N > slepian_max_samples`` (默认 ``ConstDefine.EWT_SLEPIAN_MAX_SAMPLES``
          = 32768) 时降级为普通 ``|rfft|`` 并在 ``info["slepian_degraded"]`` 标记 ——
          这是**时间预算**取舍 (dpss 在 N=262144 时约 290 ms/次), 传
          ``slepian_max_samples=None`` 可取消该上限。
        * ``Smooth`` —— 幅度谱平滑 (与 ewtpy 的 ``reg``+``lengthFilter`` 同法):
          ``smooth_kind="box"`` 走端点复制的移动平均, ``"gaussian"`` 走高斯核;
          窗宽 ``smooth_width`` 以**频点数**计 (边界检测轴宽 ``N//2+1``, 默认 5 ⇒
          ``2·fs/N·5`` Hz), box 的偶数宽会被向上取到奇数以保证窗居中。
          作用是抑制谱上由噪声/边带产生的伪峰, 使 ``maximum`` 这类"直接取前 K
          大峰"的策略同样能得到贴近真实分量的边界 —— 代价是噪声下边界抖动变大
          (实测 multitone 稳定性 0.67 → 2.4 Hz), 故**默认不开启**。

        滤波仍使用原始复数谱 ``F`` —— 与 ewtpy 一致, 本函数只改变边界检测的输入。

        Returns
        -------
        np.ndarray
            用于 :meth:`boundary` 的幅度谱 (与 ``F`` 同长度)。
        """
        kw = dict(self.pre_deal_kwargs)
        kw.update(kwargs)
        if mods is None:
            mods = self.pre_deal_freq
        else:
            mods = tuple(m for m in PRE_DEAL_MOD_FREQ if m in _canon_pre_deal_mods(mods))

        amp = np.abs(F)
        self._slepian_degraded = False
        for mod in mods:
            match Check.validate_mod(mod, EWT_PRE_DEAL_MODS):
                case EWT_PRE_DEAL_MODS.SlepianOptimize:
                    limit = kw["slepian_max_samples"]
                    if limit is not None and S.size > int(limit):
                        self._slepian_degraded = True
                    else:
                        amp = self._slepian_multitaper(S, kw["slepian_nw"], kw["slepian_k"],
                                                       mod=kw["slepian_mod"])
                case EWT_PRE_DEAL_MODS.Smooth:
                    width = kw["smooth_width"]
                    match _canon_mod_lower(kw["smooth_kind"], EWT_SMOOTH_KIND_MODS):
                        case EWT_SMOOTH_KIND_MODS.Box:
                            amp = _moving_average(amp, int(width) | 1)
                        case EWT_SMOOTH_KIND_MODS.Gaussian:
                            amp = _gaussian_smooth(amp, float(width))
        return amp

    def _slepian_multitaper(self, S: np.ndarray, nw, k, mod=None) -> np.ndarray:
        """
        Slepian 多锥幅度谱估计 (各锥按集中比加权); 返回与 ``rfft(S)`` 同长度。

        走 ``Utils.slepian`` 统一入口: 后端选择/小 N 自动换后端/进程内缓存
        都由入口负责 (本方法不再直接依赖 scipy.signal)。
        """
        n = int(S.size)
        kk = int(2 * float(nw) - 1) if k is None else int(k)
        kk = max(1, min(kk, n))
        # None = 用 ConstDefine.SLEPIAN_BACKEND; 入口把这一语义收在 "auto" 上。
        # mod 是 EWT_SLEPIAN_MODS 成员 (None 除外), 入口只认字符串 ⇒ 传 .value。
        backend = None if mod is None else Check.validate_mod(mod, EWT_SLEPIAN_MODS)
        seqs, ratios = slepian(
            n, float(nw), kk, mod="auto" if backend is None else backend.value,
            return_ratios=True, norm=2,
        )
        w = np.maximum(np.asarray(ratios, dtype=np.float64), 0.0)
        spec = np.abs(np.fft.rfft(S[None, :] * seqs, axis=1)) ** 2
        return np.sqrt(np.maximum((spec * w[:, None]).sum(axis=0) / max(float(w.sum()), 1e-300), 0.0))

    def pre_deal(self, S: np.ndarray, mod="None", **kwargs) -> np.ndarray:
        """
        时域预处理便捷入口: 显式调用 :meth:`_time_domain_pre_deal`。

        频域分支 (``Slepian-Optimize`` / ``Smooth``) 需要频谱, 只能在
        ``__init__`` 的 ``pre_deal`` 中选择、由 ``decompose`` 在 ``fft`` 之后施加。

        ``mod`` 可以是单个名字或名字序列 (大小写不敏感, ``"None"`` = 不处理),
        也可以直接给 ``EWT_PRE_DEAL_MODS`` 成员。
        """
        mods = _canon_pre_deal_mods(mod)
        freq_only = [m for m in mods if m in PRE_DEAL_MOD_FREQ]
        if freq_only:
            raise ValueError(
                f"{freq_only} 是频域分支, 需要频谱: 请在 __init__ 的 pre_deal 中选择"
            )
        return self._time_domain_pre_deal(S, mods=mods, **kwargs)

    # ------------------------------------------------------------------ #
    # 边界策略集成
    # ------------------------------------------------------------------ #
    def boundary(self, amp: np.ndarray, frequency: np.ndarray, mod=None, **kwargs) -> np.ndarray:
        """
        边界策略集成入口: 由幅度谱产出边界数组 (Hz)。

        Parameters
        ----------
        amp : np.ndarray
            幅度谱 (长度与 ``frequency`` 一致; 通常来自
            :meth:`_freq_domain_pre_deal`)。
        frequency : np.ndarray
            频率轴 (Hz, ``np.fft.rfftfreq``), 0…fs/2。
        mod : EWT_BOUNDARY_MODS_ARGS | EWT_BOUNDARY_MODS | None
            策略名 (字符串经 ``Check.validate_mod`` 归一成 ``EWT_BOUNDARY_MODS``);
            None 时用 ``self.boundary_mod``。
        **kwargs
            逐名覆盖该策略参数 (未给出用 ``self.boundary_kwargs``)。

        Returns
        -------
        np.ndarray
            边界数组 (Hz): 首元素为 ``frequency[0]``、末元素为
            ``frequency[-1]``, 中间为严格递增的内部边界。长度 = **实际频带数 + 1**;
            ``"envelope"`` 由显著峰数自动定模式数 (上限 ``num_imfs``), 其余策略
            固定返回 ``num_imfs + 1`` 个边界。

        四套策略
        --------
        ``maximum``      局部极大值法: 取幅值最大的 ``num_imfs`` 个峰, 内部边界取
                         相邻峰中点 (峰不足时退化为均匀分割)。
        ``max-min``      极大—极小型: 相邻(前 K 大)峰之间取局部极小作边界, 更符合
                         "谷底分割"; 段内无极小时按 ``max_min_fallback`` 回退到
                         中点或几何平均。
        ``scale-space``  尺度空间法: 多尺度高斯平滑幅谱, 追踪"持久极小", 再用
                         Otsu 筛显著边界; 显著边界不足时退回 ``maximum``。
        ``envelope``     包络/阈值法 (增强 EWT): 幅谱样条包络 → 按 SNR 定硬阈值
                         修平 → 检峰检谷, 模式数由显著峰数自动确定 (适合轴承/风机
                         等故障信号)。
        """
        mod = self.boundary_mod if mod is None else Check.validate_mod(mod, EWT_BOUNDARY_MODS)

        kw = dict(self.boundary_kwargs)
        kw.update(kwargs)
        amp = np.asarray(amp, dtype=np.float64)
        frequency = np.asarray(frequency, dtype=np.float64)
        f_lo, f_hi = float(frequency[0]), float(frequency[-1])

        match mod:
            case EWT_BOUNDARY_MODS.Maximum:
                interior = self._boundary_maximum(amp, frequency, **kw)
            case EWT_BOUNDARY_MODS.MaxMin:
                interior = self._boundary_max_min(amp, frequency, **kw)
            case EWT_BOUNDARY_MODS.ScaleSpace:
                interior = self._boundary_scale_space(amp, frequency, **kw)
            case EWT_BOUNDARY_MODS.Envelope:
                interior = self._boundary_envelope(amp, frequency, **kw)
            case _:     # pragma: no cover - Check.validate_mod 已挡住其它取值
                raise ValueError(f"the mod of {mod} is not supported.")

        interior = np.asarray(interior, dtype=np.float64).ravel()
        interior = interior[np.isfinite(interior)]
        interior = np.unique(interior)                          # 去重 + 升序
        interior = interior[(interior > f_lo) & (interior < f_hi)]
        return np.concatenate(([f_lo], interior, [f_hi]))

    def _peaks(self, amp: np.ndarray, **kw) -> np.ndarray:
        """谱峰检测 (``Utils.find_peaks``; ``peak_distance`` 透传); 返回内部下标。"""
        fkw = {"safe": "safe"}
        if kw.get("peak_distance") is not None:
            fkw["distance"] = int(kw["peak_distance"])
        peaks, _ = find_peaks(amp, **fkw)
        idx = np.asarray(peaks, dtype=np.int64)
        return idx[(idx > 0) & (idx < amp.size)]

    # ------------------------------------------------------------------ #
    # 过渡带宽度 γ
    # ------------------------------------------------------------------ #
    def _resolve_band_factor(self, boundary, frequency) -> tuple[float, bool]:
        """
        解析 ``band_factor`` -> ``(γ, 是否自动)``。

        显式 float 原样返回。``EWT_BAND_FACTOR_MODS.Auto``
        (即字符串 ``"auto"``, 校验后已归一成 Enum) 时按**边界间距反推**:
        算子 (:func:`Operator.Empirical_wavelet.empirical_wavelet_arr`) 把边界
        ``w`` 的过渡带放在 ``[(1−γ)w, (1+γ)w]``, 于是相邻边界 ``w_i < w_{i+1}``
        的过渡带不交叠要求

        .. math:: (1+γ)w_i \\le (1-γ)w_{i+1}
                  \\iff γ \\le \\frac{w_{i+1}-w_i}{w_{i+1}+w_i}

        上端边界还取同一形式的约束 ``(Nyquist−w_last)/(Nyquist+w_last)`` (即
        ewtpy 的 ``(π−w_last)/(π+w_last)`` 项; 这些比值与频率单位无关)。

        .. note::
           最高带的**滤波**上边界取 ``Nyquist/(1−γ)`` (见 ``decompose``),
           因此上端约束并不承担防交叠, 而是用来把 γ 压小 —— γ 越小,
           过渡带越窄, 幅度带和 ``Σ H`` 偏离 1 的频段就越少, 带和误差也越小
           (实测: 只用内部约束的 γ 在 white/spike 上让带和误差从 0.108/0.127
           升到 0.157/0.160)。

        取所有约束的最小值再乘 ``(1−1/n_bins)`` 作为"严格小于"的安全余量, 得到
        **不交叠的最大 γ** —— 此时滤波器组满足紧框架 ``Σ_i H_i² ≡ 1``, 幅度带和
        误差最小; 而 γ 偏大 (边界聚拢时取 0.15 这种量级) 会让 2–3 条带的过渡带
        叠加, 同一频段被重复计入 (实测 ``Σ H²`` 最高 3.15)。

        无内部边界 (单频带) 时没有交叠约束可言, 退回固定 ``0.15``。
        """
        if self.band_factor is not EWT_BAND_FACTOR_MODS.Auto:
            return float(self.band_factor), False

        nyq = float(frequency[-1])
        interior = np.asarray(boundary, dtype=np.float64)
        interior = interior[(interior > 0.0) & (interior < nyq)]
        if interior.size == 0:
            return 0.15, True                     # 单频带: 无交叠约束, 退回固定 0.15

        gamma = 1.0
        if interior.size >= 2:
            lo, hi = interior[:-1], interior[1:]
            gamma = min(gamma, float(np.min((hi - lo) / (hi + lo))))
        w_last = float(interior[-1])
        gamma = min(gamma, (nyq - w_last) / (nyq + w_last))
        gamma *= 1.0 - 1.0 / max(int(frequency.size), 2)
        return float(np.clip(gamma, 1e-9, 0.99)), True

    def _boundary_maximum(self, amp, frequency, **kw) -> np.ndarray:
        """局部极大值法: 前 ``num_imfs`` 大峰 + 相邻峰中点 (不足则均匀分割)。"""
        n_bands = self.num_imfs
        f_lo, f_hi = float(frequency[0]), float(frequency[-1])
        peaks = self._peaks(amp, **kw)
        if peaks.size >= n_bands:
            top = np.sort(peaks[np.argsort(amp[peaks])[::-1][:n_bands]])
            return (frequency[top[:-1]] + frequency[top[1:]]) / 2.0
        return np.linspace(f_lo, f_hi, n_bands + 1)[1:-1]

    def _boundary_max_min(self, amp, frequency, **kw) -> np.ndarray:
        """极大—极小型: 相邻(前 K 大)峰之间的局部极小作边界; 无极小时按
        ``max_min_fallback`` 回退 (中点 / 几何平均)。"""
        n_bands = self.num_imfs
        peaks = self._peaks(amp, **kw)
        if peaks.size < n_bands:
            return self._boundary_maximum(amp, frequency, **kw)   # 峰不足: 同 maximum 退路
        top = np.sort(peaks[np.argsort(amp[peaks])[::-1][:n_bands]])
        fb = kw.get("max_min_fallback", "midpoint")
        return np.array(
            [_valley_between(amp, top[i], top[i + 1], frequency, fb) for i in range(n_bands - 1)]
        )

    def _boundary_scale_space(self, amp, frequency, **kw) -> np.ndarray:
        """尺度空间法: 多尺度高斯平滑 → 追踪"持久极小" → Otsu 筛显著边界;
        显著边界不足 ``num_imfs-1`` 条时退回 ``maximum``。"""
        n_bands = self.num_imfs
        scales = max(2, int(kw["scale_space_scales"]))
        s_min = max(float(kw["scale_space_sigma_min"]), 1e-3)
        s_max = max(float(kw["scale_space_sigma_max"]), s_min)
        sigmas = np.geomspace(s_min, s_max, scales)

        cand = _local_minima(amp)
        if cand.size == 0:
            return self._boundary_maximum(amp, frequency, **kw)

        smooth = [_gaussian_smooth(amp, s) for s in sigmas]
        # 持久度 = 该极小"存活"到的最大尺度 (样本)。两点关键:
        # ① 极小会随尺度**漂移**, 故逐尺度在当前位置邻域内跟踪 (半径 ~σ), 不能钉死
        #    在同一下标;
        # ② 用**严格**不等式, 否则平坦噪底上处处都是"持久极小"。
        persist = np.zeros(cand.size, dtype=np.float64)
        for j, p0 in enumerate(cand):
            pos, last = int(p0), 0.0
            for s, sm in zip(sigmas, smooth):
                r = max(1, int(np.ceil(s)))
                a, b = max(1, pos - r), min(sm.size - 1, pos + r + 1)
                if b - a < 2:
                    break
                k = a + int(np.argmin(sm[a:b]))
                if not (0 < k < sm.size - 1 and sm[k] < sm[k - 1] and sm[k] < sm[k + 1]):
                    break
                pos, last = k, float(s)
            persist[j] = last

        thr_otsu = _otsu_threshold(persist, int(kw["scale_space_otsu_bins"]))
        thr = max(thr_otsu, float(kw["scale_space_min_persist"]) * float(sigmas[-1]))
        strong = persist > thr

        # 按持久度贪心选取, 并做非极大值抑制 (间距 >= 最大尺度, 避免边界簇拥到
        # 相邻几个频点上); 数量不足所需时先用次高持久度补足, 连通候选都活不过
        # 最小尺度 (近单调谱) 才退回局部极大值法。
        min_gap = max(1, int(np.ceil(sigmas[-1])))
        picked = []
        for j in np.argsort(persist)[::-1]:
            if not strong[j] and len(picked) >= n_bands - 1:
                break
            if np.any(np.abs(cand[j] - cand[picked]) < min_gap):
                continue
            picked.append(int(j))
            if len(picked) == n_bands - 1:
                break
        if len(picked) < n_bands - 1:
            return self._boundary_maximum(amp, frequency, **kw)
        return np.sort(frequency[cand[picked]])

    def _boundary_envelope(self, amp, frequency, **kw) -> np.ndarray:
        """包络/阈值法: 幅谱样条包络 → 按 SNR 定硬阈值修平 → 检峰检谷;
        模式数由显著峰数自动确定 (上限 ``num_imfs``)。"""
        n = int(amp.size)
        kw_peaks = dict(kw)

        # --- 1) 上包络: 过局部极大的样条, 节点不足则用原谱 --- #
        pk = self._peaks(amp, **kw_peaks)
        env = np.asarray(amp, dtype=np.float64)
        if pk.size >= 2:
            try:
                grid = np.arange(n, dtype=np.float64)
                sp = _spline_fit(
                    pk.astype(np.float64), amp[pk],
                    kw["envelope_spline"],
                )
                env = np.asarray(sp(grid), dtype=np.float64)
            except Exception:
                env = np.asarray(amp, dtype=np.float64)

        # --- 2) 按 SNR 设定硬阈值并修平 --------------------------------- #
        # 阈值 = 噪底抬升 ``envelope_snr_db`` dB: 显著峰的判据是"比噪底高
        # snr_db"。若改成"比峰值低 snr_db"(即取峰高的约 0.71 倍), 就只有全局最大
        # 峰能过线, n_bands 会退化到 1。
        noise = float(np.percentile(env, float(kw["envelope_noise_pct"])))
        peak = float(np.max(env))
        tiny = 1e-300
        thr = abs(noise) * 10.0 ** (float(kw["envelope_snr_db"]) / 20.0)
        thr = float(min(max(thr, abs(noise)), peak))
        flat = np.maximum(env, thr)

        # --- 3) 检峰: 高于阈值且突出度达标的显著峰 ----------------------- #
        cand = self._peaks(flat, **kw_peaks)
        if cand.size < 2:
            return self._boundary_maximum(amp, frequency, **kw)
        prom = np.maximum(flat[cand] - thr, 0.0) / max(peak - thr, tiny)
        # 突出度下限取"配置值"与"候选突出度的 75 分位"的较小者 (分位数口径):
        # 候选峰多时避免硬阈值把峰剪到 1 个, 候选峰稀少时仍由配置值把关。
        floor = min(float(kw["envelope_prominence"]), float(np.percentile(prom, 75.0)))
        if floor <= 0.0:
            floor = float(kw["envelope_prominence"])   # 避免 0 分位放进"被修平"的伪峰
        # 候选按**突出度**降序: 显著峰 = 过突出度下限者; 不足 2 个则取前 2 个
        # (最少 2 峰回退 ⇒ n_bands >= 2); 超过 num_imfs 个时只保留最显著的
        # num_imfs 个 —— 与 maximum / max-min 分支"取前 K 大峰"同口径。
        order = np.argsort(prom)[::-1]
        sel = np.sort(cand[order[prom[order] >= floor]])
        if sel.size < 2:
            sel = np.sort(cand[order[:2]])
        elif sel.size > self.num_imfs:
            keep = np.sort(order[:self.num_imfs])
            sel = np.sort(cand[keep])

        # --- 4) 谷底作边界; 模式数自动确定 (上限 num_imfs) ----------------- #
        # m 个显著峰最多能给出 m-1 条"相邻峰间谷底"内部边界 ⇒ m 个频带。
        n_bands = max(1, int(min(self.num_imfs, sel.size)))
        fb = kw.get("max_min_fallback", "midpoint")
        return np.array(
            [_valley_between(amp, sel[i], sel[i + 1], frequency, fb) for i in range(n_bands - 1)]
        )


# --------------------------------------------------------------------------- #
# 模块级常量: 分支登记 / 参数默认值 / 惰性后端表
#
# 放在类之后是为了让文件顶部的布局与参照模块一致
# (docstring → import → __all__ → Literal → Enum → Config 横幅 → EWTConfig →
#  Class 横幅 → EWT); 这些常量只在实例方法运行时被读取, 定义位置不影响行为。
# --------------------------------------------------------------------------- #
#: 分块滤波时算子每元素展开的 float64 临时量个数 (``np.select`` 的候选分支 +
#: 掩码 + 输出): 用于把 ``EWT_CHUNK_WORK_BYTES`` 反推成块长 (元素数)。
_OPERATOR_TMP_ELEMS = 12

#: 时域预处理分支 (施加在 ``Check_Time_and_Signal`` 之后的信号上); 由 Enum 派生。
PRE_DEAL_MOD_TIME = (
    EWT_PRE_DEAL_MODS.NoDC.value,
    EWT_PRE_DEAL_MODS.NoTrend.value,
    EWT_PRE_DEAL_MODS.Window.value,
)
#: 频域预处理分支 (施加在 ``fft`` 之后的谱上; 只影响边界检测用的幅度谱)。
#: 顺序即施加顺序: 先做多锥谱估计, 再对其结果做平滑。
PRE_DEAL_MOD_FREQ = (
    EWT_PRE_DEAL_MODS.SlepianOptimize.value,
    EWT_PRE_DEAL_MODS.Smooth.value,
)

#: 分支名别名 → canonical 名 (大小写不敏感)。
_PRE_DEAL_ALIAS = {
    "nodc": EWT_PRE_DEAL_MODS.NoDC.value,
    "no_dc": EWT_PRE_DEAL_MODS.NoDC.value,
    "notrend": EWT_PRE_DEAL_MODS.NoTrend.value,
    "no_trend": EWT_PRE_DEAL_MODS.NoTrend.value,
    "slepian": EWT_PRE_DEAL_MODS.SlepianOptimize.value,
}

#: 预处理分支参数默认值 (``__init__`` 的 kwargs 逐名覆盖; 未传入即用此值)。
PRE_DEAL_DEFAULTS = {
    "no_dc_type": EWT_NO_DC_TYPE_MODS.Linear.value,          # detrend 的 type
    "no_dc_bp": 0,                 # detrend 断点 (仅 1-D; 0 = 不使用)
    "no_trend_win": None,          # 移动平均窗长; None -> 自动 (约 N//20 的奇数, >= 3)
    "window_kind": EWT_WINDOW_KIND_MODS.Hann.value,          # 接受 "hanning" 别名
    "slepian_nw": 3.0,             # dpss 时间-带宽积 NW
    "slepian_k": None,             # dpss 锥数 K; None -> 2*NW-1
    "slepian_mod": None,           # Slepian 后端; None -> ConstDefine.SLEPIAN_BACKEND
    # 启用 Slepian 分支的样本上限 (时间预算): 默认取 ConstDefine 值 32768;
    # 传 None 表示不设上限 (大 N 下每次 decompose 多付约 O(N) 的 dpss 时间)。
    "slepian_max_samples": EWT_SLEPIAN_MAX_SAMPLES,
    "smooth_kind": EWT_SMOOTH_KIND_MODS.Box.value,           # (Smooth 分支)
    "smooth_width": 5,             # 平滑窗宽 (频点数): box = 窗长, gaussian = σ
}

#: 边界策略参数默认值 (同样由 ``__init__`` 的 kwargs 覆盖)。
BOUNDARY_DEFAULTS = {
    "peak_distance": None,             # find_peaks 的 distance 过滤 (maximum / max-min)
    "max_min_fallback": EWT_MAX_MIN_FALLBACK_MODS.Midpoint.value,   # 段内无极小的回退
    "scale_space_scales": 32,          # 尺度数 (几何级数)
    "scale_space_sigma_min": 1.0,      # 最小高斯尺度 (样本)
    "scale_space_sigma_max": 32.0,     # 最大高斯尺度 (样本)
    "scale_space_otsu_bins": 64,       # 持久度 Otsu 直方图桶数
    "scale_space_min_persist": 0.25,   # 持久度下限 (占尺度数的比例)
    "envelope_spline": EWT_ENVELOPE_SPLINE_MODS.CubicSpline.value,  # 见 ``_SPLINE_CTORS``
    "envelope_noise_pct": 10.0,        # 噪底估计分位
    "envelope_snr_db": 3.0,            # 硬阈值 = 噪底 × 10^(dB/20): 噪底之上的抬升量
    "envelope_prominence": 0.05,       # 峰的相对突出度下限
}

#: ``**kwargs`` 里除分支参数外允许出现的库级参数 (其余键一律报 TypeError)。
_KWARGS_ALLOWED = frozenset(PRE_DEAL_DEFAULTS) | frozenset(BOUNDARY_DEFAULTS) | {"safe", "copy"}

#: 包络样条后端名 → ``scipy.interpolate`` 构造器名 (本文件自持映射, 只在
#: envelope 分支真正用到时才惰性导入 scipy)。键是 ``EWT_ENVELOPE_SPLINE_MODS`` 的成员值。
_SPLINE_CTORS = {
    EWT_ENVELOPE_SPLINE_MODS.UnivariateSpline.value: "UnivariateSpline",
    EWT_ENVELOPE_SPLINE_MODS.CubicSpline.value: "CubicSpline",
    EWT_ENVELOPE_SPLINE_MODS.PCHIP.value: "PchipInterpolator",
    EWT_ENVELOPE_SPLINE_MODS.Akima.value: "Akima1DInterpolator",
}


# --------------------------------------------------------------------------- #
# 模块级工具函数: mod 归一 / 预处理分支名规范化 / 惰性样条
# --------------------------------------------------------------------------- #
def _canon_mod_lower(mod, MOD):
    """大小写不敏感的 mod 归一: 字符串先 strip+lower, 再走 ``Check.validate_mod``。

    返回值一律是 ``MOD`` 的 Enum 成员 (Enum 成员原样返回)。
    """
    if isinstance(mod, str):
        mod = mod.strip().lower()
    return Check.validate_mod(mod, MOD)


def _canon_window_kind(mod) -> EWT_WINDOW_KIND_MODS:
    """``window_kind`` → ``EWT_WINDOW_KIND_MODS`` (``"hanning"`` 是 ``"hann"`` 的别名)。"""
    if isinstance(mod, str) and mod.strip().lower() in ("hann", "hanning"):
        mod = EWT_WINDOW_KIND_MODS.Hann
    return _canon_mod_lower(mod, EWT_WINDOW_KIND_MODS)


def _canon_pre_deal_one(mod) -> str:
    """单个预处理分支名 → canonical 名 (大小写不敏感, 支持别名)。

    接受 ``EWT_PRE_DEAL_MODS`` 成员; 返回值仍是 canonical **字符串** (沿用
    ``pre_deal_mods`` / ``info["pre_deal"]`` 的既有契约), 但取值由
    ``Check.validate_mod`` 对 Enum 校验后取 ``.value``。
    """
    key = str(getattr(mod, "value", mod)).strip()
    if key == "" or key.lower() == "none":
        return "None"
    table = {name.lower(): name for name in PRE_DEAL_MOD_TIME + PRE_DEAL_MOD_FREQ}
    table.update(_PRE_DEAL_ALIAS)
    low = key.lower()
    if low not in table:
        raise ValueError(
            f"unsupported pre_deal mod {mod!r}; expected one of "
            f"{list(PRE_DEAL_MOD_TIME + PRE_DEAL_MOD_FREQ)} or 'None'"
        )
    return Check.validate_mod(table[low], EWT_PRE_DEAL_MODS).value


def _canon_pre_deal_mods(pre_deal) -> tuple:
    """单个名 / 序列 / None → canonical 顺序元组 (时域 → 频域, 去重)。"""
    if pre_deal is None:
        return ()
    items = (pre_deal,) if isinstance(pre_deal, (str, EWT_PRE_DEAL_MODS)) else tuple(pre_deal)
    seen = []
    for m in items:
        name = _canon_pre_deal_one(m)
        if name != "None" and name not in seen:
            seen.append(name)
    return tuple(m for m in (PRE_DEAL_MOD_TIME + PRE_DEAL_MOD_FREQ) if m in seen)


def _spline_fit(x, y, kind):
    """
    惰性构造包络样条 (只在 ``boundary_mod="envelope"`` 真正用到时才导入 scipy)。

    ``kind`` 是 ``EWT_ENVELOPE_SPLINE_MODS`` 成员 (或其 ``.value``); 返回可直接
    调用的插值对象 (``UnivariateSpline`` 用 ``s=0`` 精确插值, 其余后端打开
    ``extrapolate``)。scipy 缺失时只在此处报错, 不影响 EWT 的其余分支。
    """
    kind = Check.validate_mod(kind, EWT_ENVELOPE_SPLINE_MODS)
    ctor_name = _SPLINE_CTORS[kind.value]

    interpolate = Cache.import_module(
        "scipy.interpolate",
        description="scipy.interpolate: 供 EWT envelope 分支拟合幅谱包络",
    )
    if interpolate is None:
        raise ImportError(
            "EWT boundary_mod='envelope' needs scipy: python -m pip install scipy"
        )

    ctor = getattr(interpolate, ctor_name, None)
    if ctor is None:
        raise ImportError(f"scipy.interpolate has no {ctor_name!r}")

    match kind:
        case EWT_ENVELOPE_SPLINE_MODS.UnivariateSpline:
            return ctor(x, y, s=0)
        case _:
            return ctor(x, y, extrapolate=True)


# --------------------------------------------------------------------------- #
# 数值助手 (预处理/边界分支共用; 纯 numpy, 不引入额外依赖)
# --------------------------------------------------------------------------- #
def _moving_average(x: np.ndarray, win=None) -> np.ndarray:
    """
    端点复制填充的移动平均 (O(N), 纯 numpy)。

    ``win=None`` 时自动取 ``max(3, (N//20) | 1)`` (奇数, 且不超过 N)。
    """
    n = int(x.size)
    if win is None:
        win = max(3, (n // 20) | 1)
    win = int(win)
    if win < 1:
        raise ValueError(f"moving average window must be >= 1, got {win}")
    win = min(win, n)
    if win == 1:
        return np.array(x, dtype=DEFAULT_NUMPY_TYPE, copy=True)
    half = win // 2
    pad = np.pad(np.asarray(x, dtype=DEFAULT_NUMPY_TYPE), (half, win - 1 - half), mode="edge")
    cum = np.concatenate(([0.0], np.cumsum(pad)))
    return (cum[win:] - cum[:-win]) / float(win)


def _window_vec(n: int, kind="hann") -> np.ndarray:
    """时域窗向量: ``EWT_WINDOW_KIND_MODS.Hann`` (别名 ``"hanning"``) / ``Hamming``。"""
    match _canon_window_kind(kind):
        case EWT_WINDOW_KIND_MODS.Hann:
            return np.hanning(int(n))
        case EWT_WINDOW_KIND_MODS.Hamming:
            return np.hamming(int(n))


def _gaussian_smooth(x: np.ndarray, sigma: float) -> np.ndarray:
    """高斯平滑 (核半宽 3σ, 奇数长度核, ``mode="same"`` 不产生位移)。"""
    x = np.asarray(x, dtype=DEFAULT_NUMPY_TYPE)
    sigma = float(sigma)
    if sigma <= 0.0 or x.size == 0:
        return x.copy()
    half = max(1, int(np.ceil(3.0 * sigma)))
    t = np.arange(-half, half + 1, dtype=DEFAULT_NUMPY_TYPE)
    ker = np.exp(-0.5 * (t / sigma) ** 2)
    ker /= ker.sum()
    return np.convolve(x, ker, mode="same")


def _otsu_threshold(values: np.ndarray, bins: int = 64) -> float:
    """一维 Otsu 阈值 (直方图法); 常数序列返回该常数。"""
    v = np.asarray(values, dtype=DEFAULT_NUMPY_TYPE)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return 0.0
    lo, hi = float(v.min()), float(v.max())
    if hi <= lo:
        return lo
    hist, edges = np.histogram(v, bins=max(2, int(bins)), range=(lo, hi))
    p = hist.astype(DEFAULT_NUMPY_TYPE) / max(float(hist.sum()), 1.0)
    centers = 0.5 * (edges[:-1] + edges[1:])
    w0 = np.cumsum(p)
    w1 = 1.0 - w0
    m0 = np.cumsum(p * centers)
    mt = m0[-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        mu0 = m0 / w0
        mu1 = (mt - m0) / w1
        var = w0 * w1 * (mu0 - mu1) ** 2
    var = np.nan_to_num(var, nan=-1.0, posinf=-1.0, neginf=-1.0)
    return float(centers[int(np.argmax(var))])


def _local_minima(x: np.ndarray) -> np.ndarray:
    """局部极小下标 (平台取左缘, 不含端点)。"""
    x = np.asarray(x, dtype=DEFAULT_NUMPY_TYPE)
    if x.size < 3:
        return np.empty(0, dtype=np.int64)
    mask = (x[1:-1] < x[:-2]) & (x[1:-1] <= x[2:])
    return np.flatnonzero(mask) + 1


def _valley_between(
    amp,
    i0: int,
    i1: int,
    frequency,
    fallback: EWT_MAX_MIN_FALLBACK_MODS_ARGS | EWT_MAX_MIN_FALLBACK_MODS = "midpoint",
) -> float:
    """
    在 ``(i0, i1)`` 开区间内取幅值最小处作边界 ("谷底分割")。

    段内极小落在区间端点 (段单调) 或段长不足时按 ``fallback`` 回退:
    ``EWT_MAX_MIN_FALLBACK_MODS.Midpoint`` 取两峰频率中点,
    ``Geometric`` 取几何平均。``fallback`` 接受成员或其 ``.value`` 字符串。
    """
    fallback = Check.validate_mod(fallback, EWT_MAX_MIN_FALLBACK_MODS)
    amp = np.asarray(amp, dtype=DEFAULT_NUMPY_TYPE)
    frequency = np.asarray(frequency, dtype=DEFAULT_NUMPY_TYPE)
    if i1 - i0 >= 3:
        seg = amp[i0 + 1:i1]
        j = int(np.argmin(seg))
        if 0 < j < seg.size - 1:
            return float(frequency[i0 + 1 + j])
    f0, f1 = float(frequency[i0]), float(frequency[i1])
    if fallback is EWT_MAX_MIN_FALLBACK_MODS.Geometric and f0 > 0.0 and f1 > 0.0:
        return float(np.sqrt(f0 * f1))
    return 0.5 * (f0 + f1)


"""
other function for module
"""
def f_EWT(S: list | np.ndarray, T=None, fs: float = 1.0, output_memmap: bool = False, *, num_imfs: int = 8, band_factor: float | EWT_BAND_FACTOR_MODS_ARGS | EWT_BAND_FACTOR_MODS = "auto", boundary_mod: EWT_BOUNDARY_MODS_ARGS | EWT_BOUNDARY_MODS = "maximum", pre_deal: EWT_PRE_DEAL_MODS_ARGS | Sequence[EWT_PRE_DEAL_MODS_ARGS] | None = None, mirror: bool = True, dtype: type | np.dtype | None = None, **kwargs) -> DecompositionResult:
    ewt = EWT(
        num_imfs=num_imfs,
        band_factor=band_factor,
        boundary_mod=boundary_mod,
        pre_deal=pre_deal,
        mirror=mirror,
        dtype=dtype,
        **kwargs
    )

    return ewt.decompose(S, T, fs=fs, output_memmap=output_memmap)
