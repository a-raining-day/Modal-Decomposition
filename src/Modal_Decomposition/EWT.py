"""
Empirical Wavelet Transform —— 本库默认实现 (自研经验小波滤波器组)。

注册键 ``"EWT"`` 由本文件提供 (``Class.EWT`` / ``Function.EWT``)。

频域支撑上的自适应滤波器组由 ``Operator`` 层提供: ``Daubechies_polynomial``
给出单位分解光滑阶跃 β(x) (β(x)+β(1−x)=1), ``Empirical_wavelet`` 用 β 构造
带过渡带 γ 的经验小波传递函数 ``H(w0, w1)`` (``w0=0`` 时退化为低通尺度函数)。

滤波器组、边界检测、变换与逆变换全部在本库内完成, **不依赖 ewtpy**(原 ewtpy
包装版已迁到 ``EWTpy.py``, 作为可选适配层, 注册键 ``"EWTpy"``)。已知与
Gilles 2013 原文/ewtpy 的对应关系:
``Σ_k H_k²(ω) ≡ 1`` (能量型紧框架, 逐点验证误差 < 1e-16); 而幅度型
``Σ_k H_k(ω)`` 在过渡带内最大到 √2, 故单次滤波所得各带之和并不等于原信号 ——
本实现把该差额全部放进 ``Res = S − ΣIMFs``, 于是 ``reconstruct()`` 仍精确。

单位约定
--------
``boundaries`` / ``frequency`` 一律是**频率** (Hz, 0…fs/2, 取自 ``rfftfreq``),
不是频点下标; ``band_factor`` 是过渡带相对宽度 (0<γ<1), 过渡带为
``[(1−γ)w, (1+γ)w]``; 默认 ``band_factor="auto"`` 时 γ 由边界间距反推
(见 :meth:`EWT._resolve_band_factor`), 以保证相邻过渡带不交叠、``Σ H_k² ≡ 1``
逐点成立 —— 显式传 float 则沿用固定宽度 (旧默认 ``0.15``)。

References
----------
10.48550/arXiv.2304.06274
"""

import numpy as np
from dataclasses import dataclass
from typing import ClassVar, Literal, Sequence

from .Base import Config, Decomposer, DecompositionResult
from .Base.Cache import cache
from .Base.ConstDefine import (
    BIG_ARRAY,
    CACHE_KEY,
    EWT_CHUNK_WORK_BYTES,
    EWT_SLEPIAN_MAX_SAMPLES,
    SPLINE_KIND,
)
from ._Registry import register_class
from .Utils import (
    Check_Time_and_Signal,
    get_fft,
    get_mirror,
    get_peaks,
    get_slepian,
    get_spline,
)
from .Utils.Chunk import default_chunk_size, iter_chunks
from .Operator import empirical_wavelet_arr

__all__ = ["EWT", "EWTConfig"]

#: 分块滤波时算子每元素展开的 float64 临时量个数 (``np.select`` 的候选分支 +
#: 掩码 + 输出): 用于把 ``EWT_CHUNK_WORK_BYTES`` 反推成块长 (元素数)。
_OPERATOR_TMP_ELEMS = 12

# --------------------------------------------------------------------------- #
# 预处理分支登记与参数默认值
# --------------------------------------------------------------------------- #
#: 时域预处理分支 (施加在 ``Check_Time_and_Signal`` 之后的信号上)。
PRE_DEAL_MOD_TIME = ("no-dc", "no-trend", "window")
#: 频域预处理分支 (施加在 ``fft`` 之后的谱上; 只影响边界检测用的幅度谱)。
#: 顺序即施加顺序: 先做多锥谱估计, 再对其结果做平滑。
PRE_DEAL_MOD_FREQ = ("Slepian-Optimize", "Smooth")

#: 分支名别名 → canonical 名 (大小写不敏感)。
_PRE_DEAL_ALIAS = {
    "nodc": "no-dc",
    "no_dc": "no-dc",
    "notrend": "no-trend",
    "no_trend": "no-trend",
    "slepian": "Slepian-Optimize",
}

#: 预处理分支参数默认值 (``__init__`` 的 kwargs 逐名覆盖; 未传入即用此值)。
PRE_DEAL_DEFAULTS = {
    "no_dc_type": "linear",        # scipy.signal.detrend 的 type: {"linear","constant"}
    "no_dc_bp": 0,                 # detrend 断点 (仅 1-D; 0 = 不使用)
    "no_trend_win": None,          # 移动平均窗长; None -> 自动 (约 N//20 的奇数, >= 3)
    "window_kind": "hann",         # {"hann","hamming"} (接受 "hanning" 别名)
    "slepian_nw": 3.0,             # dpss 时间-带宽积 NW
    "slepian_k": None,             # dpss 锥数 K; None -> 2*NW-1
    "slepian_mod": None,           # Slepian 后端; None -> ConstDefine.SLEPIAN_BACKEND
    # 启用 Slepian 分支的样本上限 (时间预算): 默认取 ConstDefine 值 32768;
    # 传 None 表示不设上限 (大 N 下每次 decompose 多付约 O(N) 的 dpss 时间)。
    "slepian_max_samples": EWT_SLEPIAN_MAX_SAMPLES,
    "smooth_kind": "box",          # 频域平滑窗 {"box","gaussian"} (Smooth 分支)
    "smooth_width": 5,             # 平滑窗宽 (频点数): box = 窗长, gaussian = σ
}

#: 边界策略参数默认值 (同样由 ``__init__`` 的 kwargs 覆盖)。
BOUNDARY_DEFAULTS = {
    "peak_distance": None,             # find_peaks 的 distance 过滤 (maximum / max-min)
    "max_min_fallback": "midpoint",    # 段内无极小: {"midpoint","geometric"}
    "scale_space_scales": 32,          # 尺度数 (几何级数)
    "scale_space_sigma_min": 1.0,      # 最小高斯尺度 (样本)
    "scale_space_sigma_max": 32.0,     # 最大高斯尺度 (样本)
    "scale_space_otsu_bins": 64,       # 持久度 Otsu 直方图桶数
    "scale_space_min_persist": 0.25,   # 持久度下限 (占尺度数的比例)
    "envelope_spline": "CubicSpline",  # 包络样条后端 (Utils.Spline 的 canonical 名)
    "envelope_noise_pct": 10.0,        # 噪底估计分位
    "envelope_snr_db": 3.0,            # 硬阈值比峰值低多少 dB
    "envelope_prominence": 0.05,       # 峰的相对突出度下限
}


def _canon_pre_deal_one(mod) -> str:
    """单个预处理分支名 → canonical 名 (大小写不敏感, 支持别名)。"""
    key = str(mod).strip()
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
    return table[low]


def _canon_pre_deal_mods(pre_deal) -> tuple:
    """单个名 / 序列 / None → canonical 顺序元组 (时域 → 频域, 去重)。"""
    if pre_deal is None:
        return ()
    items = (pre_deal,) if isinstance(pre_deal, str) else tuple(pre_deal)
    seen = []
    for m in items:
        name = _canon_pre_deal_one(m)
        if name != "None" and name not in seen:
            seen.append(name)
    return tuple(m for m in (PRE_DEAL_MOD_TIME + PRE_DEAL_MOD_FREQ) if m in seen)


@dataclass(frozen=True, kw_only=True)
class EWTConfig(Config):
    """
    Effective parameters of an EWT run.
    """
    num_imfs: int
    band_factor: float
    boundary_mod: str
    pre_deal: tuple
    mirror: bool
    fs: float


@register_class("EWT")
class EWT(Decomposer):
    name: ClassVar[str] = "EWT"

    pre_deal_mod_constraint = ("None", "no-DC", "no-trend", "window", "Slepian-Optimize", "Smooth")
    pre_deal_mod = {"None", "no-DC", "no-trend", "window", "Slepian-Optimize", "Smooth"}
    boundary_mod_constraint = Literal["maximum", "max-min", "scale-space", "envelope"]
    boundary_mods = {"maximum", "max-min", "scale-space", "envelope"}

    def __init__(
        self,
        num_imfs: int = 5,
        band_factor: float | Literal["auto"] = "auto",
        boundary_mod: Literal["maximum", "max-min", "scale-space", "envelope"] = "maximum",
        pre_deal: str | Sequence[str] | None = None,
        mirror: bool = True,
        **kwargs
    ):
        """
        Parameters
        ----------
        num_imfs : int
            Number of modes (bands of the filter bank); must be >= 1. For the
            boundary strategies it is the **maximum**: ``"maximum"`` /
            ``"max-min"`` / ``"scale-space"`` always yield ``num_imfs`` bands,
            while ``"envelope"`` may yield fewer (模式数由显著峰数自动确定).
        band_factor : float | {"auto"}
            Relative transition width γ of each empirical wavelet: the transition
            occupies ``[(1−γ)w, (1+γ)w]`` around a boundary ``w``.

            * ``"auto"`` (default) —— **按检测到的边界间距反推**(见
              :meth:`_resolve_band_factor`): ``γ`` 取"相邻过渡带恰好不交叠"的
              上确界, 于是滤波器组满足紧框架 ``Σ_i H_i² ≡ 1``, 幅度带和误差最小
              (与 ewtpy 的自动 gamma 同法)。
            * ``float`` (``0 < γ < 1``) —— 固定过渡带宽度, 数值语义与旧版一致;
              γ 偏大时相邻过渡带叠加 (``Σ H² > 1``, 同一频段被重复计入),
              偏小则频带间出现重构空档。旧版默认值为 ``0.15``。
        boundary_mod : {"maximum", "max-min", "scale-space", "envelope"}
            Spectral boundary detection strategy (integrated by
            :meth:`boundary`; 见其文档的四套规则).
        pre_deal : str | Sequence[str] | None
            预处理分支选择 (单个名或序列; ``None``/``"None"`` = 不预处理)。
            分支按 canonical 顺序施加: 时域 ``("no-dc", "no-trend", "window")``
            在 ``Check_Time_and_Signal`` 之后作用于信号, 频域
            ``("Slepian-Optimize", "Smooth")`` 在 ``fft`` 之后作用于
            **边界检测用的幅度谱** (滤波仍用原复数谱)。``mirror`` 不属于预处理
            (它由独立参数控制, 在时域预处理之后、``fft`` 之前施加)。
            ``"Smooth"`` (幅度谱平滑, 默认**不开启**) 用于抑制伪峰、让
            ``maximum`` 也能给出贴合的边界, 代价是噪声下抖动变大。
        mirror : bool
            ``True`` (default): 频域滤波在**端点镜像延拓**后的信号上进行
            (``Utils.Mirror.mirror_signal`` 整条信号语义, 长度 ``2N``, 轴长与
            原信号共享 ``0…fs/2``), 逆变换后取中间 ``N`` 点 —— 同 ewtpy 的
            ``fMirr`` 处理, 抑制频域滤波的环形卷积边界效应。
            ``False``: 直接在原信号的单边谱上滤波 (等价于对该长度信号做循环
            卷积)。两种取值下**边界检测都在原信号谱上**做, 故 ``boundaries``
            相同; 滤波结果与代价不同 (镜像多一次 ``rfft`` 且轴长翻倍)。
        **kwargs
            分支参数 (未传入即用默认值, 见 ``PRE_DEAL_DEFAULTS`` /
            ``BOUNDARY_DEFAULTS``): 时域 ``no_dc_type`` / ``no_dc_bp`` /
            ``no_trend_win`` / ``window_kind``; 频域 ``slepian_nw`` /
            ``slepian_k`` / ``slepian_mod`` / ``slepian_max_samples`` /
            ``smooth_kind`` / ``smooth_width``;
            边界 ``peak_distance`` /
            ``max_min_fallback`` / ``scale_space_scales`` /
            ``scale_space_sigma_min`` / ``scale_space_sigma_max`` /
            ``scale_space_otsu_bins`` / ``scale_space_min_persist`` /
            ``envelope_spline`` / ``envelope_noise_pct`` / ``envelope_snr_db`` /
            ``envelope_prominence``。未识别的键保留在 ``self.kwargs``。
        """
        if not isinstance(num_imfs, int) or isinstance(num_imfs, bool) or num_imfs < 1:
            raise ValueError(f"num_imfs must be an int >= 1, got {num_imfs!r}")
        if isinstance(band_factor, str):
            if band_factor != "auto":
                raise ValueError(
                    f"band_factor must be 'auto' or a float in (0, 1), got {band_factor!r}"
                )
        elif not (0.0 < float(band_factor) < 1.0):
            raise ValueError(f"band_factor must be in (0, 1), got {band_factor!r}")
        if boundary_mod not in self.boundary_mods:
            raise ValueError(
                f"boundary_mod must be one of {sorted(self.boundary_mods)}, got {boundary_mod!r}"
            )
        if not isinstance(mirror, bool):
            raise ValueError(f"mirror must be a bool, got {mirror!r}")

        self.num_imfs = num_imfs
        self.band_factor = band_factor if band_factor == "auto" else float(band_factor)
        self.boundary_mod = boundary_mod
        self.mirror = mirror

        # 属性名用 pre_deal_mods, 以免遮蔽同名的公共方法 ``pre_deal()``。
        self.pre_deal_mods = _canon_pre_deal_mods(pre_deal)
        self.pre_deal_time = tuple(m for m in self.pre_deal_mods if m in PRE_DEAL_MOD_TIME)
        self.pre_deal_freq = tuple(m for m in self.pre_deal_mods if m in PRE_DEAL_MOD_FREQ)

        # --- 分支参数: kwargs 逐名覆盖默认值 -------------------------------- #
        self.pre_deal_kwargs = {k: kwargs.get(k, v) for k, v in PRE_DEAL_DEFAULTS.items()}
        self.boundary_kwargs = {k: kwargs.get(k, v) for k, v in BOUNDARY_DEFAULTS.items()}
        self.kwargs = kwargs

        self._validate_branch_kwargs()

        # 工具惰性取用 (与全库一致)。get_fft() 返回 Utils.FFT 模块, 模块内唯一
        # 公开名是 fft 类 (全静态方法), 故变换统一写作 ``self.fft.fft.<op>(...)``。
        self.fft = get_fft()
        self.find_peaks = get_peaks().find_peaks
        self.mirror_signal = get_mirror().mirror_signal
        self.spline = get_spline().Spline

        self.frequency = None
        self.empirical_wave = None
        self.modal = None
        self._trend = None              # no-trend 分支构造出的趋势 (保留进 result)
        self._slepian_degraded = False  # Slepian 分支是否因规模上限而降级

    def _validate_branch_kwargs(self) -> None:
        """校验分支参数取值 (非法值立即报错, 不留到 decompose)。"""
        kw, bw = self.pre_deal_kwargs, self.boundary_kwargs

        if kw["no_dc_type"] not in ("linear", "constant"):
            raise ValueError(
                f"no_dc_type must be 'linear' or 'constant', got {kw['no_dc_type']!r}"
            )
        if kw["no_trend_win"] is not None and int(kw["no_trend_win"]) < 1:
            raise ValueError(f"no_trend_win must be >= 1 or None, got {kw['no_trend_win']!r}")
        window_kind = str(kw["window_kind"]).strip().lower()
        if window_kind not in ("hann", "hanning", "hamming"):
            raise ValueError(
                f"window_kind must be 'hann' or 'hamming', got {kw['window_kind']!r}"
            )
        if float(kw["slepian_nw"]) <= 0.0:
            raise ValueError(f"slepian_nw must be > 0, got {kw['slepian_nw']!r}")
        if kw["slepian_k"] is not None and int(kw["slepian_k"]) < 1:
            raise ValueError(f"slepian_k must be >= 1 or None, got {kw['slepian_k']!r}")
        if str(kw["smooth_kind"]).strip().lower() not in ("box", "gaussian"):
            raise ValueError(
                f"smooth_kind must be 'box' or 'gaussian', got {kw['smooth_kind']!r}"
            )
        if float(kw["smooth_width"]) < 1.0:
            raise ValueError(f"smooth_width must be >= 1, got {kw['smooth_width']!r}")

        if bw["max_min_fallback"] not in ("midpoint", "geometric"):
            raise ValueError(
                f"max_min_fallback must be 'midpoint' or 'geometric', got {bw['max_min_fallback']!r}"
            )
        if int(bw["scale_space_scales"]) < 2:
            raise ValueError(f"scale_space_scales must be >= 2, got {bw['scale_space_scales']!r}")
        if float(bw["scale_space_sigma_min"]) <= 0.0 or float(bw["scale_space_sigma_max"]) <= 0.0:
            raise ValueError("scale_space sigma bounds must be > 0")
        if int(bw["scale_space_otsu_bins"]) < 2:
            raise ValueError(f"scale_space_otsu_bins must be >= 2, got {bw['scale_space_otsu_bins']!r}")
        if bw["envelope_spline"] not in SPLINE_KIND:
            raise ValueError(
                f"envelope_spline must be one of {SPLINE_KIND}, got {bw['envelope_spline']!r}"
            )
        if not (0.0 <= float(bw["envelope_noise_pct"]) <= 100.0):
            raise ValueError(f"envelope_noise_pct must be in [0, 100], got {bw['envelope_noise_pct']!r}")
        if float(bw["envelope_snr_db"]) < 0.0:
            raise ValueError(f"envelope_snr_db must be >= 0, got {bw['envelope_snr_db']!r}")
        if not (0.0 <= float(bw["envelope_prominence"]) <= 1.0):
            raise ValueError(
                f"envelope_prominence must be in [0, 1], got {bw['envelope_prominence']!r}"
            )

    def decompose(self, S, T=None, fs: float = 1.0, **kwargs) -> DecompositionResult:
        """
        Decompose the signal into ``num_imfs`` band-limited modes.

        Parameters
        ----------
        S : array-like
            Signal (input layer ``Check_Time_and_Signal`` applies the global
            memory policy).
        T : array-like, optional
            Time axis; validated for length/duplicates/monotonicity but unused
            by the algorithm.
        fs : float
            Sampling frequency, only used to express the frequency axis and
            the reported boundaries in Hz (the transform itself is
            scale-invariant).
        **kwargs
            Forwarded to the FFT call of ``Utils.FFT`` (e.g. ``mod="scipy"``
            to pick a backend for this call).

        Returns
        -------
        DecompositionResult
            ``IMFs`` (num_imfs, N) real bands in ascending frequency order;
            ``Res`` = ``S − ΣIMFs`` (so ``reconstruct()`` reproduces ``S``);
            ``info`` with ``mfb`` (the filter bank, ``(num_imfs, N//2+1)``;
            ``None`` on the chunked path — see below), ``boundaries``
            (num_imfs+1, Hz), ``boundary_mod``, ``residual_ratio``
            (``‖Res‖ / ‖S‖``), ``chunked``, ``chunk_size`` and ``mirror``.

        Mirroring (``mirror`` 参数)
        -------------------------
        边界检测一律在原信号单边谱上做, 故 ``boundaries`` 与 ``mirror`` 取值无关;
        ``mirror=True`` (默认) 时滤波改在端点镜像延拓信号 (长度 ``2N``) 上做,
        逆变换后取中间 ``N`` 点, 以抑制频域滤波的环形卷积边界效应 (ewtpy 的
        ``fMirr`` 同法); ``mirror=False`` 时即对原长度信号做循环卷积。

        Filtering paths (滤波器组的两条路径)
        ----------------------------------
        滤波器组自身的体积 (``num_imfs × (N//2+1) × 8`` 字节) 达到
        ``Base.ConstDefine.BIG_ARRAY`` 时不再一次性物化整张滤波器组, 改为
        ``Utils.Chunk`` 分块 + ``Operator`` 算子逐块求值, 并逐模态落定 ——
        每模态只保留一条单边带谱 ``(N//2+1,)`` 复数组, 工作集与 ``num_imfs``
        解耦 (``info["chunked"]`` 为 True, ``info["mfb"]`` 为 None); 未达阈值
        时沿用一次性物化的向量化路径, 两条路径逐元素结果一致。
        """
        S, T, N = Check_Time_and_Signal(S, T, ndim={1}, method=self.name)

        S = np.asarray(S, dtype=np.float64)
        S_in = S                                # 原始输入: Res / reconstruct 的基准

        # --- 时域预处理: Check_Time_and_Signal 之后立即施加 ----------------- #
        S = self._time_domain_pre_deal(S)

        # 边界检测始终在**预处理后信号的原始谱**上做 (与 ewtpy 一致: 镜像只作用
        # 于滤波阶段, 不改变 boundaries)。
        F_orig = self.fft.fft.rfft(S, **kwargs)         # (N//2+1,) 单边谱
        freq_orig = np.fft.rfftfreq(S.size, 1.0 / fs)   # 0…fs/2, 含 Nyquist
        self.frequency = freq_orig

        # --- 频域预处理: fft 之后立即施加 (产出边界检测用的幅度谱) ---------- #
        amp = self._freq_domain_pre_deal(S, F_orig)

        boundary = self.boundary(amp, freq_orig, mod=self.boundary_mod)
        n_bands = int(boundary.size) - 1        # "envelope" 分支可少于 num_imfs

        # mirror=True: 滤波在端点镜像延拓信号上进行 (Utils.Mirror 整条信号语义,
        # 长度 2N), 逆变换后取中间 N 点 —— 同 ewtpy 的 fMirr 处理, 抑制频域滤波
        # 的环形卷积边界效应。镜像轴与原轴共享 0…fs/2 频率范围, 故 boundary 通用。
        if self.mirror:
            f_mirr = self.mirror_signal(S)
            axis_len = int(f_mirr.size)
            F = self.fft.fft.rfft(f_mirr)
            frequency = np.fft.rfftfreq(axis_len, 1.0 / fs)
            left = N // 2                       # 镜像左端样本数 (与 mirror_signal 一致)
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
            self.modal = np.zeros((n_bands, N))

        # 大数组判据: 滤波器组自身的体积 (n_bands × n_bins, float64) 是否达到
        # Base.ConstDefine.BIG_ARRAY。达到则不物化整张滤波器组, 改为 Chunk 分块 +
        # 算子逐块求值, 并逐模态落定 (工作集与频带数解耦, 只留一条带谱)。
        filter_bytes = n_bands * n_bins * np.dtype(np.float64).itemsize
        chunked = filter_bytes >= BIG_ARRAY

        if chunked:
            # 块长按"目标单块工作集"反推 (与 VMD_CHUNK_WORK_BYTES 同法): 块内除
            # 算子输出外还有 _OPERATOR_TMP_ELEMS 个 float64 临时量, 故
            # base = EWT_CHUNK_WORK_BYTES / (临时量数 × 8 字节); 再交给内存策略
            # 按当前预算继续收缩 (`default_chunk_size` 的 extra_per_elem 与之同源)。
            base = max(1, EWT_CHUNK_WORK_BYTES // (_OPERATOR_TMP_ELEMS * np.dtype(np.float64).itemsize))
            chunk_size = default_chunk_size(
                filter_bytes, base=min(base, n_bins), extra_per_elem=_OPERATOR_TMP_ELEMS
            )
            self.empirical_wave = None              # 大数组下不保留整张滤波器组
            for row in range(n_bands):
                band_hat = np.empty(n_bins, dtype=np.complex128)
                pos = 0
                with np.errstate(divide="ignore", invalid="ignore"):
                    for f_chunk, f_hat_chunk in zip(
                        iter_chunks(frequency, chunk_size), iter_chunks(F, chunk_size)
                    ):
                        n_chunk = f_chunk.size
                        band_hat[pos:pos + n_chunk] = empirical_wavelet_arr(
                            f_chunk,
                            band_factor=band_factor,
                            w0=filter_boundary[row],
                            w1=filter_boundary[row + 1],
                        ) * f_hat_chunk
                        pos += n_chunk
                self.modal[row, :] = self.fft.fft.irfft(band_hat, n=axis_len)[left:left + N]
        else:
            chunk_size = None
            if self.empirical_wave is None or self.empirical_wave.shape != (n_bands, n_bins):
                self.empirical_wave = np.zeros((n_bands, n_bins))

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
            full = self.fft.fft.irfft(freq_modal, n=axis_len, axis=1)
            # 镜像路径下取中间 N 点 (复制成连续数组, 避免留下切片视图)
            self.modal = np.ascontiguousarray(full[:, left:left + N])

        IMFs = np.asarray(self.modal, dtype=np.float64)

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

        return DecompositionResult(
            IMFs,
            Res,
            {
                "mfb": self.empirical_wave,
                "boundaries": boundary,
                "boundary_mod": self.boundary_mod,
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
            EWTConfig(
                num_imfs=self.num_imfs,
                band_factor=band_factor,            # effective γ (auto 已解析)
                boundary_mod=self.boundary_mod,
                pre_deal=tuple(self.pre_deal_mods),
                mirror=self.mirror,
                fs=fs,
            ),
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
        * ``window``   —— 加窗 (``window_kind`` ∈ {"hann","hamming"};
          ``np.hanning`` / ``np.hamming``, 对称窗); 加窗损失同样并入 ``Res``。

        Parameters
        ----------
        S : np.ndarray
            时域信号 (check 之后, mirror 之前)。
        mods : Sequence[str] | None
            显式指定分支 (按 canonical 顺序重排); None 时用 ``self.pre_deal_time``。
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
            if mod == "no-dc":
                sp = cache.import_module(
                    CACHE_KEY["scipy"]["signal"],
                    description="scipy.signal: detrend (供 EWT no-dc 分支使用)",
                )
                S = np.asarray(
                    sp.detrend(S, type=kw["no_dc_type"], bp=int(kw["no_dc_bp"])),
                    dtype=np.float64,
                )
            elif mod == "no-trend":
                trend = _moving_average(S, kw["no_trend_win"])
                S = S - trend
                self._trend = trend
            elif mod == "window":
                S = S * _window_vec(S.size, kw["window_kind"])
        return S

    def _freq_domain_pre_deal(self, S: np.ndarray, F: np.ndarray, mods=None, **kwargs) -> np.ndarray:
        """
        频域预处理: 在 ``fft`` 之后立即作用于谱, 产出**边界检测用的幅度谱**。

        分支 (按 ``PRE_DEAL_MOD_FREQ`` 的顺序施加: ``Slepian-Optimize`` → ``Smooth``):

        * ``Slepian-Optimize`` —— Slepian 多锥谱估计: 经 ``Utils.get_slepian()``
          统一入口取 ``(K, N)`` 的 Slepian 序列 (``slepian_nw`` 默认 3.0,
          ``slepian_k`` 默认 ``2*NW-1``), 各锥按其集中比加权平均:
          ``amp = sqrt(Σ_k ratio_k·|rfft(S·v_k)|² / Σk ratio_k)``。
          后端由 ``slepian_mod`` 决定 (None = ``ConstDefine.SLEPIAN_BACKEND``);
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
          (实测 multitone 稳定性 0.67 → 2.4 Hz), 故**默认不开启**;
          详见 docs/EWT_Native_Report.md §5.4。

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
            if mod == "Slepian-Optimize":
                limit = kw["slepian_max_samples"]
                if limit is not None and S.size > int(limit):
                    self._slepian_degraded = True
                else:
                    amp = self._slepian_multitaper(S, kw["slepian_nw"], kw["slepian_k"],
                                                   mod=kw["slepian_mod"])
            elif mod == "Smooth":
                width = kw["smooth_width"]
                if str(kw["smooth_kind"]).strip().lower() == "box":
                    amp = _moving_average(amp, int(width) | 1)
                else:
                    amp = _gaussian_smooth(amp, float(width))
        return amp

    def _slepian_multitaper(self, S: np.ndarray, nw, k, mod=None) -> np.ndarray:
        """
        Slepian 多锥幅度谱估计 (各锥按集中比加权); 返回与 ``rfft(S)`` 同长度。

        走 ``Utils.Slepian.slepian`` 统一入口: 后端选择/小 N 自动换后端/进程内缓存
        都由入口负责 (本方法不再直接依赖 scipy.signal)。
        """
        n = int(S.size)
        kk = int(2 * float(nw) - 1) if k is None else int(k)
        kk = max(1, min(kk, n))
        seqs, ratios = get_slepian().slepian(n, float(nw), kk, mod=mod,
                                            return_ratios=True, norm=2)
        w = np.maximum(np.asarray(ratios, dtype=np.float64), 0.0)
        spec = np.abs(self.fft.fft.rfft(S[None, :] * seqs, axis=1)) ** 2
        return np.sqrt(np.maximum((spec * w[:, None]).sum(axis=0) / max(float(w.sum()), 1e-300), 0.0))

    def pre_deal(self, S: np.ndarray, mod="None", **kwargs) -> np.ndarray:
        """
        时域预处理便捷入口 (保留旧名): 显式调用 :meth:`_time_domain_pre_deal`。

        频域分支 (``Slepian-Optimize`` / ``Smooth``) 需要频谱, 只能在
        ``__init__`` 的 ``pre_deal`` 中选择、由 ``decompose`` 在 ``fft`` 之后施加。

        ``mod`` 可以是单个名字或名字序列 (大小写不敏感, ``"None"`` = 不处理)。
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
        mod : {"maximum","max-min","scale-space","envelope"} | None
            策略名; None 时用 ``self.boundary_mod``。
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
        mod = self.boundary_mod if mod is None else mod
        if mod not in self.boundary_mods:
            raise ValueError(f"the mod of {mod} is not supported.")

        kw = dict(self.boundary_kwargs)
        kw.update(kwargs)
        amp = np.asarray(amp, dtype=np.float64)
        frequency = np.asarray(frequency, dtype=np.float64)
        f_lo, f_hi = float(frequency[0]), float(frequency[-1])

        match mod:
            case "maximum":
                interior = self._boundary_maximum(amp, frequency, **kw)
            case "max-min":
                interior = self._boundary_max_min(amp, frequency, **kw)
            case "scale-space":
                interior = self._boundary_scale_space(amp, frequency, **kw)
            case "envelope":
                interior = self._boundary_envelope(amp, frequency, **kw)
            case _:
                raise ValueError(f"the mod of {mod} is not supported.")

        interior = np.asarray(interior, dtype=np.float64).ravel()
        interior = interior[np.isfinite(interior)]
        interior = np.unique(interior)                          # 去重 + 升序
        interior = interior[(interior > f_lo) & (interior < f_hi)]
        return np.concatenate(([f_lo], interior, [f_hi]))

    def _peaks(self, amp: np.ndarray, **kw) -> np.ndarray:
        """谱峰检测 (Utils.Peaks; ``peak_distance`` 透传); 返回内部下标。"""
        fkw = {}
        if kw.get("peak_distance") is not None:
            fkw["distance"] = int(kw["peak_distance"])
        peaks, _ = self.find_peaks(amp, **fkw)
        idx = np.asarray(peaks, dtype=np.int64)
        return idx[(idx > 0) & (idx < amp.size)]

    # ------------------------------------------------------------------ #
    # 过渡带宽度 γ
    # ------------------------------------------------------------------ #
    def _resolve_band_factor(self, boundary, frequency) -> tuple[float, bool]:
        """
        解析 ``band_factor`` -> ``(γ, 是否自动)``。

        显式 float 原样返回 (不改动旧行为)。``"auto"`` 时按**边界间距反推**:
        算子 (:func:`Operator.Empirical_wavelet.empirical_wavelet_arr`) 把边界
        ``w`` 的过渡带放在 ``[(1−γ)w, (1+γ)w]``, 于是相邻边界 ``w_i < w_{i+1}``
        的过渡带不交叠要求

        .. math:: (1+γ)w_i \\le (1-γ)w_{i+1}
                  \\iff γ \\le \\frac{w_{i+1}-w_i}{w_{i+1}+w_i}

        上端边界还取同一形式的约束 ``(Nyquist−w_last)/(Nyquist+w_last)`` (即
        ewtpy 的 ``(π−w_last)/(π+w_last)`` 项; 这些比值与频率单位无关)。

        .. note::
           最高带的**滤波**上边界已被抬到 ``Nyquist/(1−γ)`` (见 ``decompose``),
           故上端约束如今不再是"防交叠"所必需, 而是用来把 γ 压小 —— γ 越小,
           过渡带越窄, 幅度带和 ``Σ H`` 偏离 1 的频段就越少, 带和误差也越小
           (实测: 只用内部约束的 γ 在 white/spike 上让带和误差从 0.108/0.127
           升到 0.157/0.160)。

        取所有约束的最小值再乘 ``(1−1/n_bins)`` 作为"严格小于"的安全余量, 得到
        **不交叠的最大 γ** —— 此时滤波器组满足紧框架 ``Σ_i H_i² ≡ 1``, 幅度带和
        误差最小; 而 γ 偏大 (旧默认 0.15 在边界聚拢时) 会让 2–3 条带的过渡带
        叠加, 同一频段被重复计入 (实测 ``Σ H²`` 最高 3.15)。

        无内部边界 (单频带) 时没有交叠约束可言, 退回固定 ``0.15``。
        """
        if self.band_factor != "auto":
            return float(self.band_factor), False

        nyq = float(frequency[-1])
        interior = np.asarray(boundary, dtype=np.float64)
        interior = interior[(interior > 0.0) & (interior < nyq)]
        if interior.size == 0:
            return 0.15, True                     # 单频带: 无约束, 用旧默认值

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

        # --- 1) 上包络: 过局部极大的样条 (Utils.Spline), 节点不足则用原谱 --- #
        pk = self._peaks(amp, **kw_peaks)
        env = np.asarray(amp, dtype=np.float64)
        if pk.size >= 2:
            try:
                grid = np.arange(n, dtype=np.float64)
                sp = self.spline(
                    pk.astype(np.float64), amp[pk],
                    spline_kind=kw["envelope_spline"], extrapolate=True,
                )
                env = np.asarray(sp(grid), dtype=np.float64)
            except Exception:
                env = np.asarray(amp, dtype=np.float64)

        # --- 2) 按 SNR 设定硬阈值并修平 --------------------------------- #
        # 阈值 = 噪底抬升 ``envelope_snr_db`` dB: 显著峰的判据是"比噪底高
        # snr_db". (旧实现取"比**峰值**低 snr_db 的幅度", 即把阈值抬到峰高的
        #  ~0.71 倍, 只剩全局最大峰能过线 ⇒ n_bands 退化为 1; 见 docs/EWT_Native_Report.md)
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
        return np.array(x, dtype=np.float64, copy=True)
    half = win // 2
    pad = np.pad(np.asarray(x, dtype=np.float64), (half, win - 1 - half), mode="edge")
    cum = np.concatenate(([0.0], np.cumsum(pad)))
    return (cum[win:] - cum[:-win]) / float(win)


def _window_vec(n: int, kind) -> np.ndarray:
    """时域窗向量: ``"hann"`` (别名 ``"hanning"``) / ``"hamming"``, 对称窗。"""
    key = str(kind).strip().lower()
    if key in ("hann", "hanning"):
        return np.hanning(int(n))
    if key == "hamming":
        return np.hamming(int(n))
    raise ValueError(f"window_kind must be 'hann' or 'hamming', got {kind!r}")


def _gaussian_smooth(x: np.ndarray, sigma: float) -> np.ndarray:
    """高斯平滑 (核半宽 3σ, 奇数长度核, ``mode="same"`` 不产生位移)。"""
    x = np.asarray(x, dtype=np.float64)
    sigma = float(sigma)
    if sigma <= 0.0 or x.size == 0:
        return x.copy()
    half = max(1, int(np.ceil(3.0 * sigma)))
    t = np.arange(-half, half + 1, dtype=np.float64)
    ker = np.exp(-0.5 * (t / sigma) ** 2)
    ker /= ker.sum()
    return np.convolve(x, ker, mode="same")


def _otsu_threshold(values: np.ndarray, bins: int = 64) -> float:
    """一维 Otsu 阈值 (直方图法); 常数序列返回该常数。"""
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return 0.0
    lo, hi = float(v.min()), float(v.max())
    if hi <= lo:
        return lo
    hist, edges = np.histogram(v, bins=max(2, int(bins)), range=(lo, hi))
    p = hist.astype(np.float64) / max(float(hist.sum()), 1.0)
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
    x = np.asarray(x, dtype=np.float64)
    if x.size < 3:
        return np.empty(0, dtype=np.int64)
    mask = (x[1:-1] < x[:-2]) & (x[1:-1] <= x[2:])
    return np.flatnonzero(mask) + 1


def _valley_between(amp, i0: int, i1: int, frequency, fallback: str = "midpoint") -> float:
    """
    在 ``(i0, i1)`` 开区间内取幅值最小处作边界 ("谷底分割")。

    段内极小落在区间端点 (段单调) 或段长不足时按 ``fallback`` 回退:
    ``"midpoint"`` 取两峰频率中点, ``"geometric"`` 取几何平均。
    """
    amp = np.asarray(amp, dtype=np.float64)
    frequency = np.asarray(frequency, dtype=np.float64)
    if i1 - i0 >= 3:
        seg = amp[i0 + 1:i1]
        j = int(np.argmin(seg))
        if 0 < j < seg.size - 1:
            return float(frequency[i0 + 1 + j])
    f0, f1 = float(frequency[i0]), float(frequency[i1])
    if fallback == "geometric" and f0 > 0.0 and f1 > 0.0:
        return float(np.sqrt(f0 * f1))
    return 0.5 * (f0 + f1)
