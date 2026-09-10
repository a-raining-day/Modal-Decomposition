"""
Empirical Mode Decomposition —— 原生实现 (正式 EMD)。

EMD 已由自研实现转正: 本模块取代原 PyEMD 包装版, 注册键 ``"EMD"``
(``Class.EMD`` / ``Function.EMD``)。背景: 原包装版首次使用需 import PyEMD
(约 1.1-1.4 s), 且筛分耗时约 75% 在包络样条; 原生实现去掉 EMD 本体的第三方
分解依赖, 保留向量化极值检测与 dtype 保持。PyEMD 仍用于 EEMD / CEEMDAN 的
集合扩展与基准对比 (见 ``tests/comparison/bench_emd_new.py`` 与
``docs/EMD_vs_EMD_new_Performance_Report.md``)。

工具取用 (与 Utils 基础设施一致)
--------------------------------
- 极值检测经 ``Utils.get_peaks()`` 取 ``Peaks.find_peaks`` (scipy/numpy/
  numba 三后端, 统一返回 ``(indices, properties)``);
- 包络样条经 ``Utils.get_spline()`` 取 ``Spline.spline``: ``spline_kind``
  形参直接采用 Spline 的 canonical 后端名 ("CubicSpline"/"PCHIP"), "linear"
  特例直接用 ``np.interp``; 惰性 import 并注册进进程级 import 缓存;
- 端点镜像经 ``Utils.get_mirror()`` 取 ``Mirror.mirror_extrema`` (nbsym 语义);
- 单调性停止判据用 ``Utils.is_monotonic``。

算法 (经典 Huang 1998 筛分)
---------------------------
1. 向量化找局部极大/极小 (平台取右边缘, 与 Utils.Peaks 同规则);
2. 端点以 ``nbsym`` 个镜像极值外延, 三次样条 (或 PCHIP/线性) 插值上下
   包络, 局部均值 ``m = (up + low) / 2``;
3. ``h <- h - m``, 直到 Cauchy 型判据 ``sum(m^2)/sum(h_prev^2) < sd_thr``
   (``faster=True``) —— 或该判据与经典窄带平衡 ``|zc - ext| <= 1`` 同时
   满足 (``faster=False``, 默认), 或达到 ``max_iter``;
4. 每得一个 IMF 即从残差中减去; 残差极值不足或单调时停止, 余量即 ``Res``。

``faster`` 分支语义
------------------
* ``faster=True``: 旧高速档 —— 能量型 Cauchy SD 收敛即停, 迭代预算最小;
  带噪/弱分量场景行的 |zc−ext| 平衡不保证 (质量参考: 上一轮机制分析报告
  ``docs/EMD_Quality_Gap_and_Optimization.md``)。
* ``faster=False`` (默认, 质量档): 在 SD 收敛之外还要求当前 h 满足经典
  IMF 必要条件 ``|zc − ext| <= 1`` (zc 用与 PyEMD 一致的过零计数口径),
  不满足则继续筛分直到 ``max_iter``。对已窄带的行零额外迭代 (干净信号
  代价≈0); 对噪声/长信号迭代次数上升, 换来行级纯度 (corr/valid) 对标
  PyEMD。``sd_thr`` / ``max_iter`` / ``spline_kind`` 两档通用、语义不变。

返回契约
--------
``DecompositionResult`` (IMFs (K,N), Res, info, config), 与全库一致:
- dtype: float32/float64 全程保持原精度; float16 与整数/布尔输入提升为
  float64 计算;
- 重构精确: ``IMFs.sum(axis=0) + Res == S`` (残差是逐次减法余量)。

References
----------
10.1098/rspa.1998.0193
"""

import numpy as np
from dataclasses import dataclass
from typing import ClassVar, Literal

from .Base import Config, Decomposer, DecompositionResult
from ._Registry import register_class
from .Utils import (
    Check_Time_and_Signal,
    get_mirror,
    get_peaks,
    get_spline,
    is_monotonic,
)

__all__ = ["EMD", "EMDConfig"]

#: 包络后端 (与形参 spline_kind 的 Literal 一致; "linear" = np.interp 特例)。
_EMD_KINDS = ("CubicSpline", "PCHIP", "linear")
_PEAKS_MODS = ("scipy", "numpy", "numba")


@dataclass(frozen=True, kw_only=True)
class EMDConfig(Config):
    """
    Effective parameters of an EMD run.
    """
    nbsym: int
    spline_kind: str
    max_imf: int
    max_iter: int
    sd_thr: float
    dtype: np.dtype | None
    compile: bool
    find_peaks_mod: str
    faster: bool


def _work_dtype(dtype) -> np.dtype:
    """float32/float64 保持原精度; 其余 (float16/int/bool) 提升为 float64。"""
    if dtype.kind == "f" and dtype.itemsize >= 4:
        return dtype
    return np.dtype(np.float64)


@register_class("EMD")
class EMD(Decomposer):
    """
    Empirical Mode Decomposition (native sifting implementation).

    Decomposes a 1-D signal into intrinsic mode functions (IMFs) and a
    residual via the classical Huang-1998 sifting process:

    1. local extrema of the current signal (``Utils.Peaks``);
    2. upper/lower envelopes through mirror-extended extrema — ``nbsym``
       endpoint mirroring via ``Utils.Mirror``, envelopes via
       ``Utils.Spline`` (``spline_kind`` is the canonical backend name;
       ``"linear"`` evaluates ``np.interp`` directly);
    3. subtract the local-mean envelope repeatedly until the Cauchy-type
       criterion ``sum(m^2)/sum(h_prev^2) < sd_thr`` (``faster=True``) or
       until that criterion *and* the classic narrowband balance
       ``|zc - ext| <= 1`` hold (``faster=False``, default), or
       ``max_iter``;
    4. the sifted component is one IMF; subtract it from the residual and
       repeat until the residual is monotonic or has too few extrema.

    ``faster`` branch (两档停止策略)
    -------------------------------
    * ``faster=True`` — legacy fast branch: the energy Cauchy SD criterion
      alone stops sifting (minimal iteration budget; on noisy / long signals
      the rows are not guaranteed to satisfy the zero-crossing/extrema
      balance).
    * ``faster=False`` (default, quality branch): after the SD criterion is
      met the current ``h`` must also satisfy ``|zc - ext| <= 1``
      (zero crossings counted with the PyEMD-consistent ``indzer``
      convention; extrema counted with the MD-native ``Utils.Peaks`` rule
      ``S[i-1] < S[i] >= S[i+1]``); otherwise sifting continues up to
      ``max_iter``. Rows already narrowband exit immediately (clean signals
      cost nothing extra); noisy / long signals spend more iterations and
      gain row-level purity (tone capture and IMF-validity comparable with
      PyEMD). On plateau/quantized signals the two counting conventions
      diverge (MD counts every rising edge into a flat run, PyEMD counts
      plateau midpoints with rise-then-fall — experiment E-H in
      ``docs/EMD_vs_PyEMD_Detailed_Comparison.md``); the gate is aligned to
      the MD-native rule there, like the rest of the native engine.

    Effect (效果要点)
    -----------------
    * IMFs are extracted from high frequency to low frequency; the first IMF
      tracks the dominant high-frequency component of the signal;
    * exact reconstruction — ``IMFs.sum(axis=0) + Res == S`` (the residual is
      the successive-subtraction remainder), so ``reconstruct()`` reproduces
      the input to machine precision (both branches);
    * float32/float64 inputs keep their precision, float16 / integer / bool
      inputs are promoted to float64;
    * quality branch (default, ``faster=False``) is benchmarked in
      ``docs/EMD_faster_Branch_Comparison_Report.md`` against the fast
      branch, PyEMD and PySDKit (mode capture, row validity, runtime).

    Use as ``Class.EMD(**params).decompose(S, T)`` or ``Function.EMD(S, T,
    **params)``; a frozen ``EMDConfig`` snapshot of the effective parameters
    is attached to every ``DecompositionResult``.
    """

    name: ClassVar[str] = "EMD"

    def __init__(
        self,
        nbsym: int = 2,
        spline_kind: Literal["CubicSpline", "PCHIP", "linear"] = "CubicSpline",
        max_imf: int = -1,
        max_iter: int = 100,
        sd_thr: float = 0.01,
        dtype: np.dtype = None,
        compile: bool = False,
        find_peaks_mod: Literal["scipy", "numpy", "numba"] = "numpy",
        faster: bool = False,
        config: EMDConfig = None,
    ) -> None:
        """
        Parameters
        ----------
        nbsym : int
            Number of extrema mirrored at each boundary (classic endpoint
            treatment). 0 disables mirroring.
        spline_kind : {"CubicSpline", "PCHIP", "linear"}
            Envelope backend — the canonical spline kinds of
            ``Utils.Spline``, passed straight through; "linear" evaluates
            ``np.interp`` directly (fewer than 3 extrema also fall back to
            linear interpolation).
        max_imf : int
            Maximum number of IMFs; -1 decomposes completely.
        max_iter : int
            Sifting iterations per IMF (Cauchy-type SD guard).
        sd_thr : float
            SD stopping threshold ``sum(m^2)/sum(h_prev^2) < sd_thr``.
        dtype : np.dtype | None
            Working dtype. None: float32/float64 preserved, others promoted
            to float64.
        compile : bool
            Reserved flag (kept for future JIT backends; currently unused).
        find_peaks_mod : {"scipy", "numpy", "numba"}
            Peak-detection backend of ``Utils.Peaks``.
        faster : bool
            Stopping-policy branch. ``True`` keeps the legacy fast branch
            (Cauchy-type SD convergence stops sifting); ``False`` (default)
            additionally requires the classic narrowband balance
            ``|zc - ext| <= 1`` before a sifted row is accepted — higher
            row quality at the price of extra iterations on noisy / long
            signals (clean signals: no extra cost).
        config : EMDConfig, optional
            Frozen parameter snapshot; when given it overrides every other
            argument (``self.config`` stays that snapshot).

        Notes
        -----
        Defaults: CubicSpline envelope + ``sd_thr=0.01`` + ``faster=False``
        (quality branch). The envelope choice keeps the same spline
        semantics as the external references (PyEMD/PySDKit); the SD
        threshold is set by the mode-level validation in
        ``docs/EMD_Validation_and_Comparison_Report.md``: the loose sweep
        optimum (0.3) stops sifting before IMFs satisfy the classic
        zero-crossing/extrema balance and splits tones across adjacent rows
        (single-row tone capture ~0.65-0.87), while ``sd_thr=0.01`` restores
        single-row capture (~0.94) at a still-small cost (CubicSpline
        ~14 ms, linear ~4 ms at n=4096 vs ~56 ms for PyEMD). ``faster=False``
        adds the narrowband gate measured in
        ``docs/EMD_Quality_Gap_and_Optimization.md`` (variant V4): row-level
        validity on noise reaches the PyEMD level while remaining faster
        than PyEMD; ``faster=True`` reproduces the pre-``faster`` behaviour
        (see the four-way benchmark in
        ``docs/EMD_faster_Branch_Comparison_Report.md``).
        """
        self.config = config

        if isinstance(self.config, Config):
            self.nbsym = self.config.nbsym
            self.spline_kind = self.config.spline_kind
            self.max_imf = self.config.max_imf
            self.max_iter = self.config.max_iter
            self.sd_thr = self.config.sd_thr
            self.dtype = self.config.dtype
            self.compile = self.config.compile
            self.find_peaks_mod = self.config.find_peaks_mod
            self.faster = self.config.faster
        else:
            self.nbsym = nbsym
            self.spline_kind = spline_kind
            self.max_imf = max_imf
            self.max_iter = max_iter
            self.sd_thr = sd_thr
            self.dtype = dtype
            self.compile = compile
            self.find_peaks_mod = find_peaks_mod
            self.faster = faster

        # --- 参数校验 --------------------------------------------------- #
        if not isinstance(self.nbsym, int) or self.nbsym < 0:
            raise ValueError(f"nbsym must be an int >= 0, got {self.nbsym!r}")
        if self.spline_kind not in _EMD_KINDS:
            raise ValueError(
                f"spline_kind must be one of {_EMD_KINDS}, got {self.spline_kind!r}"
            )
        if not isinstance(self.max_imf, int) or (self.max_imf != -1 and self.max_imf < 1):
            raise ValueError(f"max_imf must be -1 or >= 1, got {self.max_imf!r}")
        if not isinstance(self.max_iter, int) or self.max_iter < 1:
            raise ValueError(f"max_iter must be an int >= 1, got {self.max_iter!r}")
        if not (0.0 < self.sd_thr <= 1.0):
            raise ValueError(f"sd_thr must be in (0, 1], got {self.sd_thr!r}")
        if self.dtype is not None and np.dtype(self.dtype).kind != "f":
            raise ValueError(
                f"dtype must be a floating numpy dtype or None, got {self.dtype!r}"
            )
        if self.find_peaks_mod not in _PEAKS_MODS:
            raise ValueError(
                f"find_peaks_mod must be one of {_PEAKS_MODS}, got {self.find_peaks_mod!r}"
            )
        if not isinstance(self.faster, bool):
            raise ValueError(f"faster must be a bool, got {self.faster!r}")

        if self.config is None:
            self.config = EMDConfig(
                nbsym=self.nbsym,
                spline_kind=self.spline_kind,
                max_imf=self.max_imf,
                max_iter=self.max_iter,
                sd_thr=self.sd_thr,
                dtype=np.dtype(self.dtype) if self.dtype is not None else None,
                compile=self.compile,
                find_peaks_mod=self.find_peaks_mod,
                faster=self.faster,
            )

        # 工具惰性取用 (Utils getter 首次访问时 import + 注册进进程级缓存)。
        _peaks = get_peaks()
        self.find_peaks = _peaks.find_peaks
        self.spline = get_spline().spline
        self.mirror_extrema = get_mirror().mirror_extrema

    def __call__(self, S, T=None) -> DecompositionResult:
        return self.decompose(S, T)

    def decompose(self, S, T=None) -> DecompositionResult:
        """
        Decompose the signal into IMFs and a residual (native sifting).

        Returns a ``DecompositionResult`` with ``IMFs`` of shape (K, N) and
        the exact reconstruction ``sum(IMFs, axis=0) + Res == S``.
        """
        S, T, N = Check_Time_and_Signal(S, T, ndim={1}, method=self.name)

        # dtype 策略: 显式 dtype 优先; 否则 f32/f64 保持, 其余提升 f64。
        target = np.dtype(self.dtype) if self.dtype is not None else _work_dtype(S.dtype)
        S = np.asarray(S).astype(target, copy=False)

        # 包络在索引空间构造与求值 (经典 EMD 语义; T 仅用于校验)。
        grid = np.arange(N, dtype=np.float64)

        Res = S.copy()
        IMFs: list = []
        iterations: list = []
        final_sds: list = []

        while self.max_imf == -1 or len(IMFs) < self.max_imf:
            if is_monotonic(Res):
                break
            up_idx, _ = self.find_peaks(Res, mod=self.find_peaks_mod)
            dn_idx, _ = self.find_peaks(-Res, mod=self.find_peaks_mod)
            if up_idx.size < 2 or dn_idx.size < 2:
                break

            imf, it, sd = self._sift(Res, up_idx, dn_idx, grid)
            IMFs.append(imf)
            iterations.append(it)
            final_sds.append(sd)
            Res = Res - imf

        IMFs_arr = (
            np.array(IMFs, dtype=target)
            if IMFs
            else np.empty((0, N), dtype=target)
        )

        return DecompositionResult(
            IMFs_arr,
            Res,
            {"iterations": iterations, "final_sd": final_sds},
            self.config,
        )

    # ------------------------------------------------------------------ #
    # 筛分内循环 (首次迭代复用 decompose 预检过的极值, 省一次全数组扫描)
    # ------------------------------------------------------------------ #
    def _sift(self, h, up_idx, dn_idx, grid):
        last_sd = 0.0
        iters = 0

        for it in range(self.max_iter):
            if it > 0:
                up_idx, _ = self.find_peaks(h, mod=self.find_peaks_mod)
                dn_idx, _ = self.find_peaks(-h, mod=self.find_peaks_mod)
                if up_idx.size < 2 or dn_idx.size < 2:
                    break

            up_pos, up_vals = self.mirror_extrema(up_idx, h[up_idx], self.nbsym)
            dn_pos, dn_vals = self.mirror_extrema(dn_idx, h[dn_idx], self.nbsym)
            up = self._envelope(up_pos, up_vals, grid)
            dn = self._envelope(dn_pos, dn_vals, grid)
            mean = (up + dn) * 0.5

            # Cauchy 型判据: sd = sum(mean^2) / sum(h_prev^2)。
            prev_energy = float(np.dot(h, h))
            if prev_energy == 0.0:
                break  # 全零: 已收敛

            h_new = h - mean
            last_sd = float(np.dot(mean, mean)) / prev_energy
            h = h_new
            iters = it + 1

            if last_sd < self.sd_thr:
                # faster=False (默认质量档): 收敛之上还需窄带平衡才放行;
                # faster=True (高速档): 能量收敛即停 (旧行为)。
                if self.faster or self._narrowband_ok(h):
                    break

        return h, iters, last_sd

    def _narrowband_ok(self, h) -> bool:
        """经典 IMF 必要条件: |过零数 − 极值数| ≤ 1。

        过零用 PyEMD ``indzer`` 同口径 (严格符号积 <0 + 零点段中点); 极值用
        MD 原生 ``Utils.Peaks`` 规则 (``S[i-1] < S[i] >= S[i+1]``, 平台取升沿
        右缘)。无平台信号上与 PyEMD 的 f2 (|ext−zc|<2) 等价; 平台/量化信号上
        两引擎计数契约不同 (见 ``docs/EMD_vs_PyEMD_Detailed_Comparison.md``
        实验 E-H), 本开关在库内对齐 MD 原生规则。
        """
        zc = _zero_cross_count(h)
        m_idx, _ = self.find_peaks(h, mod=self.find_peaks_mod)
        n_idx, _ = self.find_peaks(-h, mod=self.find_peaks_mod)
        return abs(zc - (m_idx.size + n_idx.size)) <= 1

    def _envelope(self, positions, values, grid) -> np.ndarray:
        """在索引网格 [0, N) 上按镜像后的极值插值包络 (回投工作 dtype)。"""
        if self.spline_kind == "linear":
            out = np.interp(grid, positions, values)
        else:
            # spline_kind 即 Utils.Spline 的 canonical 后端名 (Literal 直通)
            sp = self.spline(positions, values, spline_kind=self.spline_kind)
            out = sp(grid)
        return np.asarray(out, dtype=values.dtype)


def _zero_cross_count(x: np.ndarray) -> int:
    """过零计数 (PyEMD ``indzer`` 同口径): 严格符号积 <0 加零点段中点。"""
    s1, s2 = x[:-1], x[1:]
    n = int(np.sum(s1 * s2 < 0))
    if np.any(x == 0):
        indz = np.nonzero(x == 0)[0]
        if np.any(np.diff(indz) == 1):  # 存在零点段才做段合并 (PyEMD 同款条件)
            z = x == 0
            dz = np.diff(np.concatenate(([0], z, [0])))
            debz = np.nonzero(dz == 1)[0]
            n += int(debz.size)
    return n
