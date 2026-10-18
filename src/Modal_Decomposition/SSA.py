"""
奇异谱分析 (Singular Spectrum Analysis): 对信号轨迹矩阵做 SVD, 用窗 / 步长 / 秩规则
控制保留哪些奇异三元组并重建分量。属于时域嵌入类方法, 不使用时间轴。

嵌入与重建
----------
嵌入矩阵是经典 Hankel 矩阵的推广: 长度为 ``window_size`` (L) 的 patch 每隔
``stride`` (tau) 个样本取一个, 即

    X[i, j] = w[i] * S[i + stride * j],      i = 0..L-1,  j = 0..N-1,

其中 ``w`` 是可选的嵌入窗。``stride = 1`` 时与经典 Hankel 矩阵完全相同。
重建用加权重叠相加 (OLA), 窗同时充当合成权; 矩形窗下 OLA 等于经典 SSA 的对角平均
(``stride = 1`` 时逐位一致)。``stride > L`` 会留下永远不被覆盖的样本, 因此被拒绝。

秩选择 (去噪)
-------------
除了"返回全部基本分量"的经典模式 (``rank_rule=None``, 可完美重建), 秩规则决定保留
多少个前导奇异三元组:

* ``rank_rule="energy"``    —— 累积平方奇异能量达到总量 ``energy_frac``
  (默认 0.995) 所需的最小 k。
* ``rank_rule="svht"``      —— Gavish-Donoho 中位奇异值硬阈值: 保留
  sigma_i > omega(beta) * median(sigma), 其中
  omega(beta) = 0.56*beta^3 - 0.95*beta^2 + 1.82*beta + 1.43,
  beta = min(L, N) / max(L, N)。
* ``rank_rule="svht_clip"`` —— 带保护的规则: clip(SVHT, floor=2,
  cap=``svht_cap_frac`` 处的 energy 秩), 即保留数不少于 2、也不超过能量封顶给出的秩。

给出秩规则时, 返回的 IMFs 是保留的基本 (或合并) 分量, ``sum(IMFs, axis=0)`` 即去噪
信号, 被丢弃的分量不再出现 (``Res`` 恒为 None)。``rank_rule=None`` (默认) 时返回
全部分量, ``reconstruct()`` 可在机器精度内还原输入信号。

``"svht_clip"`` 的保留数下界是 floor (默认 2), 上界是 ``svht_cap_frac`` 处的 energy
秩 —— 它与 ``"energy"`` 使用的 ``energy_frac`` 是两个独立阈值。

窗函数 rect / hann / hamming 作用于嵌入矩阵的行, 在加权 OLA 合成时被除掉。
hann / hamming 在端点为 (近) 零, 因此加窗重建只在内部样本上精确还原信号。

References
----------
10.1016/j.mex.2020.101015
10.1109/TIT.2014.2312327   (Gavish-Donoho 最优硬阈值)

字符串形参约定
--------------
``window`` / ``rank_rule`` 按库内 ``Literal | Enum`` 约定同时接受字符串与 Enum 成员:
模块内声明 ``SSA_WINDOW_MODS_ARGS`` / ``SSA_RANK_MODS_ARGS`` (``Literal``) 与
``SSA_WINDOW_MODS`` / ``SSA_RANK_MODS`` (``Enum``), 入参两种写法都可以,
``param_check`` 经 ``Check.validate_mod`` 把实例属性规范化为 Enum 成员;
``rank_rule=None`` (经典模式) 保持 ``None``。
"""

import numpy as np
from dataclasses import dataclass
from enum import Enum
from typing import ClassVar, Literal, Sequence

from .Define import Config, Decomposer, DecompositionResult, check_output_memmap, finalize_output
from .Utils import Check, Check_Time_and_Signal

__all__ = [
    "SSA",
    "SSAConfig",
    "SSA_WINDOW_MODS",
    "SSA_WINDOW_MODS_ARGS",
    "SSA_RANK_MODS",
    "SSA_RANK_MODS_ARGS",
    "WINDOW_OPTIONS",
    "RANK_RULES",
    "window_vec",
    "svht_omega",
    "energy_rank",
    "svht_rank",
    "svht_clip_rank",
    "f_SSA",
]

SSA_WINDOW_MODS_ARGS = Literal["rect", "hann", "hamming"]
SSA_RANK_MODS_ARGS = Literal["energy", "svht", "svht_clip"]

class SSA_WINDOW_MODS(Enum):
    Rect = "rect"
    Hann = "hann"
    Hamming = "hamming"

class SSA_RANK_MODS(Enum):
    Energy = "energy"
    SVHT = "svht"
    SVHT_Clip = "svht_clip"

# 全部取值的字符串视图, 供外部按字面量迭代: RANK_RULES 首位 None 表示经典模式
WINDOW_OPTIONS: tuple[str, ...] = tuple(mod.value for mod in SSA_WINDOW_MODS)
RANK_RULES: tuple[str | None, ...] = (None, *(mod.value for mod in SSA_RANK_MODS))


"""
Config
"""
@dataclass(frozen=True, kw_only=True)
class SSAConfig(Config):
    """
    一次 SSA 运行的有效参数快照。
    """
    window_size: int | None
    stride: int
    groups: tuple[tuple[int, ...], ...] | None
    window: SSA_WINDOW_MODS
    rank_rule: SSA_RANK_MODS | None
    energy_frac: float
    svht_cap_frac: float
    dtype: type | np.dtype | None


"""
Class and Function of decomposer
"""
class SSA(Decomposer):
    name: ClassVar[str] = "SSA"

    def __init__(
        self,
        window_size: int | None = None,
        groups: Sequence[Sequence[int]] | None = None,
        stride: int = 1,
        window: SSA_WINDOW_MODS_ARGS | SSA_WINDOW_MODS = "rect",
        rank_rule: SSA_RANK_MODS_ARGS | SSA_RANK_MODS | None = None,
        energy_frac: float = 0.995,
        svht_cap_frac: float = 0.95,
        dtype: type | np.dtype | None = None,
        **kwargs,
    ):
        """
        Parameters
        ----------
        window_size : int | None
            嵌入 patch 长度 ``L``; None 时自动取 ``N // 3``。
        groups : Sequence[Sequence[int]] | None
            一起重建的奇异三元组分组; None 表示每个基本分量单独返回。
        stride : int
            相邻 patch 起点的间距 (``1`` = 经典 Hankel)。
        window : SSA_WINDOW_MODS_ARGS | SSA_WINDOW_MODS
            作用在嵌入矩阵行上的分析窗, 可给字面量名 (``"rect"`` / ``"hann"`` /
            ``"hamming"``, ``"none"`` 是 ``"rect"`` 的别名) 或 ``SSA_WINDOW_MODS``
            成员; 实例上存 Enum 成员。
        rank_rule : SSA_RANK_MODS_ARGS | SSA_RANK_MODS | None
            秩选择规则 (``"energy"`` / ``"svht"`` / ``"svht_clip"`` 或其 Enum
            成员); None 表示保留全部分量。
        energy_frac : float
            ``rank_rule="energy"`` 的累积平方奇异能量阈值。
        svht_cap_frac : float
            ``rank_rule="svht_clip"`` 的能量上限。
        dtype : type | np.dtype | None
            返回 IMFs 的工作精度。``None`` 跟随输入精度; 整型 / bool 输入升到
            float64。
        """
        self.window_size: int | None = window_size
        self.groups: Sequence[Sequence[int]] | None = groups
        self.stride: int = stride
        self.window: SSA_WINDOW_MODS | SSA_WINDOW_MODS_ARGS = window
        self.rank_rule: SSA_RANK_MODS | SSA_RANK_MODS_ARGS | None = rank_rule
        self.energy_frac: float = energy_frac
        self.svht_cap_frac: float = svht_cap_frac
        self.dtype: type | np.dtype | None = dtype

        unknown = set(kwargs) - {"safe", "copy"}
        if unknown:
            raise TypeError(f"unexpected keyword arguments: {unknown}")

        self.safe: str = kwargs.get("safe", "unknown")
        self.copy: bool = kwargs.get("copy", True)

        # 最近一次 decompose 的有效参数 (None = 尚未运行): config_generate 用它快照
        # 自动选出的窗长与被秩规则裁剪后真正保留的分组
        self._effective_window_size: int | None = None
        self._effective_groups: tuple[tuple[int, ...], ...] | None = None

        self.components_ = None
        self.sigma_ = None
        self.U_ = None
        self.V_ = None

        self.param_check()

    def param_check(self) -> None:
        if self.window_size is not None:
            if not isinstance(self.window_size, (int, np.integer)) \
                    or int(self.window_size) < 1:
                raise ValueError(
                    f"window_size must be None or a positive integer, "
                    f"got {self.window_size!r}"
                )
            self.window_size = int(self.window_size)

        if not isinstance(self.stride, (int, np.integer)) or int(self.stride) < 1:
            raise ValueError(f"stride must be a positive integer, got {self.stride!r}")
        self.stride = int(self.stride)

        # window: "none"/None 先规范化为 "rect" (名字大小写不敏感), 再由
        # Check.validate_mod 转成 Enum; 其余非法值在 validate_mod 处抛 ValueError。
        self.window = Check.validate_mod(_canon_window(self.window), SSA_WINDOW_MODS)

        # rank_rule: None 表示经典模式, 保持 None; 非 None 一律转成 Enum
        if self.rank_rule is not None:
            self.rank_rule = Check.validate_mod(self.rank_rule, SSA_RANK_MODS)

        if not isinstance(self.energy_frac, (int, float)) \
                or not (0.0 < self.energy_frac <= 1.0):
            raise ValueError(f"energy_frac must be in (0, 1], got {self.energy_frac!r}")
        if not isinstance(self.svht_cap_frac, (int, float)) or not (
            0.0 < self.svht_cap_frac <= 1.0
        ):
            raise ValueError(
                f"svht_cap_frac must be in (0, 1], got {self.svht_cap_frac!r}"
            )
        self.energy_frac = float(self.energy_frac)
        self.svht_cap_frac = float(self.svht_cap_frac)

        if self.dtype is not None and np.dtype(self.dtype).kind != "f":
            raise ValueError("the `dtype` must be a float numpy dtype or None.")

        if not isinstance(self.copy, bool):
            raise TypeError("the `copy` must be bool")

        # 只做合法性校验, 取值仍按 str 保存, 由 Check_Time_and_Signal 统一解释
        Check.validate_safe(self.safe)

    def config_generate(self) -> SSAConfig:
        """
        本次运行的有效参数快照。

        首次 ``decompose`` 之前报构造参数 (``window_size`` 可能仍是 None = 自动);
        之后报上一次运行的有效值, 即解析出的 patch 长度与秩规则真正保留的分组。
        ``window`` / ``rank_rule`` 是 Enum 成员 (经典模式下 ``rank_rule`` 为 None)。
        """
        if self._effective_window_size is None:
            window_size = self.window_size
            groups = (
                None
                if self.groups is None
                else tuple(tuple(int(i) for i in g) for g in self.groups)
            )
        else:
            window_size = self._effective_window_size
            groups = self._effective_groups

        return SSAConfig(
            window_size=window_size,
            stride=self.stride,
            groups=groups,
            window=self.window,
            rank_rule=self.rank_rule,
            energy_frac=self.energy_frac,
            svht_cap_frac=self.svht_cap_frac,
            dtype=self.dtype,
        )

    # ------------------------------------------------------------------ #
    # 分解主流程
    # ------------------------------------------------------------------ #
    def decompose(self, S, T=None, output_memmap: bool = False) -> DecompositionResult:
        """
        把信号分解成被保留的重建分量。

        Returns
        -------
        DecompositionResult
            ``IMFs`` 形状 (K, N), K 是保留的分量数 (秩规则选出的, 或
            ``rank_rule=None`` 时的全部); ``Res`` 恒为 None; ``config`` 是本次有效
            参数的快照。``info`` 含 ``k`` (保留数) / ``n_padded`` (右端延拓后的
            长度) / ``window`` / ``rank_rule``, 后两者是字符串字面量。
        """
        # 形参校验先于任何计算: 传错类型应立即失败, 而不是跑完整个分解再报错
        check_output_memmap(output_memmap)
        # SSA 是纯时域嵌入方法, 不用时间轴; 调用方显式给了 T 才校验它。need_T=False
        # 省掉 arange(N) 这一份与信号等长的临时数组。
        S, N = Check_Time_and_Signal(S, T=T, ndim={1}, method=self.name,
                                     dtype=self.dtype, need_T=False,
                                     safe=self.safe, copy=self.copy)

        # int/bool 的原地运算会被静默截断, f16 精度不足以承载 SVD: 统一升到 float64。
        # f32/f64 保留原精度 (结果 cast 回 S.dtype)。
        if not (S.dtype.kind == "f" and S.dtype.itemsize >= 4):
            S = Check.to_dtype(S, np.dtype(np.float64))

        # 每次调用都从干净状态开始
        self.components_ = None
        self.sigma_ = None
        self.U_ = None
        self.V_ = None
        self._effective_window_size = None
        self._effective_groups = None

        if self.window_size is None:
            L = max(N // 3, 1)
        else:
            L = int(self.window_size)

        # 边界条件: 1 <= L < N, patch 长度只要求小于信号长度。
        # N == 1 到不了这里: 共享输入层已拒绝会被压成 0 维的信号。
        if L < 1 or L >= N:
            raise ValueError(
                f"window_size (L={L}) must satisfy 1 <= L < N (N={N})"
            )

        s = self.stride
        if L < s:
            raise ValueError(
                f"window_size (L={L}) must be >= stride ({s}); "
                f"otherwise some samples are never covered"
            )

        # 右端边缘延拓: 补 pad = stride - (N - L) % stride 个样本, 使等距窗口覆盖全部
        # 采样点; stride == 1 时补 0 个, 即经典 Hankel 矩阵。
        pad = (-(N - L)) % s
        if pad:
            S_pad = np.pad(S, (0, int(pad)), mode="edge")
        else:
            S_pad = S
        N_pad = N + pad

        win = window_vec(L, self.window)
        x = self.hankel(S_pad, L, s)
        X = np.asarray(x, dtype=np.float64)
        if self.window is not SSA_WINDOW_MODS.Rect:
            X = X * win[:, None]

        U, sv, Vt = np.linalg.svd(X, full_matrices=False)

        # --- 秩规则: 保留分量的数量 k ------------------------------------ #
        K_cols = X.shape[1]
        match self.rank_rule:
            case None:
                k = len(sv)
            case SSA_RANK_MODS.Energy:
                k = energy_rank(sv, self.energy_frac)
            case SSA_RANK_MODS.SVHT:
                beta = min(X.shape) / max(X.shape)
                k = svht_rank(sv, beta)
            case _:  # SSA_RANK_MODS.SVHT_Clip
                beta = min(X.shape) / max(X.shape)
                k = svht_clip_rank(sv, beta, floor=2, cap_frac=self.svht_cap_frac)
        k = int(min(max(k, 0), len(sv)))

        # --- 分组: 与保留的秩取交集, 交集为空的组丢弃 --------------------- #
        if self.groups is None:
            groups_eff: list[tuple[int, ...]] = [(i,) for i in range(k)]
        else:
            groups_eff = []
            for g in self.groups:
                members = tuple(int(i) for i in g)
                if not members:
                    continue
                if any(i < 0 or i >= len(sv) for i in members):
                    raise ValueError(
                        f"group {members} contains indices outside "
                        f"[0, {len(sv)}), the number of singular values"
                    )
                kept = tuple(i for i in members if i < k)
                if kept:
                    groups_eff.append(kept)

        # --- 逐组重建: 加权 OLA / 对角平均 -------------------------------- #
        RCs = []
        for group in groups_eff:
            # 保持原有的乘法顺序: stride == 1 时与经典实现逐位相同
            X_rec = U[:, group] @ (sv[group, None] * Vt[group, :])

            if self.window is SSA_WINDOW_MODS.Rect:
                rc = self.diagonal_average_fast(X_rec, L, K_cols, s)[:N]
            else:
                rc = self._ola_windowed(X_rec, win, L, s, K_cols)[:N]
            RCs.append(rc)

        # 输出精度跟随输入 (dtype=None) 或调用方指定的 dtype; 内部计算仍是 float64
        IMFs: np.ndarray = (
            np.asarray(RCs, dtype=S.dtype) if RCs else np.empty((0, N), dtype=S.dtype)
        )

        self.components_ = IMFs
        self.sigma_ = sv
        self.U_ = U
        self.V_ = Vt

        groups_snapshot: tuple[tuple[int, ...], ...] | None
        if self.groups is None and self.rank_rule is None:
            groups_snapshot = None  # 纯基本分量分解, 不做秩裁剪
        else:
            groups_snapshot = (
                tuple(tuple(int(i) for i in g) for g in groups_eff)
                if groups_eff
                else None
            )

        # config_generate() 快照本次有效参数, 不是构造参数
        self._effective_window_size = L
        self._effective_groups = groups_snapshot

        return finalize_output(
            DecompositionResult(
                IMFs,
                None,
                {
                    "k": k,
                    "n_padded": int(N_pad),
                    # info 是给人看的运行日志: 放字符串而不是 Enum 成员, 便于直接
                    # 打印与序列化
                    "window": self.window.value,
                    "rank_rule": (
                        None if self.rank_rule is None else self.rank_rule.value
                    ),
                },
                self.config_generate(),
            ),
            output_memmap,
        )

    # ------------------------------------------------------------------ #
    # 嵌入与重建原语
    # ------------------------------------------------------------------ #
    def hankel(self, S, L: int, stride: int = 1):
        """
        用 numpy 滑窗视图构造轨迹矩阵: 每隔 ``stride`` 个样本取一个长 ``L`` 的
        patch, 即 ``X[i, j] = S[i + stride * j]``, 返回 (L, K) 形状。

        ``stride = 1`` 即经典 Hankel 矩阵。分析窗不在这里施加 (见 ``decompose``)。
        返回的是滑窗视图的转置, 不复制数据。
        """
        S = np.asarray(S)
        W = np.lib.stride_tricks.sliding_window_view(S, L)[:: int(stride)]
        return W.T

    @staticmethod
    def diagonal_average_fast(X_rec, L: int, K: int, stride: int = 1) -> np.ndarray:
        """
        重建轨迹矩阵的等距对角平均: 对固定的 ``i + stride * j`` 求 ``X[i, j]`` 的
        平均。``stride = 1`` 即经典反对角平均 (Hankel 化)。

        返回长度 ``L + stride * (K - 1)``; 没有样本落到的位置计数按 1 处理, 不出 NaN。
        """
        n_idx = (np.arange(L)[:, None] + int(stride) * np.arange(K)[None, :]).ravel()
        values = np.asarray(X_rec).ravel()

        Sum = np.bincount(n_idx, weights=values, minlength=L + int(stride) * (K - 1))
        counts = np.bincount(n_idx, minlength=L + int(stride) * (K - 1))
        counts[counts == 0] = 1

        return Sum / counts

    @staticmethod
    def _ola_windowed(
        X_rec: np.ndarray, win: np.ndarray, L: int, stride: int, K: int
    ) -> np.ndarray:
        """
        重建 patch 矩阵的加权重叠相加: 第 ``j`` 列把 ``win`` 加权后的样本累加到
        ``[j*stride, j*stride+L)``, 累加和再除以累加的窗权。

        累加窗权为 0 的样本 (只可能出现在零端点窗的边界) 输出 0。矩形窗下与
        :meth:`diagonal_average_fast` 一致; ``stride <= L`` 保证每个样本都被覆盖。
        """
        padded_len = L + int(stride) * (K - 1)
        n_idx = (np.arange(L)[:, None] + int(stride) * np.arange(K)[None, :]).ravel()
        # 按 C 序展平后 (i, j) 的下标是 i*K + j, 故元素 (i, j) 的窗权是 win[i]:
        # 用 repeat 而不是 tile。
        win_repeat = np.repeat(win, K)

        Sum = np.bincount(n_idx, weights=np.asarray(X_rec).ravel(),
                          minlength=padded_len)
        Wsum = np.bincount(n_idx, weights=win_repeat, minlength=padded_len)
        out = np.zeros(padded_len, dtype=np.float64)
        np.divide(Sum, Wsum, out=out, where=Wsum > 0)
        return out

"""
other function for module
"""
def _canon_window(window: SSA_WINDOW_MODS_ARGS | SSA_WINDOW_MODS | None) -> str:
    """
    把窗名规范成小写 ``str``。

    ``None`` 与 ``"none"`` 都表示 ``"rect"``; Enum 成员取其 ``.value``。
    无法识别的名字原样返回, 由调用方的 ``Check.validate_mod`` 报错。
    """
    if isinstance(window, SSA_WINDOW_MODS):
        return window.value
    if window is None:
        return "rect"
    key = str(window).strip().lower()
    if key == "none":
        return "rect"
    return key


def window_vec(L: int, window: SSA_WINDOW_MODS_ARGS | SSA_WINDOW_MODS) -> np.ndarray:
    """
    返回长度为 ``L`` 的分析窗 (``"rect"`` 即全 1)。

    字面量名 (``"rect"`` / ``"hann"`` / ``"hamming"``, ``"none"`` 是 ``"rect"``
    的别名) 与 ``SSA_WINDOW_MODS`` 成员都接受, 参数在这里一并校验。

    端点: hann / hamming 在两端为 (近) 零, 加窗 OLA 重建的首末样本合成权也因此
    (近) 为零, 加权 OLA 只在内部样本上精确还原信号。
    """
    mod = Check.validate_mod(_canon_window(window), SSA_WINDOW_MODS)
    match mod:
        case SSA_WINDOW_MODS.Hann:
            return np.hanning(L)
        case SSA_WINDOW_MODS.Hamming:
            return np.hamming(L)
        case _:  # SSA_WINDOW_MODS.Rect
            return np.ones(L, dtype=np.float64)


def svht_omega(beta: float) -> float:
    """m x n 矩阵的 Gavish-Donoho 中位值系数, beta = min(m,n)/max(m,n)。"""
    return 0.56 * beta ** 3 - 0.95 * beta ** 2 + 1.82 * beta + 1.43


def energy_rank(sv: np.ndarray, fraction: float) -> int:
    """
    累积平方奇异能量达到 ``fraction`` 所需的最小 k。

    零能量 (或退化) 谱返回 ``len(sv)``。
    """
    total = float(np.sum(sv ** 2))
    if total <= 0.0:
        return len(sv)
    k = int(np.argmax(np.cumsum(sv ** 2) >= fraction * total)) + 1
    return min(k, len(sv))


def svht_rank(sv: np.ndarray, beta: float) -> int:
    """Gavish-Donoho 中位硬阈值秩 (不加下限/上限)。"""
    if sv.size == 0:
        return 0
    tau = svht_omega(beta) * float(np.median(sv))
    return int(np.sum(sv > tau))


def svht_clip_rank(sv: np.ndarray, beta: float,
                   floor: int = 2, cap_frac: float = 0.95) -> int:
    """
    带保护的 SVHT 规则: 把 SVHT 秩夹在 [floor, cap_frac 处的 energy 秩] 内。
    """
    k_sv = svht_rank(sv, beta)
    k_cap = energy_rank(sv, cap_frac)
    return int(min(max(k_sv, floor), k_cap))


"""
other function for module
"""
def f_SSA(S: list | np.ndarray, T=None, output_memmap: bool = False, *, window_size: int | None = None, groups: Sequence[Sequence[int]] | None = None, stride: int = 1, window: SSA_WINDOW_MODS_ARGS | SSA_WINDOW_MODS = "rect", rank_rule: SSA_RANK_MODS_ARGS | SSA_RANK_MODS | None = None, energy_frac: float = 0.995, svht_cap_frac: float = 0.95, dtype: type | np.dtype | None = None, **kwargs) -> DecompositionResult:
    ssa = SSA(
        window_size=window_size,
        groups=groups,
        stride=stride,
        window=window,
        rank_rule=rank_rule,
        energy_frac=energy_frac,
        svht_cap_frac=svht_cap_frac,
        dtype=dtype,
        **kwargs
    )

    return ssa.decompose(S, T, output_memmap=output_memmap)
