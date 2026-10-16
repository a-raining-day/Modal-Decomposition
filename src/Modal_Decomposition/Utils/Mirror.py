"""
Extrema endpoint mirror extension (端点镜像外拓, 全库共享)。

模态分解 (EMD / LMD / 极值样条包络等) 都需要在信号端点外**外延样本或极值**,
再对延拓后的序列做样条插值, 以避免端点塌陷 / 边界伪振荡。本模块统一提供
:func:`mirror`, 覆盖五种语义:

- ``mirror`` (端点镜像): 两端各镜像 ``nbsym`` 个样本, 反射轴是端点内侧的第一个
  样本 (与 ``np.pad(mode="reflect")`` 同义), 返回整条镜像信号;
- ``extrema`` (极值延拓): 每端以首 / 末极值为对称中心外延 ``nbsym`` 个极值
  (EMD 经典 nbsym 端点处理, PyEMD 同款);
- ``polynomial`` (多项式拟合外推): 用两端各 ``nbsym + 1`` 个极值拟合
  "幅值-时间"与"时间-序号"多项式, 外推端点外的 ``nbsym`` 个极值;
- ``parallel`` (平行延拓): 用端点两个极值的连线斜率与平均极值间隔直线外推;
- ``BLCSE`` (边界局部特征尺度延拓): 按端点附近同类极值的平均幅值与平均间隔,
  每端补一对 (极小, 极大), 使边界处的特征尺度连续。

输入约定 (关键)
--------------
**所有模式的输入都是信号本身** (1-D), 极值域模式在内部检出极值:

- 极大 / 极小 = ``find_peaks(S, mod="both")`` —— 一次拿到按样本序归并的极值表与
  ``is_max`` 掩码 (归并在 ``find_peaks`` 内 O(P) 完成), 取值恒为原信号幅值,
  不存在"极值表里分不清极大极小"的问题 (``BLCSE`` 按掩码切出两组有序序列);
- 时间轴是独立入参 ``T``: ``None`` 时取样本序号 (不构造 ``arange``); 给出时按
  时间重排极值, 非均匀 / 非单调仅在 ``verbose=True`` 时告警;
- 返回值是**极值表** ``(2, L')``: 第 0 行时间, 第 1 行幅值, 按时间升序、极大极小
  交替 (``mod="mirror"`` 例外, 它返回 1-D 信号)。

"""
import numpy as np
from enum import Enum
from typing import Literal
from warnings import warn

from ..Define import DEFAULT_NUMPY_TYPE

from .Check import Check
from .Peaks import PEAKS_BACKEND, PEAKS_BACKEND_ARGS, find_peaks
from .Poly import fast_poly
from .Monotonicity import monotonic

__all__ = ["mirror", "MIRROR_MODS", "MIRROR_MODS_ARGS"]


MIRROR_MODS_ARGS = Literal["mirror", "extrema", "polynomial", "parallel", "BLCSE"]
class MIRROR_MODS(Enum):
    Mirror = "mirror"  # mirror extension: 端点镜像
    Extrema = "extrema"  # extrema extension: 极值延拓
    Polynomial = "polynomial"  # polynomial fitting extension: 多项式拟合外推
    Parallel = "parallel"  # parallel extension: 平行延拓
    BLCSE = "BLCSE"  # boundary local characteristic scale extension: 边界局部特征尺度延拓


def mirror(S: np.ndarray | np.memmap, T: np.ndarray = None,
           mod: MIRROR_MODS_ARGS | MIRROR_MODS = "mirror",
           nbsym: int = 3,
           poly_order: int = 2, time_poly_order: int = 2,
           *,
           verbose: bool = False,
           peaks_backend: PEAKS_BACKEND_ARGS | PEAKS_BACKEND = "numpy",
           **kwargs) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """
    信号端点外拓: 端点镜像, 或"检出极值 → 外延极值表"。

    Parameters
    ----------
    S : np.ndarray | np.memmap
        1-D 实数信号。所有模式都吃信号本身, 不需要调用方先给极值表。
    T : np.ndarray | None
        时间轴, 长度须与 ``S`` 一致。``None`` (默认) 时取样本序号 —— 此时不分配
        时间数组、也不做均匀性 / 单调性诊断。给出时极值一律按其值重排, 非均匀 /
        非单调仅在 ``verbose=True`` 时告警 (``mod="mirror"`` 不使用 ``T``)。
    mod : MIRROR_MODS_ARGS | MIRROR_MODS
        延拓模式。
    nbsym : int
        每端外延的极值个数; ``mod="mirror"`` 时即每端镜像的样本数 (0 = 不处理)。
    poly_order, time_poly_order : int
        仅 ``mod="polynomial"`` 使用: 幅值 / 时间轴多项式的阶数, 需满足
        ``1 <= order <= nbsym`` (每端只有 ``nbsym + 1`` 个极值参与拟合)。
    verbose : bool
        是否输出 ``T`` 的诊断告警 (非均匀 / 非单调)。``False`` 时连诊断本身都
        不做, 省掉两次 O(N) 扫描。
    backend : {"numpy", "scipy", "numba"}
        极值检测后端, 透传 :func:`Utils.Peaks.find_peaks` (默认
        ``DEFAULT_BACKEND``)。
    **kwargs
        ``dtype`` (输出 dtype, 默认跟随输入: 浮点原样保留, 整数抬到 float);
        ``safe`` (校验等级, 透传 ``Check.to_array``);
        ``squeeze`` (是否 squeeze 单例维, 透传 ``Check.to_array``)。

    Returns
    -------
    np.ndarray
        ``mod="mirror"``: 1-D 镜像信号, 长度 ``N + 2*nbsym``。
    (mirror_arr, is_max)
        其余模式: ``mirror_arr`` 是 ``(2, L')`` 极值表 (第 0 行时间、第 1 行幅值,
        时间升序), ``is_max`` 是等长 bool 掩码 (``True`` = 该列是极大)。
        ``extrema`` / ``polynomial`` / ``parallel`` 为 ``L' = L + 2*nbsym``,
        ``BLCSE`` 为 ``L' = L + 4`` (每端固定补一对极值, 与 ``nbsym`` 无关);
        其中 ``L`` 是检出的极值数 (极大 + 极小); ``nbsym=0`` 时只有中段。
        调用方按掩码取两组分别拟合上 / 下包络; 端部外延点只参与拟合。

    Raises
    ------
    TypeError
        ``mod`` / ``safe`` / ``verbose`` / ``dtype`` / ``squeeze`` / ``nbsym`` /
        多项式阶数的类型不合法。
    ValueError
        上述参数的取值不合法; 信号点数不足以检出极值; ``T`` 长度与 ``S`` 不符;
        极值太少 (不足以外延 ``nbsym`` 个) 或缺少极大 / 极小; 极值域模式收到
        非浮点 dtype (外推时间可能不是整数, 整型会静默截断)。

    Examples
    --------
    >>> import numpy as np
    >>> from Modal_Decomposition.Utils.Mirror import mirror
    >>> mirror(np.arange(6.0), nbsym=2).tolist()   # 1-D, 反射不含端点
    [2.0, 1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0]
    >>> S = np.sin(np.linspace(0.0, 4.0 * np.pi, 200))
    >>> extrema_arr, is_max = mirror(S, mod="extrema", nbsym=2)
    >>> extrema_arr.shape, is_max.shape            # (2, L + 4) 与同长掩码
    ((2, 8), (8,))
    """
    dtype = kwargs.get("dtype", None)
    safe = kwargs.get("safe", "unknown")
    squeeze = kwargs.get("squeeze", True)

    mod: MIRROR_MODS = Check.validate_mod(mod, MIRROR_MODS)

    Check.validate_safe(safe)

    peaks_backend: PEAKS_BACKEND = Check.validate_mod(peaks_backend, PEAKS_BACKEND)

    if not isinstance(verbose, bool):
        raise TypeError("verbose must be bool!")

    if not isinstance(squeeze, bool):
        raise TypeError("the squeeze must be bool!")

    if dtype is not None:
        try:
            dtype = np.dtype(dtype)
        except TypeError:
            raise TypeError("the dtype is error!")

    if isinstance(nbsym, bool) or not isinstance(nbsym, int):
        raise TypeError(f"the nbsym must be int, got {type(nbsym).__name__}!")

    if not (nbsym >= 0):
        raise ValueError("the nbsym must be >= 0!")

    S: np.ndarray | np.memmap = Check.to_array(
        S, dtype=dtype, safe=safe, ndim=1,
        need_squeeze=squeeze
    )

    if mod == MIRROR_MODS.Mirror:
        output_dtype: np.dtype = S.dtype
    else:
        arr_length: int = S.size

        if dtype is None:
            output_dtype = S.dtype if Check.return_dtype_kind(S) == "float" else DEFAULT_NUMPY_TYPE
        else:
            output_dtype = dtype

        if output_dtype.kind != "f":
            raise ValueError("the dtype must be float when the mod is not 'mirror'!")

        # find_peaks 对 size < 3 的信号会把每个样本都当成峰, 这里直接挡住
        if not (arr_length >= 3):
            raise ValueError(f"the signal must have at least 3 points to detect the extrema ({arr_length})!")

        # 极大 / 极小: 一次 both 拿到"按样本序归并好的极值表 + 类型掩码" (归并在 find_peaks 内 O(P) 完成; 不再需要第二次检测与 searchsorted/insert)
        extrema_idx, extrema_arr, is_max = find_peaks(S, mod="both", backend=peaks_backend, safe="safe")
        extrema_length: int = extrema_idx.size

        if T is None:
            extrema_T = extrema_idx          # 样本序号即时间轴, 不构造 arange
        else:
            T: np.ndarray = Check.to_array(T, dtype=None, safe=safe, ndim=1, need_squeeze=squeeze)

            if T.size != arr_length:
                raise ValueError(f"the T must have the same size as the S ({T.size} != {arr_length})!")

            # 非严格递增 (允许相等) 用 Monotonicity.monotonic 判; 不满足则整表按时间重排
            increasing = monotonic(T, mod="increasing", safe="safe")
            if verbose:
                if not Check.is_uniform(T):
                    warn("the T is not uniform!")

                if not increasing:
                    warn("the T is not increasing monotonically! It'll be rearranged!")

            extrema_T = T[extrema_idx]

            # 按时间重排 (T 非单调时, 样本序 ≠ 时间序): 整表一次 argsort, 掩码同步
            if not increasing:
                extrema_order: np.ndarray = np.argsort(extrema_T)
                extrema_T = extrema_T[extrema_order]
                extrema_arr = extrema_arr[extrema_order]
                is_max = is_max[extrema_order]

    match mod:
        case MIRROR_MODS.Mirror:
            if S.size == 0:
                raise ValueError("the signal must not be empty!")

            if nbsym == 0:
                return S.copy()

            if S.size == 1:
                return np.pad(S, (nbsym, nbsym), mode="edge")

            if nbsym > S.size - 1:
                return np.pad(S, (nbsym, nbsym), mode="reflect")

            mirror_length: int = S.size + 2 * nbsym
            mirror_arr: np.ndarray = np.empty(shape=mirror_length, dtype=S.dtype)

            mirror_arr[:nbsym] = S[nbsym:0:-1]  # S[E], ..., S[1]
            mirror_arr[nbsym:mirror_length - nbsym] = S

            # 右端: 取 S[N-2], S[N-3], ... 共 nbsym 个 (用 mirror_length - nbsym 定位, 避免 -0 取整行)
            mirror_arr[mirror_length - nbsym:] = S[S.size - 2::-1][:nbsym]

            return mirror_arr

        case MIRROR_MODS.Extrema:
            if not (extrema_length >= 1):
                raise ValueError("the signal has no extrema to extend!")

            if not (nbsym <= extrema_length - 1):
                raise ValueError(f"the nbsym must be <= the number of the extrema - 1 ({extrema_length - 1})!")

            mirror_length: int = extrema_length + 2 * nbsym
            mirror_arr: np.ndarray = np.empty(shape=(2, mirror_length), dtype=output_dtype)
            mirror_type: np.ndarray = np.empty(shape=mirror_length, dtype=bool)

            # 左端: 以首极值为对称中心, 把它之后的 nbsym 个极值镜像出去 (型别随镜像源照抄)
            # mirror_arr[0, :nbsym] = 2 * extrema_T[0] - extrema_T[1: nbsym + 1][::-1]
            # mirror_arr[1, :nbsym] = extrema_arr[1: nbsym + 1][::-1]
            mirror_arr[0, :nbsym] = 2 * extrema_T[0] - extrema_T[nbsym:0:-1]
            mirror_arr[1, :nbsym] = extrema_arr[nbsym:0:-1]
            mirror_type[:nbsym] = is_max[nbsym:0:-1]

            mirror_arr[:, nbsym: mirror_length - nbsym] = (extrema_T, extrema_arr)
            mirror_type[nbsym: mirror_length - nbsym] = is_max

            # 右端: 以末极值为对称中心, 把它之前的 nbsym 个极值镜像出去
            # mirror_arr[0, mirror_length - nbsym:] = 2 * extrema_T[-1] - extrema_T[extrema_length - nbsym - 1: -1][::-1]
            # mirror_arr[1, mirror_length - nbsym:] = extrema_arr[extrema_length - nbsym - 1: -1][::-1]
            # (起点写 mirror_length - nbsym: 用 -nbsym 的话 nbsym=0 会取到整行)
            mirror_arr[0, mirror_length - nbsym:] = 2 * extrema_T[-1] - extrema_T[-2:-nbsym - 2:-1]
            mirror_arr[1, mirror_length - nbsym:] = extrema_arr[-2:-nbsym - 2:-1]
            mirror_type[mirror_length - nbsym:] = is_max[-2:-nbsym - 2:-1]

        case MIRROR_MODS.Polynomial:
            if isinstance(poly_order, bool) or not isinstance(poly_order, int):
                raise TypeError("the poly_order must be int!")

            if not (0 < poly_order):
                raise ValueError("the poly_order must be > 0!")

            if isinstance(time_poly_order, bool) or not isinstance(time_poly_order, int):
                raise TypeError("the time_poly_order must be int!")

            if not (0 < time_poly_order):
                raise ValueError("the time_poly_order must be > 0!")

            if not (poly_order <= nbsym and time_poly_order <= nbsym):
                raise ValueError("the poly_order and the time_poly_order must be <= nbsym!")

            if not (nbsym <= extrema_length - 1):
                raise ValueError(f"the nbsym must be <= the number of the extrema - 1 ({extrema_length - 1})!")

            mirror_length: int = extrema_length + 2 * nbsym
            mirror_arr: np.ndarray = np.empty(shape=(2, mirror_length), dtype=output_dtype)
            mirror_type: np.ndarray = np.empty(shape=mirror_length, dtype=bool)
            mirror_arr[:, nbsym: mirror_length - nbsym] = (extrema_T, extrema_arr)
            mirror_type[nbsym: mirror_length - nbsym] = is_max

            head_T, head_arr = extrema_T[: nbsym + 1], extrema_arr[: nbsym + 1]
            tail_T, tail_arr = extrema_T[extrema_length - nbsym - 1:], extrema_arr[extrema_length - nbsym - 1:]

            # 时间轴: 以"极值序号"为自变量拟合 T, 在越界的序号上求值
            head_ordinal: np.ndarray = np.arange(nbsym + 1)
            tail_ordinal: np.ndarray = np.arange(extrema_length - nbsym - 1, extrema_length)
            # left_ordinal: np.ndarray = np.arange(-nbsym, 0)
            # right_ordinal: np.ndarray = np.arange(extrema_length, extrema_length + nbsym)
            left_ordinal = head_ordinal[:nbsym] - nbsym
            right_ordinal = np.arange(extrema_length, extrema_length + nbsym)

            left_T: np.ndarray = fast_poly(head_ordinal, head_T, time_poly_order, left_ordinal, safe="safe")
            left_arr: np.ndarray = fast_poly(head_T, head_arr, poly_order, left_T, safe="safe")
            right_T: np.ndarray = fast_poly(tail_ordinal, tail_T, time_poly_order, right_ordinal, safe="safe")
            right_arr: np.ndarray = fast_poly(tail_T, tail_arr, poly_order, right_T, safe="safe")

            # 拟合出来的时间在端点外不保证单调 (会回折 -> 时间行不递增, 样条不可用):
            # 一旦折返或越过边界, 就退回"端点时刻 ± 端点附近平均间隔", 与 parallel 同款兜底。
            if not (np.all(np.diff(left_T) > 0) and left_T[-1] < head_T[0]):
                left_T = head_T[0] - np.arange(nbsym, 0, -1, dtype=np.float64) * np.abs(np.diff(head_T)).mean()

            if not (np.all(np.diff(right_T) > 0) and right_T[0] > tail_T[-1]):
                right_T = tail_T[-1] + np.arange(1, nbsym + 1, dtype=np.float64) * np.abs(np.diff(tail_T)).mean()

            mirror_arr[0, :nbsym] = left_T
            mirror_arr[1, :nbsym] = left_arr
            mirror_arr[0, mirror_length - nbsym:] = right_T
            mirror_arr[1, mirror_length - nbsym:] = right_arr

            # 端部是合成点: 型别按"从边界极值出发向外交替" (最近的一个与边界相反)
            near_left, near_right = np.bool_(is_max[0]), np.bool_(is_max[-1])
            mirror_type[:nbsym] = np.where(np.arange(nbsym, 0, -1) % 2 == 1, ~near_left, near_left)
            mirror_type[mirror_length - nbsym:] = np.where(np.arange(1, nbsym + 1) % 2 == 1, ~near_right, near_right)

        case MIRROR_MODS.Parallel:
            if not (extrema_length >= 2):
                raise ValueError("the signal must have at least 2 extrema for the parallel extension!")

            if not (nbsym <= extrema_length - 1):
                raise ValueError(f"the nbsym must be <= the number of the extrema - 1 ({extrema_length - 1})!")

            mirror_length: int = extrema_length + 2 * nbsym
            mirror_arr: np.ndarray = np.empty(shape=(2, mirror_length), dtype=output_dtype)
            mirror_type: np.ndarray = np.empty(shape=mirror_length, dtype=bool)
            # 平行延拓的步长: 极值间隔的平均值 (与方向无关, 取绝对值)
            step = np.abs(np.diff(extrema_T)).mean()

            # 左端: 用最左两个极值的连线斜率, 沿直线往外推 (由远到近写入)
            left_slope = (extrema_arr[1] - extrema_arr[0]) / (extrema_T[1] - extrema_T[0]) if extrema_T[1] != extrema_T[0] else 0.0
            left_distance: np.ndarray = np.arange(nbsym, 0, -1, dtype=np.float64) * step
            mirror_arr[0, :nbsym] = extrema_T[0] - left_distance
            mirror_arr[1, :nbsym] = extrema_arr[0] - left_distance * left_slope

            # 右端: 用最右两个极值的连线斜率 (由近到远写入)
            right_slope = (extrema_arr[-1] - extrema_arr[-2]) / (extrema_T[-1] - extrema_T[-2]) if extrema_T[-1] != extrema_T[-2] else 0.0
            right_distance: np.ndarray = np.arange(1, nbsym + 1, dtype=np.float64) * step
            mirror_arr[0, mirror_length - nbsym:] = extrema_T[-1] + right_distance
            mirror_arr[1, mirror_length - nbsym:] = extrema_arr[-1] + right_distance * right_slope

            mirror_arr[:, nbsym: mirror_length - nbsym] = (extrema_T, extrema_arr)
            mirror_type[nbsym: mirror_length - nbsym] = is_max

            # 端部是合成点: 型别按"从边界极值出发向外交替" (最近的一个与边界相反)
            near_left, near_right = np.bool_(is_max[0]), np.bool_(is_max[-1])
            mirror_type[:nbsym] = np.where(np.arange(nbsym, 0, -1) % 2 == 1, ~near_left, near_left)
            mirror_type[mirror_length - nbsym:] = np.where(np.arange(1, nbsym + 1) % 2 == 1, ~near_right, near_right)

        case MIRROR_MODS.BLCSE:
            # 极大 / 极小各自的序列: 从归并表按掩码切出 (掩码已随表按时间重排, 切出后仍升序)
            max_T, min_T = extrema_T[is_max], extrema_T[~is_max]
            max_arr, min_arr = extrema_arr[is_max], extrema_arr[~is_max]

            if not (min_T.size >= 1 and max_T.size >= 1):
                raise ValueError("the BLCSE needs both the maxima and the minima of the signal!")

            if not (nbsym >= 1):
                raise ValueError("the nbsym must be >= 1 for the BLCSE (is the averaging window)!")

            # 幅值: 端点附近 nbsym 个同类极值的均值 (极大 / 极小各取自己的序列)
            n_use: int = min(nbsym, max_arr.size, min_arr.size)
            max_arr_left, min_arr_left = max_arr[:n_use].mean(), min_arr[:n_use].mean()
            max_arr_right, min_arr_right = max_arr[-n_use:].mean(), min_arr[-n_use:].mean()

            # 间隔: 同类极值在端点附近的平均间隔 (不足则退回全体极值间隔)
            max_step: np.ndarray = np.diff(max_T)
            min_step: np.ndarray = np.diff(min_T)
            overall_step = np.diff(extrema_T).mean()
            overall_step = overall_step if overall_step > 0 else 1.0

            window: int = min(nbsym, max_step.size)
            max_step_left = max_step[:window].mean() if window > 0 else overall_step
            max_step_right = max_step[-window:].mean() if window > 0 else overall_step

            window = min(nbsym, min_step.size)
            min_step_left = min_step[:window].mean() if window > 0 else overall_step
            min_step_right = min_step[-window:].mean() if window > 0 else overall_step

            # 每端补一对 (极小, 极大), 按时间先后落位
            mirror_length: int = extrema_length + 4
            mirror_arr: np.ndarray = np.empty(shape=(2, mirror_length), dtype=output_dtype)
            mirror_type: np.ndarray = np.empty(shape=mirror_length, dtype=bool)
            mirror_arr[:, 2: 2 + extrema_length] = (extrema_T, extrema_arr)
            mirror_type[2: 2 + extrema_length] = is_max

            left_min_T = min_T[0] - min_step_left
            left_max_T = max_T[0] - max_step_left

            # 修正 1: 两个候选点可能算到同一时刻 (两类局部间隔不同) -> 拉开一个平均间隔
            if left_max_T == left_min_T:
                left_max_T = left_min_T + overall_step

            # 修正 2: 端点对必须严格落在表外, 否则会与中段首列撞时间 (重复时间戳 -> 样条不可用)
            left_late = max(left_min_T, left_max_T)
            if left_late >= extrema_T[0]:
                shift = left_late - extrema_T[0] + overall_step
                left_min_T, left_max_T = left_min_T - shift, left_max_T - shift

            if left_min_T <= left_max_T:
                mirror_arr[0, 0], mirror_arr[1, 0] = left_min_T, min_arr_left
                mirror_type[0] = False
                mirror_arr[0, 1], mirror_arr[1, 1] = left_max_T, max_arr_left
                mirror_type[1] = True
            else:
                mirror_arr[0, 0], mirror_arr[1, 0] = left_max_T, max_arr_left
                mirror_type[0] = True
                mirror_arr[0, 1], mirror_arr[1, 1] = left_min_T, min_arr_left
                mirror_type[1] = False

            right_min_T = min_T[-1] + min_step_right
            right_max_T = max_T[-1] + max_step_right

            # 右端同理: 先拉开重合的两个候选点, 再整体推到末列之后
            if right_max_T == right_min_T:
                right_max_T = right_min_T + overall_step

            right_early = min(right_min_T, right_max_T)
            if right_early <= extrema_T[-1]:
                shift = extrema_T[-1] + overall_step - right_early
                right_min_T, right_max_T = right_min_T + shift, right_max_T + shift

            if right_min_T <= right_max_T:
                mirror_arr[0, 2 + extrema_length], mirror_arr[1, 2 + extrema_length] = right_min_T, min_arr_right
                mirror_type[2 + extrema_length] = False
                mirror_arr[0, 3 + extrema_length], mirror_arr[1, 3 + extrema_length] = right_max_T, max_arr_right
                mirror_type[3 + extrema_length] = True
            else:
                mirror_arr[0, 2 + extrema_length], mirror_arr[1, 2 + extrema_length] = right_max_T, max_arr_right
                mirror_type[2 + extrema_length] = True
                mirror_arr[0, 3 + extrema_length], mirror_arr[1, 3 + extrema_length] = right_min_T, min_arr_right
                mirror_type[3 + extrema_length] = False

        case _:
            raise ValueError(f"the mods of mirror only support {MIRROR_MODS._member_names_}!")

    return mirror_arr, mirror_type
