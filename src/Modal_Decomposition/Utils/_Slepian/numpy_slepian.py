"""
Pure-NumPy DPSS generator (discrete prolate spheroidal / Slepian sequences).

算法
----
DPSS 是下述**对称三对角矩阵**的特征向量 (Slepian 1978; Percival & Walden 1993,
"Spectral Analysis for Physical Applications", eq. 380):

    d_i = ((N - 1 - 2i) / 2)^2 * cos(2*pi*W),      i = 0 .. N-1   (主对角)
    e_i = i * (N - i) / 2,                         i = 1 .. N-1   (次对角)
    W   = halfBW / N                                (半带宽, 时间-带宽积 NW/N)

第 k 个特征向量即第 k 阶 Slepian 序列 (按集中比降序)。注意该三对角问题的特征值
``l2`` **不是**集中比 ``l1`` (Percival & Walden 1993, pg 379/390): 真集中比由
自相关法给出 ``l1 = sum_j r_j * autocorr_j``, ``r_j = 4W*sinc(2W*j)`` (``r_0`` 取
``2W``), 本模块按同一公式给出, 与 ``scipy.signal.windows.dpss`` 的 ``ratios`` 一致。

中心对称性折半 (本后端的核心优化)
--------------------------------
上述矩阵满足 ``d_i = d_{N-1-i}``、``e_i = e_{N-i}``, 即对换矩阵 J 有 ``J T J = T``
(中心对称), 于是特征向量具有确定奇偶性 ``v_{N-1-i} = sigma * v_i`` (sigma = ±1),
问题解耦为两个**半尺寸**问题。取自由半区 ``u_j = v_j`` (j = 0 … h-1):

* 约化方程本身**非对称** (奇数 N 的对称支: 末行折回项落在下三角), 但经
  ``u -> G^{-1/2} u`` (``G = diag(W_j)``) 相似变换后与**对称三对角**矩阵同谱:

      B[j][j]   = d_j + fold_j
      B[j][j+1] = e_{j+1} * sqrt(W_j / W_{j+1})
      u_j       = (B 的特征向量分量) / sqrt(W_j)

  权重 ``W_j = 2``, 唯独"自映射"的自由点取 1 (仅奇数 N 的对称支的中间点)。
* 折回项 ``fold``: N 偶时落在末主对角 (对称 ``+e_h`` / 反对称 ``-e_h``); N 奇时
  对称支的折回被权重完全吸收 (该结构恒有 ``e_m = e_{m+1}``), 反对称支中间点恒为
  0 故无折回。

规模对照 (h = 半长):

===========  ==========================  ==========================
N 的奇偶      对称支 (sigma=+1)            反对称支 (sigma=-1)
===========  ==========================  ==========================
``N = 2m+1``  ``h = m+1``, ``W[-1] = 1``   ``h = m`` (中间点强制 0)
``N = 2m``    ``h = m``, ``fold = +e_m``   ``h = m``, ``fold = -e_m``
===========  ==========================  ==========================

相对"直接构造 N×N 稠密矩阵再 eigh"的朴素实现, 本实现时间约 O(N^3/4)、内存约
O(N^2/2) (约 4x 快、2x 省内存), 且不再构造全长稠密矩阵。稠密分解仍使本后端只适合
中等长度 (大 N 请用 ``mod="C"``)。

Python version: 3.10

Lib and Version:
    numpy - 2.2.6

Only accessed by: Utils.Slepian (经 Utils.get_slepian() 惰性注册)

Modify:
    2026.9.15
"""

import numpy as np

from ...Base.ConstDefine import SLEPIAN_NUMPY_MAX_BYTES, DEFAULT_NUMPY_TYPE

__all__ = ["generate_slepian_numpy"]

#: ``norm`` 取值: 2 = 单位能量 (scipy 在给出 Kmax 时的默认), None = 同 2,
#: "approximate" = 峰值 1 并按 ``M^2/(M^2+NW)`` 修正偶长度的功率差 (scipy 同名取法)。
_NORMS = (2, None, "approximate")


def _tridiag_entries(N: int, halfBW: float) -> tuple:
    """
    构造全长 DPSS 三对角矩阵的 (主对角 ``d``, 次对角 ``e``)。

    ``e`` 用长度 N 的数组表示, ``e[i]`` 即连接下标 ``i-1`` 与 ``i`` 的元素
    (``e[0]`` 不使用), 与折半公式的下标一一对应。
    """
    cosW = float(np.cos(2.0 * np.pi * (float(halfBW) / float(N))))
    i = np.arange(N, dtype=np.float64)
    d = ((N - 1 - 2.0 * i) / 2.0) ** 2 * cosW
    e = np.zeros(N, dtype=np.float64)
    j = np.arange(1, N, dtype=np.float64)
    e[1:] = j * (N - j) / 2.0
    return d, e


def _reduced(d: np.ndarray, e: np.ndarray, parity: int) -> tuple:
    """
    折出半尺寸**对称三对角** (主对角 ``diag``, 次对角 ``off``) 与其权重 ``w``。

    ``parity = +1`` 取对称支 (``v_{N-1-i} = +v_i``), ``-1`` 取反对称支。
    ``off`` 已含权重因子; 特征向量需按 ``u_j = b_j / sqrt(w_j)`` 还原。
    """
    n = int(d.size)
    if parity > 0:
        h = (n + 1) // 2
        w = np.full(h, 2.0)
        if n % 2 == 1:
            w[-1] = 1.0                     # 奇数 N: 中间点自映射, 权重 1
        diag = d[:h].copy()
        if n % 2 == 0:
            diag[-1] += e[h]                # 折回落在末主对角 (对称 +)
    else:
        h = n // 2
        w = np.full(h, 2.0)
        diag = d[:h].copy()
        if n % 2 == 0:
            diag[-1] -= e[h]                # 折回落在末主对角 (反对称 -)
    off = e[1:h] * np.sqrt(w[:-1] / w[1:]) if h > 1 else np.empty(0, dtype=np.float64)
    return diag, off, w


def _dense_tridiag(diag: np.ndarray, off: np.ndarray) -> np.ndarray:
    """由 (主对角, 次对角) 组装稠密对称三对角矩阵 (仅半尺寸, 故只需 O(h^2) 内存)。"""
    h = int(diag.size)
    t = np.diag(diag)
    if h > 1:
        t += np.diag(off, 1) + np.diag(off, -1)
    return t


def _expand(half: np.ndarray, n: int, parity: int) -> np.ndarray:
    """
    把半长向量 (已去权重) 按奇偶性镜像展开到长度 N。

    ``out[i] = parity * out[N-1-i]``; 奇数 N 的反对称支中间点自映射, 强制为 0
    (它正好落在尾段首个位置)。
    """
    h = int(half.size)
    out = np.empty(n, dtype=np.float64)
    out[:h] = half
    tail = np.arange(h, n)
    src = n - 1 - tail
    free = src < h                          # 镜像源落在自由半区
    out[tail[free]] = parity * out[src[free]]
    if not np.all(free):                    # 自映射的中间点 -> 反对称强制 0
        out[tail[~free]] = 0.0
    return out


def _sign_fix(tapers: np.ndarray, parities: list) -> None:
    """
    就地统一符号 (Percival & Walden 1993, pg 379; 与 scipy 同约定):

    * 偶阶 (对称支): 序列和为正;
    * 奇阶 (反对称支): 首个显著瓣为正 (阈值 ``max(1e-7, 1/N)``)。
    """
    n = tapers.shape[1]
    thresh = max(1e-7, 1.0 / n)
    for k, parity in enumerate(parities):
        row = tapers[k]
        if parity > 0:
            if row.sum() < 0.0:
                row *= -1.0
        else:
            sig = row[row * row > thresh]
            if sig.size and sig[0] < 0.0:
                row *= -1.0


def _concentration_ratios(tapers: np.ndarray, W: float) -> np.ndarray:
    """
    真集中比 (Percival & Walden 1993, pg 390 的自相关法, 与 scipy 同式)。

    ``l1_k = sum_j r_j * autocorr_k[j]``, ``r_j = 4W*sinc(2W*j)`` 且 ``r_0 = 2W``。
    自相关用 FFT 计算 (线性自相关, 长度 ``2N-1``); 该量与序列的整体缩放无关。
    """
    n = int(tapers.shape[1])
    use_n = 2 * n - 1
    spec = np.fft.rfft(tapers, n=use_n, axis=-1)
    autocorr = np.fft.irfft(spec * spec.conj(), n=use_n, axis=-1)[:, :n]
    j = np.arange(n, dtype=np.float64)
    r = 4.0 * W * np.sinc(2.0 * W * j)
    r[0] = 2.0 * W
    return autocorr @ r


def _normalize(tapers: np.ndarray, norm, halfBW: float) -> np.ndarray:
    """归一化: 2/None = 单位能量; "approximate" = 峰值 1 (+ 偶长度功率修正)。"""
    if norm == 2 or norm is None:
        nrm = np.linalg.norm(tapers, axis=1, keepdims=True)
        nrm[nrm == 0.0] = 1.0
        return tapers / nrm
    m = tapers.shape[1]
    out = tapers / tapers.max()
    if m % 2 == 0:
        out *= m ** 2 / float(m ** 2 + halfBW)
    return out


def generate_slepian_numpy(N: int, halfBW: float, nTapers: int = None,
                           return_ratios: bool = False, norm=None,
                           sym: bool = True) -> np.ndarray:
    """
    生成 Slepian (DPSS) 序列 —— 纯 NumPy 后端。

    Parameters
    ----------
    N : int
        序列长度 (>= 1)。
    halfBW : float
        时间-带宽积 ``NW`` (> 0); 半带宽 ``W = halfBW / N``。
    nTapers : int | None
        返回条数 (阶数 0 … nTapers-1)。``None`` 时取
        ``max(1, floor(2*halfBW - 1))``, 并截断到 ``N``。
    return_ratios : bool
        为 True 时同时返回集中比 (降序, 与 scipy 同式)。
    norm : {2, None, "approximate"}
        ``2``/``None`` (默认) 单位能量; ``"approximate"`` 峰值 1 并按
        ``M^2/(M^2+NW)`` 修正。``"subsample"`` 请走 ``mod="scipy"``。
    sym : bool
        ``True`` (默认) 生成长度 ``N`` 的对称序列; ``False`` 生成周期性
        (DFT-even) 序列, 即取长度 ``N+1`` 的解后去掉末点 (同 scipy)。

    Returns
    -------
    np.ndarray | tuple
        ``(nTapers, N)`` 序列 (行 = 阶数, 集中比降序); ``return_ratios=True`` 时
        返回 ``(tapers, ratios)``。

    Raises
    ------
    ValueError
        ``N < 1`` / ``halfBW <= 0`` / ``nTapers < 1`` / 未知 ``norm``。
    """
    N = int(N)
    if N < 1:
        raise ValueError(f"N must be >= 1, got {N}")
    if not np.isfinite(halfBW) or float(halfBW) <= 0.0:
        raise ValueError(f"halfBW must be > 0, got {halfBW!r}")
    if norm not in _NORMS:
        raise ValueError(
            f"norm must be one of {_NORMS} (norm='subsample' 请用 mod='scipy'), got {norm!r}"
        )

    solve_n = N if sym else N + 1           # sym=False: 解 N+1 再截断 (scipy 同法)

    # 内存预算守卫: 折半后最大半长 hm = ceil(solve_n/2), 需 (hm×hm) 稠密矩阵。
    # 超预算直接拒绝 —— 否则 N 稍大就会 GB 级分配甚至打爆内存 (本后端的固有代价,
    # 半尺寸稠密 eigh); 大 N 请用 mod="C"/"scipy" (两者内存均为 O(N))。
    hm = (solve_n + 1) // 2
    need_bytes = hm * hm * np.dtype(DEFAULT_NUMPY_TYPE).itemsize
    if need_bytes > SLEPIAN_NUMPY_MAX_BYTES:
        n_max = 2 * int(np.sqrt(SLEPIAN_NUMPY_MAX_BYTES
                                // np.dtype(DEFAULT_NUMPY_TYPE).itemsize))
        raise ValueError(
            f"numpy 后端需构造 {hm}x{hm} 稠密矩阵 (~{need_bytes / 1024 ** 2:.0f} MB), "
            f"超过 SLEPIAN_NUMPY_MAX_BYTES={SLEPIAN_NUMPY_MAX_BYTES / 1024 ** 2:.0f} MB; "
            f"该后端仅适合 N <= {n_max}, 请改用 mod='C' 或 mod='scipy'"
        )

    if nTapers is None:
        nTapers = max(1, int(np.floor(2.0 * float(halfBW) - 1)))
    nTapers = int(nTapers)
    if nTapers < 1:
        raise ValueError(f"nTapers must be >= 1, got {nTapers}")
    nTapers = min(nTapers, solve_n)         # 至多 N 条 (N 维正交基)

    W = float(halfBW) / float(solve_n)
    d, e = _tridiag_entries(solve_n, halfBW)
    ds, es, ws = _reduced(d, e, +1)
    da, ea, wa = _reduced(d, e, -1)

    # 每支只要"降序前 nTapers 个"; eigh 只给升序全谱, 故从**尾部**取起 ——
    # 下标用矩阵尺寸 hs/ha, 不是条数 take_s/take_a。
    hs, ha = int(ds.size), int(da.size)
    take_s, take_a = min(nTapers, hs), min(nTapers, ha)
    w_s, v_s = np.linalg.eigh(_dense_tridiag(ds, es))
    w_a, v_a = np.linalg.eigh(_dense_tridiag(da, ea))

    cand_w = np.concatenate([w_s[::-1][:take_s], w_a[::-1][:take_a]])
    cand_src = [("s", j) for j in range(take_s)] + [("a", j) for j in range(take_a)]
    order = np.argsort(cand_w)[::-1][:nTapers]

    tapers = np.empty((nTapers, solve_n), dtype=np.float64)
    parities = []
    for row, pos in enumerate(order):
        which, j = cand_src[int(pos)]
        if which == "s":
            half = v_s[:, hs - 1 - j] / np.sqrt(ws)     # 去权重 -> 自由半区
        else:
            half = v_a[:, ha - 1 - j] / np.sqrt(wa)
        tapers[row] = _expand(half, solve_n, +1 if which == "s" else -1)
        parities.append(+1 if which == "s" else -1)

    _sign_fix(tapers, parities)
    ratios = _concentration_ratios(tapers, W) if return_ratios else None

    tapers = _normalize(tapers, norm, float(halfBW))
    if not sym:
        tapers = tapers[:, :-1]

    if return_ratios:
        return tapers, ratios
    return tapers
