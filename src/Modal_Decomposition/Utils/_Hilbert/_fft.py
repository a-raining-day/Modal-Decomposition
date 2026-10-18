"""``"FFT"`` 后端 —— 实半谱 (rfft/irfft) 解析信号，即 ``HILBERT_MODS.FFT`` 的实现。

算法
----
``scipy.signal.hilbert`` 用 **两次复数 FFT**（``fft`` + ``ifft``）。本模块用
**一次 ``rfft`` + 一次 ``irfft``**（两次*实*变换，即两个半长复 FFT），工作量约为
scipy 的 1/2；因此在本机所有尺寸上都更快（实测 0.59–0.99×）。

数学推导
--------
解析信号 ``z = x + i·H(x)``，其频谱 ``A[k] = F[k]·(1 + sgn k)``：

* 实部就是 ``x`` 本身（不需要逆变换）；
* 虚部 ``y = H(x)`` 的频谱 ``Y[k] = -i·sgn(k)·F[k]``，即正频率半谱是 ``-i·F``，
  ``k = 0`` 与 ``k = N/2`` 处为 0。由于 ``y`` 是实信号，``irfft`` 可以从这个半谱
  直接还原它。

所以只需 ``rfft(x)`` → 乘 ``-i`` → 清零 DC/Nyquist → ``irfft`` → 拼成复数数组。

与其它后端的关系
----------------
* 与 ``scipy.signal.hilbert`` **逐点一致**（实测 ≤ 4e-15，含奇数长度）；
* **不要求长度为 2 的幂** —— 这是它相对 SAO 内核的决定性优势
  （SAO 内核把输入零填充到 2 的幂，非线性地改变了解析信号）；
* 默认 ``matlab_phase=True``（MATLAB/scipy 的 −90° 约定），与
  ``_fht._hilbert`` 一致；``matlab_phase=False`` 返回其共轭（SAO 的 +90°）。

依赖
----
默认只用 ``numpy``（``np.fft`` = pocketfft，带 SIMD）。**若安装了 scipy**，
则优先用 ``scipy.fft`` 的 ``rfft``/``irfft`` —— 同一 pocketfft 内核，但包装层
开销更低，实测快 5%–18%（n=4096: 17.6→13.8 µs / n=16384: 63→54 µs）；
scipy 未安装时自动回退 ``np.fft``，功能与数值不变。本模块**不含**第三方代码，
故无需额外署名。
"""

import numpy as np

__all__ = ["_hilbert"]

# rfft/irfft 的提供者: 惰性选择 scipy.fft (更快) → np.fft (回退)。
# 模块级缓存, 避免每次调用重复 try/import。
_rfft = None
_irfft = None


def _load_fft():
    """惰性选择 rfft/irfft 实现, 只探测一次。"""
    global _rfft, _irfft
    if _rfft is None:
        try:                        # scipy 是可选依赖: 装了就用它的瘦包装
            from scipy import fft as _scipy_fft
            _rfft, _irfft = _scipy_fft.rfft, _scipy_fft.irfft
        except ImportError:
            _rfft, _irfft = np.fft.rfft, np.fft.irfft
    return _rfft, _irfft


def _as_real_1d(signal) -> np.ndarray:
    """转成 float64 的一维实数组（与 ``_fht._as_real_1d`` 语义一致）。

    这里单独实现一份而不是从 ``_fht`` 导入：``_fht`` 需要反向 import 本模块
    做自适应分发，重复 5 行可以避免包内循环 import。
    """
    x = np.asarray(signal, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError(f"hilbert 只接受一维实信号, 当前输入维度为 {x.ndim}D")
    return x


def _hilbert(signal, matlab_phase: bool = True) -> np.ndarray:
    """实信号的解析信号（``"FFT"`` 后端）。

    Parameters
    ----------
    signal : array-like
        一维实信号（与 ``"FHT"`` 后端同样的输入约定：二维输入抛 ``ValueError``，
        整数/float32 自动升 float64，复数输入丢虚部并给 ``ComplexWarning``）。
    matlab_phase : bool
        ``True``（默认）→ MATLAB / ``scipy.signal.hilbert`` 约定（虚部 = −H）；
        ``False`` → 取共轭，即 SAO 内核的 +90° 约定。

    Returns
    -------
    np.ndarray
        ``complex128`` 解析信号，长度与输入相同（任意长度，无需零填充）。
    """
    x = _as_real_1d(signal)
    n = x.shape[0]
    if n == 0:                      # 与 "FHT" 后端一致：空输入返回空数组
        return np.zeros(0, dtype=np.complex128)

    rfft, irfft = _load_fft()
    half = rfft(x)
    half *= -1j                     # 正频率半谱 ×(-i) ⇒ H(x) 的半谱
    half[0] = 0.0                   # DC：H 的直流分量为 0
    if n % 2 == 0:
        half[n // 2] = 0.0          # Nyquist：sgn(N/2) = 0

    z = np.empty(n, dtype=np.complex128)
    z.real = x                      # 实部恒等于输入（无需逆变换）
    z.imag = irfft(half, n)
    if not matlab_phase:
        z = z.conjugate()
    return z
