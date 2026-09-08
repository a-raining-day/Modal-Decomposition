# src/Modal_Decomposition/Utils/_Hilbert/_fht.py
"""快速 Hartley 变换 (FHT) 与 Hilbert 变换 —— 对外统一入口。

本模块是 ``Modal-Decomposition`` 对外提供 FHT / Hilbert 变换的统一入口
(``Hilbert.hilbert(S, "FHT")`` 及直接导入 ``fht_forward`` 等都会用到它):

    * fht_forward       : 前向离散 Hartley 变换 (自然序 DHT)
    * fht_inverse       : 逆 DHT(利用自逆性除以长度 N)
    * hilbert_transform : 实信号的 Hilbert 变换, 返回解析信号
    * _hilbert          : Hilbert 变换对外统一入口

实现说明:
    本文件为纯 NumPy 参考实现, 不依赖 Cython / C 编译器, 因此在任何
    环境(包括 pip 安装的纯 Python 包)都可以直接运行。
    若需要用 Cython 加速, 请用同目录 ``setup.py`` 编译 ``_fht.pyx``,
    编译产物模块名为 ``_fht_native`` —— 刻意与 ``_fht.py`` 不同名,
    编译后不会遮蔽本文件; 本模块检测到编译产物时会自动优先使用它,
    找不到时自动回退到本文件里的纯 Python 实现。

数值语义与 ``_fht.pyx`` / C 实现保持一致:
    * 输入长度不是 2 的幂时自动零填充到下一个 2 的幂
    * 前向输出为自然序 DHT: H[k] = sum_n x[n] * cas(2*pi*k*n/N),
      cas(θ) = cos(θ) + sin(θ)
    * 逆变换 = 前向变换后除以 N (DHT 的自逆性)

第三方代码声明:
    可选的 Cython 加速内核(``_fht_native``, 见 ``setup.py`` 与 ``_C/_fht``)
    源自 Smithsonian Astrophysical Observatory (SAO) "am" 项目 (S. Paine),
    经 https://github.com/waddafunk/Smithsonians_Discrete_Hilbert_Fourier_Hartley_Transforms
    拆出, 可自由使用但须注明出处与致谢 SAO(完整许可见 README
    "Acknowledgement" 章节)。其 Hilbert 变换相位约定为 +90°, 与
    MATLAB/SciPy 的 -90° 相反: matlab_phase=True 等价于把解析信号虚部乘
    -1, 得到与 MATLAB 一致的结果。纯 NumPy 参考实现采用同一约定, 因此
    无论是否使用编译内核, 输出语义完全一致。
"""

import numpy as np

__all__ = ["_hilbert", "fht_forward", "fht_inverse"]


def _next_pow2(n: int) -> int:
    """返回 >= n 的最小 2 的幂(与 _fht.pyx 的 C 循环一致)。"""
    p = 1
    while p < n:
        p <<= 1
    return p


def _as_real_1d(signal) -> np.ndarray:
    """转成 float64 的一维实数组(维度不合法时抛错, 对齐编译版行为)。"""
    x = np.asarray(signal, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError(
            f"_fht 只接受一维实信号, 当前输入维度为 {x.ndim}D")
    return x


def _pad_pow2(x: np.ndarray) -> np.ndarray:
    """零填充到下一个 2 的幂(长度不变则原样拷贝)。"""
    n_orig = x.shape[0]
    n = _next_pow2(n_orig)
    buf = np.zeros(n, dtype=np.float64)
    buf[:n_orig] = x
    return buf


# ---------------------------------------------------------------------------
# 纯 NumPy 参考实现
# ---------------------------------------------------------------------------

def _dht(x: np.ndarray) -> np.ndarray:
    """
    自然序 DHT 核心: H[k] = sum_n x[n] * (cos(2*pi*k*n/N) + sin(2*pi*k*n/N))。

    通过 FFT 计算(O(N log N)):
        F[k] = sum_n x[n] * (cos - i*sin)  =>  Re(F) - Im(F) = sum x*(cos + sin)
    """
    f = np.fft.fft(x)
    return f.real - f.imag


def _fht_forward_pure(x: np.ndarray, normalize_order: bool = True) -> np.ndarray:
    """
    前向 FHT(纯 NumPy 参考实现)。
    参数与语义和 Cython 版本 ``_fht.pyx::fht_forward`` 一致:

    x: 实信号, 长度最好是 2 的幂; 否则自动零填充到下一个 2 的幂。
    normalize_order: Cython 版本里 False 会保留位反转序输出。
                     纯 NumPy 版本始终返回自然序, 该参数仅为接口兼容保留。
    返回: Hartley 变换结果(实数组, 长度为下一个 2 的幂)。
    """
    x = _as_real_1d(x)
    buf = _pad_pow2(x)
    return _dht(buf)


def _fht_inverse_pure(x: np.ndarray, input_is_normal_order: bool = True) -> np.ndarray:
    """
    逆 FHT(纯 NumPy 参考实现)。
    DHT 自逆: 再做一次前向变换后除以长度 N。

    x: 实数组, 长度最好是 2 的幂; 否则自动零填充到下一个 2 的幂。
    input_is_normal_order: Cython 版本里 False 表示输入是位反转序。
                           纯 NumPy 版本只处理自然序输入, 参数仅为接口兼容保留。
    返回: 逆变换结果(实数组, 长度为下一个 2 的幂)。
    """
    x = _as_real_1d(x)
    buf = _pad_pow2(x)
    return _dht(buf) / buf.shape[0]


def _hilbert_transform_pure(
    real_signal: np.ndarray, matlab_phase: bool = False
) -> np.ndarray:
    """
    计算实信号的 Hilbert 变换, 返回解析信号 (纯 NumPy 参考实现)。

    matlab_phase:
        True  -> 与 MATLAB / scipy.signal.hilbert 相同的相位约定
                 (解析信号频谱只保留正频率, 虚部为 +90° Hilbert 分量)。
        False -> 默认约定 (+90°, 即 MATLAB 的共轭; 幅度/包络不变)。
                 与 _fht.pyx 的默认输出约定一致。
    返回: complex128 解析信号, 长度与输入相同(零填充只发生在内部)。
    """
    x = _as_real_1d(real_signal)
    n_orig = x.shape[0]
    buf = _pad_pow2(x)
    n = buf.shape[0]

    f = np.fft.fft(buf)
    analytic = np.zeros_like(f)
    analytic[0] = f[0]                    # DC 保持
    analytic[1:(n // 2)] = 2.0 * f[1:(n // 2)]  # 正频率 ×2
    analytic[n // 2] = f[n // 2]          # Nyquist 保持
    z = np.fft.ifft(analytic)             # 标准解析信号: 实部=原信号, 虚部=+90° 分量

    if not matlab_phase:
        z = z.conjugate()                 # 默认约定: 取共轭(与 .pyx 默认一致)
    return z[:n_orig]


# ---------------------------------------------------------------------------
# 对外 API(优先使用已编译的 Cython 加速模块, 否则使用上面的参考实现)
# ---------------------------------------------------------------------------

def fht_forward(x: np.ndarray, normalize_order: bool = True) -> np.ndarray:
    """
    前向 FHT: H[k] = sum_n x[n] * cas(2*pi*k*n/N), 返回自然序结果。
    长度不足 2 的幂时自动零填充到下一个 2 的幂。
    """
    return _fht_forward_pure(x, normalize_order=normalize_order)


def fht_inverse(x: np.ndarray, input_is_normal_order: bool = True) -> np.ndarray:
    """
    逆 FHT: 再做一次前向变换并除以 N(DHT 自逆)。
    长度不足 2 的幂时自动零填充到下一个 2 的幂。
    """
    return _fht_inverse_pure(x, input_is_normal_order=input_is_normal_order)


def hilbert_transform(real_signal: np.ndarray, matlab_phase: bool = False) -> np.ndarray:
    """
    实信号的 Hilbert 变换, 返回解析信号(长度与输入相同)。
    matlab_phase=True 时与 scipy.signal.hilbert 相位约定一致。
    """
    return _hilbert_transform_pure(real_signal, matlab_phase=matlab_phase)


try:  # 可选加速: setup.py 编译出的 _fht_native(与本文件不同名, 不会互相遮蔽)
    from . import _fht_native as _native
except (ImportError, AttributeError):  # 未编译 -> 使用纯 NumPy 参考实现
    _native = None

if _native is not None:
    # 数值语义经实测与参考实现一致(含非 2 的幂零填充、两种相位约定),
    # 只换更快的计算内核。
    fht_forward = _native.fht_forward
    fht_inverse = _native.fht_inverse
    hilbert_transform = _native.hilbert_transform


def _hilbert(signal: np.ndarray, matlab_phase: bool = False) -> np.ndarray:
    """
    对实信号做 Hilbert 变换, 返回解析信号。
    这是 Modal-Decomposition 库对外的统一入口。
    matlab_phase=True 时与 MATLAB / scipy 的相位约定一致。
    """
    signal = _as_real_1d(signal)
    return hilbert_transform(signal, matlab_phase=matlab_phase)
