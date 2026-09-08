# cython: language_level=3

# ===========================================================================
# 第三方代码声明:
#   本文件的 C 内核(transforms.c / am_sysdep.c 等, 见 _C/_fht/)源自美国
#   Smithsonian Astrophysical Observatory (SAO) 亚毫米接收机实验室的 "am"
#   项目 (S. Paine), 经第三方仓库拆出供 C/C++ 独立使用:
#     https://github.com/waddafunk/Smithsonians_Discrete_Hilbert_Fourier_Hartley_Transforms
#   可自由使用, 但须注明出处与致谢 Smithsonian Astrophysical Observatory
#   (完整许可声明见项目 README "Acknowledgement" 章节)。
#
#   相位约定: SAO 实现的 Hilbert 变换相位为 +90°, 而 MATLAB/SciPy 为 -90°;
#   需要 MATLAB 结果时把解析信号虚部乘 -1, 即本文件的 matlab_phase=True。
# ===========================================================================

import numpy as np
cimport numpy as cnp
from libc.stdlib cimport malloc, free

cnp.import_array()

# 声明外部 C 函数(Cython 3 起必须显式 nogil 才能在 with nogil 中调用)
cdef extern from "transforms.h" nogil:
    void fht_dif(double*, unsigned long) nogil
    void fht_dit(double*, unsigned long) nogil
    void bitrev_permute_real(double*, unsigned long) nogil
    void hilbert(double*, unsigned long) nogil

def _next_pow2(int n):
    """返回 >= n 的最小 2 的幂"""
    cdef int p = 1
    while p < n:
        p <<= 1
    return p

def fht_forward(cnp.ndarray[cnp.float64_t, ndim=1] x not None, bint normalize_order=True):
    """
    前向 FHT。
    参数:
        x: 实信号，长度最好是 2 的幂；否则自动零填充到下一个 2 的幂
        normalize_order: True 时输出转回正常序（调用方通常想要这个）
    返回:
        Hartley 变换结果（实数组）
    """
    cdef int n_orig = x.shape[0]
    cdef int n = _next_pow2(n_orig)
    # 零填充 + 确保 C-contiguous
    cdef cnp.ndarray[cnp.float64_t, ndim=1] buf = np.zeros(n, dtype=np.float64)
    buf[:n_orig] = x
    cdef double[::1] view = buf

    with nogil:
        fht_dif(&view[0], n)
        if normalize_order:
            bitrev_permute_real(&view[0], n)
    return buf

def fht_inverse(cnp.ndarray[cnp.float64_t, ndim=1] x not None, bint input_is_normal_order=True):
    """
    逆 FHT。
    参数:
        input_is_normal_order: True 表示输入是正常序（会先转位反转序再调用 fht_dit）
    """
    cdef int n = _next_pow2(x.shape[0])
    cdef cnp.ndarray[cnp.float64_t, ndim=1] buf = np.zeros(n, dtype=np.float64)
    buf[:x.shape[0]] = x
    cdef double[::1] view = buf

    with nogil:
        if input_is_normal_order:
            bitrev_permute_real(&view[0], n)
        fht_dit(&view[0], n)
    # DHT 是自逆的（乘以 1/n 归一化）
    buf /= n
    return buf

def hilbert_transform(cnp.ndarray[cnp.float64_t, ndim=1] real_signal not None, bint matlab_phase=False):
    """
    计算实信号的 Hilbert 变换，返回解析信号 (实部=原信号, 虚部=Hilbert 变换)。
    相位约定: SAO C 实现为 +90°, 而 MATLAB/SciPy 为 -90° —— 与 MATLAB
    对齐等价于把虚部乘 -1 (matlab_phase=True), 此时输出与
    MATLAB / scipy.signal.hilbert 一致。
    """
    cdef int n_orig = real_signal.shape[0]
    cdef int n = _next_pow2(n_orig)
    # hilbert() 需要 2n 个 double 的复数交错数组; 只把前 n_orig 个实部
    # 槽位填上信号, 其余自动零填充(与 fht_forward/fht_inverse 语义一致)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] z = np.zeros(2 * n, dtype=np.float64)
    z[0:2 * n_orig:2] = real_signal  # 实部
    cdef double[::1] view = z

    with nogil:
        hilbert(&view[0], n)

    if matlab_phase:
        z[1::2] *= -1  # 对齐 MATLAB 的 -90°

    # 返回复数解析信号
    cdef cnp.ndarray[cnp.complex128_t, ndim=1] analytic = \
        np.zeros(n, dtype=np.complex128)
    analytic.real = z[0::2]
    analytic.imag = z[1::2]
    return analytic[:real_signal.shape[0]]