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
#
# ---------------------------------------------------------------------------
# 本项目（Modal-Decomposition）对上游代码的修改记录
# ---------------------------------------------------------------------------
#   上游 = 上述 GitHub 仓库拆出的 SAO C 代码 (transforms.c / transforms.h /
#          am_sysdep.c / am_sysdep.h / math_const.h)。
#
#   1. **C 源码 (transforms.c 等) 未做任何修改** —— 逐字节与上游一致（见
#      README "Acknowledgement" 的 "Modifications" 段）。
#   2. 本 .pyx 包装层为本项目所写（上游不含 .pyx），此处记录本轮优化：
#      2026-10-19  `hilbert_transform()` 改为零拷贝 —— 直接分配 complex128
#      输出数组、以其 float64 视图作为 SAO `hilbert()` 的 2n-double 交错
#      缓冲原地调用，删除了旧版"2n double 临时缓冲 + 两次跨步拷贝拼复数"
#      （去掉 1 次分配与 2 次跨步搬运，见 docs/test/Hilbert_Backend_Report.md §7）。
#   3. 编译选项由 setup.py 控制（本项目文件），未改动 C 语义。
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

    实现（本项目对包装层的优化，SAO C 源码未改）:
      SAO 的 ``hilbert()`` 接受 ``2n`` 个 double 的复数交错缓冲并**原地**改写。
      复数数组在内存里就是实部/虚部交错的 double，因此这里直接分配
      ``complex128`` 输出数组、用它的 ``float64`` 视图当缓冲，让 C 内核原地
      写回 —— 省掉旧版 "``2n`` double 临时缓冲 + 两次跨步拷贝拼复数"
      （合计 3 次跨步内存搬运）。剩下的一次跨步写只用于填实部。
    """
    cdef int n_orig = real_signal.shape[0]
    cdef int n = _next_pow2(n_orig)
    # 复数输出数组 = 2n 个交错 double，正好是 hilbert() 要的缓冲区布局
    cdef cnp.ndarray[cnp.complex128_t, ndim=1] analytic = np.zeros(n, dtype=np.complex128)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] flat = analytic.view(np.float64)
    cdef double[::1] view = flat

    # 实部槽位填信号 (0,2,4,...)，其余(虚部与零填充)保持 0
    flat[0:2 * n_orig:2] = real_signal

    with nogil:
        hilbert(&view[0], n)

    if matlab_phase:
        analytic.imag *= -1  # 对齐 MATLAB 的 -90°

    return analytic[:real_signal.shape[0]]
