"""
Size constants shared by the whole package.

所有涉及分块 / 内存策略的大小常量统一从本字典取用
(见 ``Utils.Chunk`` / ``Utils.Check`` / ``Utils.Memory`` /
``Utils.Monotonicity``), 避免各模块各自书写魔数。
"""

__all__ = [
    "SIZE",
    "SPLINE_KIND",
    "DEFAULT_NUMPY_TYPE",
    "CACHE_KEY",
    "HILBERT_BACKEND",
    "BIG_ARRAY",
    "FFT_BACKEND",
    "FFT_BACKEND_SMALL",
    "FFT_BACKEND_BIG",
    "FFT_BACKEND_LIST",
    "FFT_BACKEND_ALIAS",
    "FFT_PIP_PACKAGE",
    "FFT_THREAD_MIN_ELEMS",
    "FFT_TILED_MIN_ELEMS",
    "MIN_CHUNK_ELEMS",
    "ADAPT_MIN_BYTES",
    "DEFAULT_FILL_CHUNK_ELEMS"
]

import numpy as np

"""
definition: size of arr
"""
SIZE = \
{
    "1KB": 1024,            # 2**10 bytes
    "1MB": 1024 ** 2,       # 2**20 bytes
    "1GB": 1024 ** 3,       # 2**30 bytes
}

"""
definition: big array
"""
#: 大数组分界 (字节)。凡"是否按大数组处理"的判断一律引用本常量, 不再各自写魔数
BIG_ARRAY = 300 * SIZE["1MB"]       # 314572800 bytes

"""
definition: for FFT
"""
#: FFT 后端 canonical 名单 (``Utils.FFT`` 的 ``mod`` 取值; "auto" 额外允许)。
FFT_BACKEND_LIST = \
[
    "numpy",
    "scipy",
    "pyfftw",
    "tiled",
    "cupy",
]

#: 默认 FFT 后端。"auto" = 按数组体积分流: ``< BIG_ARRAY`` 用 ``FFT_BACKEND_SMALL``, 否则用 ``FFT_BACKEND_BIG``。
FFT_BACKEND = "auto"

#: "auto" 在小数组 (< BIG_ARRAY) 上选用的后端。批量/热循环场景可改 "pyfftw" (暖态 10MB 快 5x、100MB 快 3.1x, 但冷态要付 ~0.15s 规划)
FFT_BACKEND_SMALL = "numpy"

#: "auto" 在大数组 (>= BIG_ARRAY) 上选用的后端。实测 >=500MB 时 numpy 既更快更省内存
#: (1GB: 2.87s/4106MB vs pyfftw 5.71s/6796MB)。
FFT_BACKEND_BIG = "numpy"

#: FFT 后端别名 → canonical 名 (Utils.FFT 接受这些写法)。
FFT_BACKEND_ALIAS = \
{
    "np": "numpy",
    "fftw": "pyfftw",
    "chunked": "tiled",
    "gpu": "cupy",
}

#: FFT 后端缺库时提示安装的 pip 包名 (导入名 != 包名: cupy 的 wheel 叫 cupy-cudaXXx)。
FFT_PIP_PACKAGE = \
{
    "scipy": "scipy",
    "pyfftw": "pyfftw",
    "cupy": "cupy-cuda12x",
}

#: FFT 尺寸阈值 (元素数): 低于 FFT_THREAD_MIN_ELEMS 不起多线程 (线程开销盖过收益);
#: 低于 FFT_TILED_MIN_ELEMS 不走四步分块 (不如直算)。
FFT_THREAD_MIN_ELEMS = 1 << 20      # 1M 元素
FFT_TILED_MIN_ELEMS = 1 << 18       # 256K 元素

"""
definition: for spline
"""
SPLINE_KIND = \
[
    "UnivariateSpline",
    "CubicSpline",
    "PCHIP",
    "Akima",
]

"""
definition: default type of numpy type
"""
DEFAULT_NUMPY_TYPE = np.float64

"""
definition: cache
"""
CACHE_KEY = \
{
    "scipy": \
        {
            "signal": "scipy.signal",
            "interpolate": "scipy.interpolate",
        },
    "fft": \
        {
            "scipy": "scipy.fft",
            "pyfftw": "pyfftw.interfaces.numpy_fft",
            "pyfftw_cache": "pyfftw.interfaces.cache",
            "cupy": "cupy",
        },
}

"""
definition: for hilbert
"""
HILBERT_BACKEND = \
[
    "Scipy",
    "Fourth-order Wave",
    "FHT",
    "SB-Hilbert",
    "Kramers-Kronig",
    "FIR",
    "HST",
    "FFT",
    "Optional-order FIR",
]

"""
defintion: chunk
"""
#: 自适应收缩的块大小下限 (元素数) = 4KB 元素。
MIN_CHUNK_ELEMS = 4 * SIZE["1KB"]

#: 输入低于该字节数时不咨询全局内存策略 (直接沿用调用方块大小)。
ADAPT_MIN_BYTES = 64 * SIZE["1MB"]

#: 流式填充的默认块大小 (元素数) = 8M 元素。
DEFAULT_FILL_CHUNK_ELEMS = 8 * SIZE["1MB"]