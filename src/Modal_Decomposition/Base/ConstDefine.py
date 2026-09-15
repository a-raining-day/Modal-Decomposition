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
    "DEFAULT_FILL_CHUNK_ELEMS",
    "VMD_MIN_SAMPLES",
    "VMD_UHAT_INFO_LIMIT",
    "VMD_PEAK_INIT_LIMIT",
    "VMD_CHUNK_WORK_BYTES",
    "EWT_CHUNK_WORK_BYTES",
    "EWT_SLEPIAN_MAX_SAMPLES",
    "SLEPIAN_BACKEND_LIST",
    "SLEPIAN_BACKEND",
    "SLEPIAN_NUMPY_MAX_BYTES",
    "SLEPIAN_SMALL_N_ORDER",
    "SLEPIAN_CACHE",
    "SLEPIAN_CACHE_SIZE",
    "SLEPIAN_CACHE_MAX_BYTES",
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

#: "auto" 在小数组 (< BIG_ARRAY) 上选用的后端。默认 numpy: 零可选依赖、冷态最快;
#: pyfftw 暖态更快但不是声明依赖, 需要时显式传 mod="pyfftw" (见 docs/FFT_Backend_Report.md)。
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
definition: for VMD
"""
#: VMD 最短信号长度 (再短则镜像延拓与 K 个非退化模态都无意义)。
VMD_MIN_SAMPLES = 8

#: VMD 诊断谱 ``info["u_hat"]`` 的体积上限: 超过则不再计算 (它是 K·N 复数)。
VMD_UHAT_INFO_LIMIT = 64 * SIZE["1MB"]

#: VMD ``init_mod="peak"`` 需要的整谱幅度上限: 超过则退化为 uniform 初值。
VMD_PEAK_INIT_LIMIT = 256 * SIZE["1MB"]

#: VMD 分块引擎的目标单块工作集 (字节): 据此按 K 反推默认块长。
VMD_CHUNK_WORK_BYTES = 64 * SIZE["1MB"]

"""
definition: for EWT
"""
#: EWT 分块滤波的目标单块工作集 (字节): 据此按算子每元素临时量反推默认块长
#: (滤波器组体积达到 ``BIG_ARRAY`` 时启用分块路径, 见 ``EWT``)。
EWT_CHUNK_WORK_BYTES = 64 * SIZE["1MB"]

#: EWT ``pre_deal="Slepian-Optimize"`` 的样本数上限: ``scipy.signal.windows.dpss``
#: 的代价随 N 增长过快 (构造 N×N 三对角矩阵的特征分解), 超过该长度时该分支降级
#: 为普通 ``|rfft|`` (并在 ``info["slepian_degraded"]`` 标记)。
EWT_SLEPIAN_MAX_SAMPLES = 1 << 15       # 32768 样本

"""
definition: for slepian
"""
#: Slepian 后端 canonical 名单 (``Utils.Slepian`` 的 ``mod`` 取值; "auto" 额外允许)。
SLEPIAN_BACKEND_LIST = \
[
    "numpy",
    "scipy",
    "C",
]

#: 默认后端: 实测最优者 (稳态最快 + 峰值内存最低 + 零构建), 见
#: docs/Slepian_Backend_Report.md §6。备选: "C" (冷启动最快、调用路径不依赖 scipy,
#: 自研实现) 与 "numpy" (无 scipy 依赖, 但有 N 上限); 用 mod="C"/"auto" 可切换。
SLEPIAN_BACKEND = "scipy"

#: numpy 后端的稠密矩阵内存预算 (字节)。折半后需对 (N/2)×(N/2) 稠密矩阵做 eigh,
#: 故 N 稍大就会 GB 级分配 (N=65536 → 8.6 GB); 超过该预算时 numpy 后端直接报错并
#: 提示改用 "C"/"scipy" (两者的内存都是 O(N))。对应样本上限 N_max = 2*sqrt(B/8)。
SLEPIAN_NUMPY_MAX_BYTES = 256 * SIZE["1MB"]     # 256 MB -> N_max = 8192

#: scipy 后端的固有规模限制 ``scipy.signal.windows.dpss`` 要求 ``NW < M/2``
#: (即 ``2*halfBW < N``); 不满足时该后端必然抛 ValueError, 此时按下面的顺序改用
#: 其他后端 (默认参数 NW=3.0 时即 N <= 6)。
SLEPIAN_SMALL_N_ORDER = \
[
    "C",        # 小 N 实测 4.5-6.7 us / 2.3-3.2 KB, 比 numpy 快 7-10x、内存小 2.4x
    "numpy",
]

#: Slepian 的 **Tier-1 进程内缓存** (纯函数记忆化): 相同 (后端, N, NW 精确位, 阶数,
#: sym, norm, 是否需要集中比) 的结果直接复用。命中返回副本, 主副本只读。
SLEPIAN_CACHE = True

#: 缓存条数上限 (LRU)。
SLEPIAN_CACHE_SIZE = 8

#: 缓存总字节上限: 超出按 LRU 淘汰; 单条超过该值则不缓存 (只算不存)。
SLEPIAN_CACHE_MAX_BYTES = 256 * SIZE["1MB"]

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
    # 可选第三方后端: EWTpy 适配层用它经 cache.import_module 惰性导入 ewtpy
    # (未安装时只有 EWTpy 入口不可用, EWT 自研实现不受影响)。
    "ewtpy": "ewtpy",
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