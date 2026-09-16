"""
FFT 后端分发 —— 对外只有一个 ``fft`` 类 (全部方法为静态方法, 直接 ``fft.xxx(...)``)。

为什么需要这个模块
------------------
1. ``np.fft`` 单线程; ``scipy.fft`` 有 ``workers``, ``pyfftw`` 有 FFTW planner/wisdom,
   ``cupy`` 可上 GPU, 超大数组还需要分块 (四步 FFT) 路径;
2. 需要一处统一入口, 让换后端只改常量 (``Base.ConstDefine``) 而不动算法代码。

分支 (canonical 名, 见 ``Base.ConstDefine.FFT_BACKEND_LIST``)
--------------------------------------------------------------
============  ==========================================================
``"numpy"``   ``np.fft`` —— 零依赖基线, 也是数值参考实现 (默认)
``"scipy"``   ``scipy.fft`` —— 支持 ``workers`` 多线程与 float32
``"pyfftw"``  ``pyfftw.interfaces.numpy_fft`` —— FFTW (暖态最快, 冷态要付规划)
``"tiled"``   四步 (four-step) 分块 FFT, 仅复数 ``fft``/``ifft``; 其余回落直算
``"cupy"``    ``cupy.fft`` —— GPU (含往返传输), 仅显式请求时导入
============  ==========================================================

``mod="auto"`` (默认, 取值自 ``Base.ConstDefine.FFT_BACKEND``) 按体积分流:
``数组字节数 < BIG_ARRAY`` → ``FFT_BACKEND_SMALL``, 否则 ``FFT_BACKEND_BIG``。

后端库由 import cache 把守 (``docs/CacheConventions.md`` §4.1)
--------------------------------------------------------------
不写裸 ``import scipy`` / ``import pyfftw`` / ``import cupy``: 一律经
``cache.import_module(...)`` (键见 ``Base.ConstDefine.CACHE_KEY["fft"]``), 于是每个
第三方库每进程只导入一次、异常不缓存可重试。``available_backends()`` 默认用
``importlib.util.find_spec`` 判断是否安装 —— **不导入**, 不会被 cupy 的慢导入卡住。

缺库时的行为
------------
显式使用未安装的后端 → ``ImportError``, 消息附可直接照抄的安装命令;
默认路径 (``mod=None``/``"auto"``) 选中未安装的后端 → 降级 ``numpy`` 并发一次
``UserWarning`` (默认路径不该因可选依赖缺失而抛错)。

实测依据 (冷/暖态、各尺寸、各后端) 见 ``docs/FFT_Backend_Report.md``。

Python version: 3.10

Lib and Version:
    numpy - 2.2.6 (必需)
    scipy / pyfftw / cupy-cuda12x (可选)

Only accessed by: 全库 FFT 调用点 (经 ``Utils.get_fft()`` 惰性注册)

Modify:
    2026.3.4
"""

import os
import importlib.util
import math
import sys
import warnings
from typing import Optional

import numpy as np

from ..Base.Cache import cache
from ..Base.ConstDefine import (
    BIG_ARRAY,
    CACHE_KEY,
    FFT_BACKEND,
    FFT_BACKEND_ALIAS,
    FFT_BACKEND_BIG,
    FFT_BACKEND_LIST,
    FFT_BACKEND_SMALL,
    FFT_PIP_PACKAGE,
    FFT_THREAD_MIN_ELEMS,
    FFT_TILED_MIN_ELEMS,
)
from .Memory import resolve_workers

__all__ = ["fft"]


class fft:
    """
    傅里叶变换工具

    用法::

        from Modal_Decomposition.Utils import fft

        fft.fft(x)                     # 默认变换 (复数 FFT)
        fft.ifft(X); fft.rfft(x); fft.irfft(X, n=N)
        fft.fftshift(X); fft.ifftshift(X)
        fft.rfft(x, mod="pyfftw", workers=8)     # 单次指定后端
        fft.resolve_backend(None, x.nbytes)      # 该体积会选中哪个后端
        fft.available_backends(); fft.describe()

    ``mod`` / ``workers`` / ``planner_effort`` 是**工具层**参数: ``workers`` 只对
    ``scipy`` / ``pyfftw`` 生效, 其它后端忽略它 (而不是报错), 调用方无需按后端分支。
    """

    # ------------------------------------------------------------------ #
    # 变换
    # ------------------------------------------------------------------ #
    @staticmethod
    def fft(a, mod=None, **kwargs) -> np.ndarray:
        """复数 FFT (``axis`` / ``n`` 等参数透传)。"""
        return fft._dispatch("fft", a, mod=mod, **kwargs)

    @staticmethod
    def ifft(a, mod=None, **kwargs) -> np.ndarray:
        """复数逆 FFT。"""
        return fft._dispatch("ifft", a, mod=mod, **kwargs)

    @staticmethod
    def rfft(a, mod=None, **kwargs) -> np.ndarray:
        """实数 FFT (半谱, 长度 ``n//2+1``)。"""
        return fft._dispatch("rfft", a, mod=mod, **kwargs)

    @staticmethod
    def irfft(a, mod=None, **kwargs) -> np.ndarray:
        """实数逆 FFT (``n`` 指定输出长度)。"""
        return fft._dispatch("irfft", a, mod=mod, **kwargs)

    # ------------------------------------------------------------------ #
    # 频谱搬移 (与后端无关的纯索引重排)
    # ------------------------------------------------------------------ #
    @staticmethod
    def fftshift(a, axes=None) -> np.ndarray:
        """把零频分量搬到中心 (等价 ``np.fft.fftshift``)。"""
        return np.fft.fftshift(a, axes=axes)

    @staticmethod
    def ifftshift(a, axes=None) -> np.ndarray:
        """``fftshift`` 的逆操作 (等价 ``np.fft.ifftshift``)。"""
        return np.fft.ifftshift(a, axes=axes)

    # ------------------------------------------------------------------ #
    # 后端解析与探测
    # ------------------------------------------------------------------ #
    @staticmethod
    def resolve_backend(mod=None, nbytes: int = 0) -> str:
        """
        解析生效后端名 (不会是 ``"auto"``)。

        ``mod=None`` → 用 ``FFT_BACKEND``; ``"auto"`` 按体积分流
        (``< BIG_ARRAY`` → ``FFT_BACKEND_SMALL``, 否则 ``FFT_BACKEND_BIG``)。
        """
        mod = FFT_BACKEND if mod is None else mod
        name = fft._canonical(mod)
        if name == "auto":
            small = int(nbytes) < BIG_ARRAY
            return fft._canonical(FFT_BACKEND_SMALL if small else FFT_BACKEND_BIG)
        return name

    @staticmethod
    def available_backends(probe: bool = False) -> dict:
        """
        各后端可用性: ``{后端名: True / False / "installed-not-probed" / 异常描述}``。

        ``probe=False`` (默认) 只查 ``find_spec`` (不导入, 不会被 cupy 卡住);
        ``probe=True`` 才真正尝试导入 (cupy 还会检查 GPU 可见性)。
        """
        out = {"numpy": True, "tiled": fft._scipy_installed()}
        for name in ("scipy", "pyfftw", "cupy"):
            if importlib.util.find_spec(name) is None:
                out[name] = False
                continue
            if not probe:
                out[name] = "installed-not-probed"
                continue
            try:
                {"scipy": fft._load_scipy, "pyfftw": fft._load_pyfftw,
                 "cupy": fft._load_cupy}[name]()
                out[name] = True
            except Exception as exc:
                out[name] = "%s: %s" % (type(exc).__name__, str(exc)[:80])
        return out

    @staticmethod
    def describe() -> str:
        """一行概览: 默认后端 + 各分支可用性 (供日志/报告引用)。"""
        avail = fft.available_backends()
        parts = []
        for name in FFT_BACKEND_LIST:
            state = avail.get(name)
            mark = "OK" if state is True else ("--" if state is False else str(state))
            parts.append("%s=%s" % (name, mark))
        return "fft default=%s small=%s big=%s (BIG_ARRAY=%.0fMB) | %s" % (
            FFT_BACKEND, FFT_BACKEND_SMALL, FFT_BACKEND_BIG,
            BIG_ARRAY / 1024 ** 2, ", ".join(parts),
        )

    # ------------------------------------------------------------------ #
    # 内部: 后端库取用 (全部经 import cache)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _canonical(mod) -> str:
        """别名 → canonical 后端名; 未知名字报错并列出可选项。"""
        key = str(mod).strip().lower() if isinstance(mod, str) else mod
        if key in FFT_BACKEND_ALIAS:
            key = FFT_BACKEND_ALIAS[key]
        if key not in FFT_BACKEND_LIST and key != "auto":
            raise ValueError(
                "unknown FFT backend %r; expected one of %s or 'auto'"
                % (mod, list(FFT_BACKEND_LIST))
            )
        return key

    @staticmethod
    def _scipy_installed() -> bool:
        return importlib.util.find_spec("scipy") is not None

    @staticmethod
    def _installed(name: str) -> bool:
        """后端是否已安装 (只查 find_spec, 不导入)。"""
        if name == "numpy":
            return True
        if name == "tiled":
            return fft._scipy_installed()
        return importlib.util.find_spec(name) is not None

    @staticmethod
    def _install_hint(backend: str) -> str:
        """缺库报错文本 (含可直接照抄的安装命令)。"""
        return (
            "FFT backend %r 所需的库未安装; 请先安装:\n"
            "    %s -m pip install %s"
            % (backend, sys.executable, FFT_PIP_PACKAGE.get(backend, backend))
        )

    @staticmethod
    def _load_scipy():
        """取 ``scipy.fft``。"""
        return cache.import_module(
                CACHE_KEY["fft"]["scipy"],
                description="scipy.fft: 多线程 / float32 FFT 后端 (Utils.FFT)",
            )

    @staticmethod
    def _load_pyfftw():
        """取 ``pyfftw.interfaces.numpy_fft`` 并启用其 wisdom 缓存 (plan 复用)。"""
        pnf = cache.import_module(
            CACHE_KEY["fft"]["pyfftw"],
            description="pyfftw.interfaces.numpy_fft: FFTW 后端 (Utils.FFT)",
        )
        cache.import_module(
            CACHE_KEY["fft"]["pyfftw_cache"],
            description="pyfftw.interfaces.cache: FFTW plan 缓存开关 (Utils.FFT)",
        ).enable()
        return pnf

    @staticmethod
    def _load_cupy():
        """取 ``cupy`` 并确认有可见 GPU (仅显式请求时调用; 首次导入可能很慢)。"""
        cp = cache.import_module(
            CACHE_KEY["fft"]["cupy"],
            description="cupy: GPU FFT 后端 (Utils.FFT, 仅显式请求)",
        )

        try:
            count = cp.cuda.runtime.getDeviceCount()
        except Exception as exc:
            raise RuntimeError(
                "FFT backend 'cupy' 已安装但 CUDA 运行时不可用 (%s: %s); "
                "请检查驱动与 cupy-cudaXXx 版本" % (type(exc).__name__, exc)
            ) from exc
        if not count:
            raise RuntimeError(
                "FFT backend 'cupy' 已安装但看不到可用 GPU (getDeviceCount()==0)"
            )
        return cp

    # ------------------------------------------------------------------ #
    # 内部: 各分支实现
    # ------------------------------------------------------------------ #
    @staticmethod
    def _threads_for(nbytes: int, workers: Optional[int]) -> int:
        """``workers`` → 线程数 (统一走 ``Utils.Memory.resolve_workers``)。

        - ``workers=None``: 默认策略 —— 小数组单线程 (线程开销盖过收益),
          大数组用默认核数 ``CPU_DEFAULT_RATIO`` × 可用核 (默认 **不占满**);
        - ``workers=-1``: 用满当前可用核;
        - ``workers=N``: 恰好 N 个线程 (上限为可用核数)。
        """
        if int(nbytes) < FFT_THREAD_MIN_ELEMS * 8:
            return 1
        return resolve_workers(workers)

    @staticmethod
    def _run_numpy(op: str, a: np.ndarray, **kw):
        # workers / planner_effort 只对 scipy / pyfftw 有意义, numpy 后端显式忽略
        # (否则 fft.rfft(x, workers=8) 在默认后端下会抛 TypeError)。
        for name in ("workers", "planner_effort"):
            kw.pop(name, None)
        return getattr(np.fft, op)(a, **kw)

    @staticmethod
    def _run_scipy(op: str, a: np.ndarray, workers=None, **kw):
        sf = fft._load_scipy()
        n = int(a.size) * int(a.dtype.itemsize)
        return getattr(sf, op)(a, workers=fft._threads_for(n, workers), **kw)

    @staticmethod
    def _run_pyfftw(op: str, a: np.ndarray, workers=None,
                    planner_effort: str = "FFTW_ESTIMATE", **kw):
        pnf = fft._load_pyfftw()
        n = int(a.size) * int(a.dtype.itemsize)
        return getattr(pnf, op)(
            a, threads=fft._threads_for(n, workers), planner_effort=planner_effort, **kw
        )

    @staticmethod
    def _run_cupy(op: str, a: np.ndarray, **kw):
        cp = fft._load_cupy()
        host = cp.asarray(a)                     # 主机 → 设备
        out = getattr(cp.fft, op)(host, **kw)    # 设备上变换
        return cp.asnumpy(out)                   # 设备 → 主机 (库契约: 返回主机数组)

    @staticmethod
    def _factor_pair(n: int):
        """给 ``n`` 找一个接近 ``sqrt(n)`` 的因子对; 素数返回 None。"""
        if n < 2:
            return None
        for n1 in range(int(math.isqrt(n)), 1, -1):
            if n % n1 == 0:
                return n1, n // n1
        return None

    @staticmethod
    def _tiled_fft_1d(a: np.ndarray, inverse: bool, workers=None):
        """
        四步 (four-step) 分块 FFT: ``N = N1·N2``, 逐轴变换 + 旋转因子。

        ``X[k2·N1 + k1] = FFT_{n2}[W_N^{n2·k1}·FFT_{n1}[a]]``: 用两次小尺寸变换完成
        一次大长度变换, 峰值只多一个 ``(N1, N2)`` 中间量, 可在 memmap 上就地读写。
        逆变换用 ``ifft(x) = conj(fft(conj(x)))/N`` 复用前向实现。长度不是合数或过短
        时返回 ``None``, 由调用方回落直算。
        """
        n = int(a.shape[-1])
        pair = fft._factor_pair(n)
        if pair is None or n < FFT_TILED_MIN_ELEMS:
            return None

        n1, n2 = pair
        sf = fft._load_scipy()               # 逐轴变换用 scipy (可多线程)
        th = fft._threads_for(n * a.dtype.itemsize, workers)
        batch_shape = a.shape[:-1]

        if inverse:
            return np.conj(fft._tiled_fft_1d(np.conj(a), False, workers)) / n

        work = np.array(a.reshape(batch_shape + (n1, n2)), copy=True)
        if work.dtype.kind != "c":
            work = work.astype(np.complex128)
        work = sf.fft(work, axis=-2, workers=th)                # 沿 n1

        k1 = np.arange(n1, dtype=np.float64)[:, None]
        n2i = np.arange(n2, dtype=np.float64)[None, :]
        work *= np.exp((-2j * np.pi / n) * (k1 * n2i))          # 旋转因子 W_N^{n2·k1}

        work = sf.fft(work, axis=-1, workers=th)                # 沿 n2
        out = np.ascontiguousarray(work.swapaxes(-1, -2))       # [k2,k1] 展平 = k2·N1+k1
        return out.reshape(batch_shape + (n,))

    @staticmethod
    def _run_tiled(op: str, a: np.ndarray, workers=None, **kw):
        """``tiled``: 仅复数 ``fft``/``ifft``; 其它情形返回 None 由调用方回落。

        ``rfft`` 不分块的原因: 半谱索引在二维分解下呈阶梯形, 两次变换都需要完整复数
        中间量, 峰值 >= 12N 字节, 比直算 (输入 4N + 半谱 8N) 更大且更慢
        (实测见 docs/FFT_Backend_Report.md)。
        """
        if op not in ("fft", "ifft") or kw.get("n") is not None:
            return None
        return fft._tiled_fft_1d(a, inverse=(op == "ifft"), workers=workers)

    # ------------------------------------------------------------------ #
    # 内部: 统一分发
    # ------------------------------------------------------------------ #
    @staticmethod
    def _dispatch(op: str, a, mod=None, **kwargs):
        """
        解析后端 → 该后端不支持此 op 时回落 → 执行。

        ``mod=None``/``"auto"`` 是**默认路径**: 选中的后端未安装时降级 ``numpy`` 并
        发一次 ``UserWarning``; 显式指定后端时按"缺库 → ImportError + 安装命令"处理。
        """
        arr = np.asarray(a)
        nbytes = int(arr.size) * int(arr.dtype.itemsize)
        name = fft.resolve_backend(mod, nbytes)

        explicit = not (mod is None or fft._canonical(mod) == "auto")
        if not explicit and name != "numpy" and not fft._installed(name):
            warnings.warn(
                "FFT backend %r (来自 FFT_BACKEND 配置) 未安装, 本次降级为 numpy; "
                "安装后可恢复: %s -m pip install %s"
                % (name, sys.executable, FFT_PIP_PACKAGE.get(name, name)),
                UserWarning,
                stacklevel=3,
            )
            name = "numpy"

        if name == "numpy":
            return fft._run_numpy(op, arr, **kwargs)
        if name == "scipy":
            return fft._run_scipy(op, arr, **kwargs)
        if name == "pyfftw":
            return fft._run_pyfftw(op, arr, **kwargs)
        if name == "cupy":
            return fft._run_cupy(op, arr, **kwargs)

        out = fft._run_tiled(op, arr, **kwargs)      # tiled 不支持的情形 → None
        if out is not None:
            return out
        if fft._scipy_installed():
            keep = {k: v for k, v in kwargs.items() if k in ("workers",)}
            return fft._run_scipy(op, arr, **keep)
        drop = ("workers", "planner_effort")
        return fft._run_numpy(op, arr, **{k: v for k, v in kwargs.items() if k not in drop})


if __name__ == "__main__":       # pragma: no cover - 手动自检
    print(fft.describe())
    x = np.random.default_rng(0).standard_normal(1 << 16)
    print("fft.fft   vs np.fft.fft  max|d| = %.2e" % np.abs(fft.fft(x) - np.fft.fft(x)).max())
    print("fft.rfft scipy vs numpy  max|d| = %.2e"
          % np.abs(fft.rfft(x, "numpy") - fft.rfft(x, "scipy")).max())
