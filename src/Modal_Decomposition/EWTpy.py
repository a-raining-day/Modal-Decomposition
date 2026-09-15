"""
EWTpy 适配层 —— 可选第三方后端接口

把 ``ewtpy`` (Gilles 的 EWT 参考实现) 作为**可选后端**接进本库: 本模块只提供
``Class.EWTpy`` / ``Function.EWTpy`` 这一个入口, 参数与返回结构与 ewtpy 的
``EWT1D`` 一一对应, 便于与原实现逐位对照。

与 ``EWT`` (自研实现, 本库默认) 的关系
------------------------------------
* ``EWT``   —— 滤波器组/边界检测/变换全部在本库内完成, **不依赖 ewtpy**;
* ``EWTpy`` —— 本模块, 直接调用 ewtpy, **ewtpy 未安装时只有本入口不可用**。

与直接调用 ewtpy 的唯一差异 (有意为之)
------------------------------------
* ``IMFs`` / ``mfb`` / ``boundaries`` 与 ewtpy 的输出**逐位相同**;
* ``Res`` 按**本库契约**取 ``S − ΣIMFs``, 而不是 ewtpy 的最后一条频带 ——
  ewtpy 是幅度型滤波器组, 过渡带内 ``ΣH`` 可达 ``√2``, 故 ``ΣIMFs + Res ≠ S``
  (实测偏差 1.5e-2 ~ 4.1e-1), 而本库其余方法都满足 ``reconstruct()`` 精确。
  ewtpy 原样的最后一条带仍保留在 ``info["ewtpy_residual"]``, 原始约定的带和
  偏差也在 ``info["ewtpy_band_sum_error"]``, 需要逐位对照时直接取用即可。

可选依赖的落地方式
------------------
``ewtpy`` 是 optional-dependencies (见 ``pyproject.toml``), 因此:

* **本模块顶层不 import ewtpy** —— 导入 ``Modal_Decomposition`` 不需要它;
* 真正调用 ``decompose`` 时才经 ``Base.Cache.cache.import_module("ewtpy")``
  惰性导入并注册进进程级 import 缓存 (键 ``ConstDefine.CACHE_KEY["ewtpy"]``);
* 缺库时 ``ImportError`` 原样传播且**不被缓存**, 报错文本附可直接照抄的安装
  命令; 装上之后再次调用即可成功, 无需重启进程。

References
----------
10.48550/arXiv.2304.06274
"""

import sys

import numpy as np
from dataclasses import dataclass
from typing import ClassVar

from .Base import Config, Decomposer, DecompositionResult
from .Base.Cache import cache
from .Base.ConstDefine import CACHE_KEY
from ._Registry import register_class
from .Utils import Check_Time_and_Signal

__all__ = ["EWTpy", "EWTpyConfig"]

#: ewtpy 的可选依赖名 (用于拼装安装提示)。
_PIP_PACKAGE = "ewtpy"


def _load_ewtpy():
    """
    取 ``ewtpy`` 模块 (惰性 + 进程内唯一); 缺库时报错并附安装命令。

    走 ``cache.import_module``: 首次调用 import 并注册, 之后直接返回缓存实例;
    ``ImportError`` 原样传播且不缓存 (装好后再调用即可成功)。
    """
    try:
        return cache.import_module(
            CACHE_KEY["ewtpy"],
            description="ewtpy: EWT 参考实现 (Modal_Decomposition.EWTpy 可选后端)",
        )
    except ImportError as exc:
        raise ImportError(
            "EWTpy 需要可选的第三方依赖 'ewtpy', 但它未安装; 请先安装:\n"
            f"    {sys.executable} -m pip install {_PIP_PACKAGE}\n"
            "或改用自研实现 EWT (不依赖 ewtpy):\n"
            "    Modal_Decomposition.Class.EWT(...).decompose(S)"
        ) from exc


@dataclass(frozen=True, kw_only=True)
class EWTpyConfig(Config):
    """
    Effective parameters of an EWTpy (ewtpy-backed) run.
    """
    N: int
    log: int
    detect: str
    completion: int
    reg: str
    lengthFilter: int
    sigmaFilter: int


@register_class("EWTpy")
class EWTpy(Decomposer):
    """
    Empirical Wavelet Transform, backed by the optional ``ewtpy`` package.

    与自研的 :class:`Modal_Decomposition.EWT.EWT` 并列: 本类只做参数转发与结果
    归一化 (``ewtpy`` 的 modes 按列返回, 这里转置成 ``(n_modes, N)`` 的行式布局,
    末行作为 ``Res``), 用于与原实现对照或复现 ewtpy 的结果。
    """

    name: ClassVar[str] = "EWTpy"

    def __init__(
        self,
        N: int = 5,
        log: int = 0,
        detect: str = "locmax",
        completion: int = 0,
        reg: str = "average",
        lengthFilter: int = 10,
        sigmaFilter: int = 5,
    ):
        """
        Parameters
        ----------
        N : int
            Number of modes (ewtpy 的频带数).
        log : int
            Logarithm of the number of Fourier bounds (``1`` = 对幅度谱取对数).
        detect : str
            Boundary detection method: ``"locmax"`` / ``"locmaxmin"`` /
            ``"locmaxminf"``.
        completion : int
            Whether to complete the boundary set (ewtpy 的 ``completion``).
        reg : str
            Regularization of the spectrum before peak picking:
            ``"average"`` / ``"gaussian"`` (或其它值 = 不平滑)。
        lengthFilter : int
            Length of the regularization filter.
        sigmaFilter : int
            Standard deviation of the Gaussian regularization filter.
        """
        self.N = N
        self.log = log
        self.detect = detect
        self.completion = completion
        self.reg = reg
        self.lengthFilter = lengthFilter
        self.sigmaFilter = sigmaFilter

        self._degraded = False             # ewtpy 是否缺失 (装饰结果信息)

    def decompose(self, S, T=None, fs: float = 1.0) -> DecompositionResult:
        """
        Decompose the signal into modes and a residual using ewtpy.

        The time axis is validated but not used by the algorithm (与 ewtpy 一致:
        ``fs`` 不参与边界检测, 谱轴按样本数归一)。``fs`` 参数**被接受但不使用**
        —— 仅为与自研 :meth:`Modal_Decomposition.EWT.EWT.decompose` 的调用签名
        对齐, 使两者可以互换而不必改调用点。
        """
        ewt1d = _load_ewtpy().EWT1D

        S, T, _N = Check_Time_and_Signal(S, T, ndim={1}, method=self.name)

        ewt, mfb, boundaries = ewt1d(
            np.asarray(S, dtype=np.float64), self.N, self.log, self.detect,
            self.completion, self.reg, self.lengthFilter, self.sigmaFilter,
        )
        ewt = np.asarray(ewt).T
        mfb = np.asarray(mfb).T

        imfs = ewt[:-1, :]
        # 规范化残差: ewtpy 直接把最后一条频带当残差, 而它是**幅度型**滤波器组,
        # 过渡带内 ΣH 可达 √2, 于是 ΣIMFs + Res ≠ S (实测 1.5e-2 ~ 4.1e-1) ——
        # 本库其余方法都满足 reconstruct() 精确, 故这里按库内契约定残差
        # Res = S − ΣIMFs, 而**保留 ewtpy 原样的最后一条带**到 info 里
        # (info["ewtpy_residual"]), 以保证"模态逐位等同 ewtpy"与
        # "重构契约成立"两者兼得。
        res = np.asarray(S, dtype=np.float64) - imfs.sum(axis=0)
        norm_s = float(np.linalg.norm(S))

        return DecompositionResult(
            imfs,
            res,
            {
                "mfb": mfb,
                "boundaries": boundaries,
                "backend": "ewtpy",
                "ewtpy_residual": ewt[-1, :],       # ewtpy 自己的"残差"(最后一带)
                "residual_convention": "S - sum(IMFs)",
                # ewtpy 原始约定的带和偏差 (跨实现对比时可直接引用)
                "ewtpy_band_sum_error": (
                    float(np.linalg.norm(ewt.sum(axis=0) - np.asarray(S, dtype=np.float64)) / norm_s)
                    if norm_s > 0.0 else 0.0
                ),
            },
            EWTpyConfig(
                N=self.N,
                log=self.log,
                detect=self.detect,
                completion=self.completion,
                reg=self.reg,
                lengthFilter=self.lengthFilter,
                sigmaFilter=self.sigmaFilter,
            ),
        )
