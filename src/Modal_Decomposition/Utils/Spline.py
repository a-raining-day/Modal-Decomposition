"""
Spline interpolation over ``scipy.interpolate`` (统一样条封装)。

为什么需要这个模块
------------------
算法模块里反复出现“用样条拟合包络 / 局部均值 / 插值回退”的样板代码
(见 docs/ComponentStatisticsReport.md 的去重候选清单), 且 ``scipy.interpolate``
各构造器签名不统一。本模块提供:

* 一套**统一的输入校验** (有限值 / 单调递增 / 去重 / 长度一致性 / 维度),
  与 ``Utils.Check`` 的哲学一致;
* 一个**可切换后端**的样条对象: 默认对外表现为
  :class:`scipy.interpolate.UnivariateSpline` (FITPACK), 同时保留
  ``"CubicSpline"`` / ``"PCHIP"`` / ``"Akima"`` 等其它选择;
* 惰性注册进进程级 import 缓存 (``Base.Cache.cache``), 供全库/全局取用。

设计要点
--------
* 默认后端 ``"UnivariateSpline"`` 且默认 ``s=0`` (精确插值, 与其它插值型后端
  语义对齐; 传入 ``s`` 可切换为平滑样条)。scipy 原生的 ``s=None`` 平滑默认
  并不适合做“插值封装”, 故不沿用。
* ``x`` 必须严格递增: 乱序输入会按 ``x`` 排序并发出 ``UserWarning``; 重复
  ``x`` 直接报错 (样条要求唯一节点, 与 scipy 行为一致)。
* 边界外推语义随后端而异且透明代理 (scipy 1.15 实测): UnivariateSpline 恒
  外推; CubicSpline / PCHIP 默认外推, ``extrapolate=False`` 时界外返回 NaN;
  Akima 默认界外返回 NaN, ``extrapolate=True`` 打开外推。封装层不做截断/掩膜。

Cache 集成
----------
本模块自身**不接触缓存** (保持与其它 Utils 工具同一路径): 统一由
``Utils.get_spline()`` 在首次访问时惰性 import 并以键
``"Modal_Decomposition.Utils.Spline"`` 注册进进程级 import 缓存
(``Base.Cache.cache``); 之后任意组件可 ``cache.get(...)`` 获得进程内唯一实例。
"""

from typing import Any, Callable, Union

import numpy as np

__all__ = [
    "Spline",
    "spline",
    "BACKEND_OPTIONS",
]

#: 可用的后端名 (canonical 名称, 大小写/分隔符不敏感, 见 ``_normalize``)。
BACKEND_OPTIONS: tuple[str, ...] = (
    "UnivariateSpline",
    "CubicSpline",
    "PCHIP",
    "Akima",
)

_DEFAULT_BACKEND: str = "UnivariateSpline"


def _normalize(backend: str) -> str:
    """
    归一化后端名: 去空格/下划线/连字符并统一小写。
    """
    return (
        backend.strip()
        .lower()
        .replace("-", "")
        .replace("_", "")
        .replace(" ", "")
    )


_ALIAS_TO_CANONICAL = {_normalize(b): b for b in BACKEND_OPTIONS}
_ALIAS_TO_CANONICAL.update(
    {
        "pchipinterpolator": "PCHIP",
        "akima1dinterpolator": "Akima",
        "unispline": "UnivariateSpline",
        "smoothing": "UnivariateSpline",
        "cubic": "CubicSpline",
    }
)


def _resolve_backend(backend: str) -> str:
    """
    把用户输入解析为 canonical 后端名。

    Raises
    ------
    ValueError
        未知后端名 (附可用列表)。
    """
    key = _ALIAS_TO_CANONICAL.get(_normalize(backend))
    if key is None:
        raise ValueError(
            f"Unknown spline backend {backend!r}; expected one of "
            f"{BACKEND_OPTIONS} (case/separator insensitive)"
        )
    return key


def _validate(x, y) -> tuple[np.ndarray, np.ndarray]:
    """
    统一校验并规范化 ``(x, y)`` 节点对。

    Returns
    -------
    (x64, y64)
        float64、1-D、严格递增、无 NaN/Inf 的 ``x`` 及其对应 ``y``。
        乱序 (但唯一) 的输入按 ``x`` 升序重排并发出 ``UserWarning``。

    Raises
    ------
    ValueError
        空/长度不一致/重复节点/非数值/含 NaN/Inf/维度错误/样本过少。
    """
    xa = np.asarray(x)
    ya = np.asarray(y)

    if xa.ndim != 1 or ya.ndim != 1:
        raise ValueError(
            f"Spline nodes must be 1-D, got x.ndim={xa.ndim}, y.ndim={ya.ndim}"
        )
    if xa.size < 2:
        raise ValueError(f"Spline requires at least 2 nodes, got {xa.size}")
    if xa.size != ya.size:
        raise ValueError(
            f"Length mismatch between x ({xa.size}) and y ({ya.size})"
        )

    try:
        x64 = np.asarray(xa, dtype=np.float64)
        y64 = np.asarray(ya, dtype=np.float64)
    except (TypeError, ValueError):
        raise ValueError("Spline nodes must be numeric (convertible to float64)")

    if not np.all(np.isfinite(x64)) or not np.all(np.isfinite(y64)):
        raise ValueError("Spline nodes must be finite (no NaN/Inf)")

    order = np.argsort(x64, kind="stable")
    if not np.array_equal(x64[order], x64):
        import warnings

        warnings.warn(
            "Spline x is not monotonically increasing; x/y are reordered by ascending x.",
            UserWarning,
            stacklevel=3,
        )
        x64 = x64[order]
        y64 = y64[order]

    if np.any(np.diff(x64) == 0):
        raise ValueError("Spline requires strictly increasing x (duplicate nodes)")

    return x64, y64


def _lazy_import_interpolate():
    """惰性取 scipy.interpolate (仅首次 import 一次)。"""
    try:
        from scipy import interpolate as _interpolate
    except ImportError:
        raise ImportError(
            "Scipy is not installed; Spline requires scipy.interpolate"
        )
    return _interpolate


class Spline:
    """
    Unified spline wrapper over :mod:`scipy.interpolate`.

    Parameters
    ----------
    x, y : array-like
        1-D 节点 (``x`` 须严格递增; 乱序自动重排并告警; 重复节点报错)。
    backend : str
        后端选择, 默认 ``"UnivariateSpline"`` (FITPACK)。可选
        ``"CubicSpline"`` / ``"PCHIP"`` / ``"Akima"`` (大小写/分隔符不敏感)。
    **kwargs
        透传给 scipy 构造器的额外参数。常用:

        * UnivariateSpline: ``k`` (样条阶数, 默认 3), ``s`` (平滑因子,
          默认 0 = 精确插值), ``w`` (权重, 长度须等于节点数);
        * CubicSpline: ``bc_type`` (默认 not-a-knot), ``extrapolate``
          (默认外推; False 时界外求值为 NaN);
        * PCHIP: ``extrapolate`` (默认外推; False 时界外求值为 NaN);
        * Akima: ``extrapolate`` (默认 False → 界外 NaN; True 打开外推)。

    Attributes
    ----------
    backend : str
        Canonical 后端名。
    x_, y_ : np.ndarray
        规范化后的 (严格递增 float64) 节点。
    params : dict
        生效参数快照 (仅记录显式传入项)。
    fitted : object
        scipy 构造的插值对象 (UnivariateSpline / CubicSpline /
        PchipInterpolator / Akima1DInterpolator)。

    Examples
    --------
    >>> from Modal_Decomposition.Utils.Spline import spline
    >>> import numpy as np
    >>> x = np.linspace(0, 10, 11); y = np.sin(x)
    >>> sp = spline(x, y)                    # 默认 UnivariateSpline(s=0)
    >>> np.allclose(sp(np.linspace(0, 10, 1001)), np.sin(np.linspace(0, 10, 1001)), atol=0.05)
    True
    """

    def __init__(
        self,
        x,
        y,
        backend: str = _DEFAULT_BACKEND,
        **kwargs: Any,
    ) -> None:
        canonical = _resolve_backend(backend)
        x64, y64 = _validate(x, y)
        n = x64.size

        interp = _lazy_import_interpolate()
        if canonical == "UnivariateSpline":
            k = int(kwargs.get("k", 3))
            if k < 1:
                raise ValueError(f"UnivariateSpline order k must be >= 1, got {k}")
            if n <= k:
                raise ValueError(
                    f"UnivariateSpline with k={k} requires at least {k + 1} nodes, "
                    f"got {n} (use k=1 for two-point linear interpolation)"
                )
            if "s" not in kwargs:
                kwargs["s"] = 0  # 精确插值默认; 与其它插值型后端语义对齐

        factory = _factory_for(canonical, interp)
        fitted = factory(x64, y64, **kwargs)

        self.backend: str = canonical
        self.x_: np.ndarray = x64
        self.y_: np.ndarray = y64
        self.params: dict = dict(kwargs)
        self.fitted = fitted

    # ------------------------------------------------------------------ #
    # 求值 & 便捷代理
    # ------------------------------------------------------------------ #
    def __call__(self, xi) -> Union[np.ndarray, float]:
        """
        在 ``xi`` 处求值 (标量输入返回 Python float, 数组输入返回同形数组)。
        """
        out = self.fitted(xi)
        try:
            scalar = np.asarray(xi).ndim == 0
        except Exception:  # 无法判维的输入按标量路径处理
            scalar = True
        if scalar and hasattr(out, "item"):
            return out.item()
        return out

    def __repr__(self) -> str:
        return (
            f"<Spline backend={self.backend!r} n={self.x_.size} "
            f"params={self.params!r}>"
        )

    def __getattr__(self, item: str):
        # 代理 scipy 插值对象的能力 (derivative/antiderivative/roots/integral/
        # get_coeffs/get_knots/...): 只在对象真正具备时才透传。
        fitted = object.__getattribute__(self, "fitted")
        if hasattr(fitted, item):
            return getattr(fitted, item)
        raise AttributeError(
            f"{self.__class__.__name__!r} (backend={self.backend!r}) has no "
            f"attribute {item!r}"
        )


def _factory_for(canonical: str, interp) -> Callable:
    """
    返回 canonical 后端对应的 scipy.interpolate 构造器。

    构造器在调用期解析, 避免模块 import 时绑定具体 scipy 符号 (保持惰性)。
    """
    if canonical == "UnivariateSpline":
        return interp.UnivariateSpline
    if canonical == "CubicSpline":
        return interp.CubicSpline
    if canonical == "PCHIP":
        return interp.PchipInterpolator
    if canonical == "Akima":
        return interp.Akima1DInterpolator
    raise ValueError(f"Unreachable: unknown canonical backend {canonical!r}")


def spline(
    x,
    y,
    backend: str = _DEFAULT_BACKEND,
    **kwargs: Any,
) -> Spline:
    """
    Build a unified spline over :mod:`scipy.interpolate`.

    默认后端为 :class:`scipy.interpolate.UnivariateSpline` (``s=0`` 精确插值,
    k=3), 也可通过 ``backend`` 选择其它实现。缓存注册统一由
    ``Utils.get_spline()`` 完成 (首次访问时以键
    ``"Modal_Decomposition.Utils.Spline"`` 注册进 ``Base.Cache.cache``),
    之后全局可经 ``cache.get(...)`` 取用同一实例。

    Parameters
    ----------
    x, y : array-like
        1-D 节点对 (``x`` 严格递增; 乱序自动重排 + UserWarning; 重复节点 /
        NaN / Inf / 空 / 长度不一致均抛 ``ValueError``)。
    backend : str
        ``"UnivariateSpline"`` (默认) | ``"CubicSpline"`` | ``"PCHIP"`` |
        ``"Akima"`` (大小写与分隔符不敏感, 见 ``BACKEND_OPTIONS``)。
    **kwargs
        透传 scipy 构造器参数 (见 :class:`Spline`)。

    Returns
    -------
    Spline
        可调用对象: ``spline(x, y)(xi)`` 直接求值; 也提供 ``derivative`` /
        ``integral`` / ``roots`` 等能力代理 (取决于后端)。

    Notes
    -----
    输入校验汇总 (与 tests/test_spline.py 一致):

    +---------------------------+-------------------------------------------+
    | 情况                      | 行为                                      |
    +===========================+===========================================+
    | 节点数 < 2 / 空           | ValueError                                |
    +---------------------------+-------------------------------------------+
    | x 与 y 长度不一致          | ValueError                                |
    +---------------------------+-------------------------------------------+
    | 非数值 / 含 NaN / Inf      | ValueError                                |
    +---------------------------+-------------------------------------------+
    | 重复 x 节点               | ValueError (样条要求严格递增)             |
    +---------------------------+-------------------------------------------+
    | x 乱序但唯一              | UserWarning + 按 x 升序重排               |
    +---------------------------+-------------------------------------------+
    | y 全为常量                | 正常返回常量插值                           |
    +---------------------------+-------------------------------------------+
    | UnivariateSpline 节点数<=k| ValueError (提示改用 k=1)                 |
    +---------------------------+-------------------------------------------+
    | 未知 backend              | ValueError (附可用列表)                    |
    +---------------------------+-------------------------------------------+

    Performance report (measured, tests/test_spline.py 一并覆盖)
    ---------------------------------------------------------------
    环境: Windows, Python 3.10.11, numpy 2.2.6, scipy 1.15.3;
    中位数 (7 次重复); x = linspace(0,1,n), y = sin(40x);
    求值点 1e6, 全部后端均精确穿过节点。

    +------------------+----------------+-----------------+--------------+
    | backend          | fit @ 2k nodes | fit @ 20k nodes | eval (1e6 pt)|
    +==================+================+=================+==============+
    | UnivariateSpline | ~0.2 ms        | ~2.2 ms         | ~44.9 ms     |
    +------------------+----------------+-----------------+--------------+
    | CubicSpline      | ~0.1 ms        | ~1.1 ms         | ~6.3 ms      |
    +------------------+----------------+-----------------+--------------+
    | PCHIP            | ~0.1 ms        | ~0.8 ms         | ~6.5 ms      |
    +------------------+----------------+-----------------+--------------+
    | Akima            | ~0.1 ms        | ~1.0 ms         | ~6.4 ms      |
    +------------------+----------------+-----------------+--------------+

    结论: 拟合耗时各后端基本同级 (0.1–2 ms 量级); **大量批量求值时
    CubicSpline/PCHIP/Akima 约比 UnivariateSpline 快 ~7×** (FITPACK
    splev 逐点开销更高)。默认取 UnivariateSpline 是设计取向 (接口/能力最全:
    平滑因子 s、权重 w、derivative/integral/roots), 追求纯插值性能时可显式选
    ``backend="CubicSpline"`` (语义与默认 k=3/s=0 插值一致)。极端输入行为
    汇总见上表; 完整用例与耗时明细见 ``tests/test_spline.py`` (42 项)。

    References
    ----------
    * FITPACK / UnivariateSpline: https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.UnivariateSpline.html
    * CubicSpline: https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.CubicSpline.html
    * PchipInterpolator: https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.PchipInterpolator.html
    * Akima1DInterpolator: https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.Akima1DInterpolator.html
    """
    return Spline(x, y, backend=backend, **kwargs)
