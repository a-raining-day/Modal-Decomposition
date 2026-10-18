"""
第三方子模块的 import 缓存 (进程级注册表)。

进程级、线程安全的注册表, 存放已导入的第三方子模块 (如 ``scipy.signal`` /
``scipy.interpolate``)。同一进程内每个模块至多 import 一次, 之后所有使用方
都经 :meth:`cache.get` 取用同一实例。

唯一性契约
----------
- 键是**区分大小写、非空**的 str (推荐模块式名字, 如 ``"scipy.signal"``);
  ``""``、纯空白和非 str 名字在入口处即被拒绝。
- :meth:`cache.add` **永不覆盖**: 注册已存在的名字返回 ``False`` 并保留原 api;
  替换只能先显式 :meth:`cache.remove` 再 :meth:`cache.add`。
- 存储是类属性, 由类和所有实例共享, 因此唯一性保证是进程级的, 与谁注册无关。
- 每个条目带一条描述串 (见 :meth:`cache.describe`); 注册时不给就用自动生成的说明。

线程安全
--------
所有读写都在一把可重入锁内, 因此并发注册不同名字安全, 并发注册同名恰好成功一次。
"""

import functools
import threading
import types
from typing import Any, Dict, List, Optional, Tuple

__all__ = ["Cache", "cache"]


class cache:
    """
    Import cache: 已导入模块的唯一名字注册表。

    公开 API 以 classmethod 实现, 两种写法等价且共享同一份进程级存储::

        cache.add("scipy.signal", scipy.signal)   # 类上直接调用
        cache().add("scipy.signal", scipy.signal) # 经实例调用

    共享状态 (类属性, 实例也读写同一份)
    ----------------------------------
    ``_entries``: ``{name: api}``; ``_descriptions``: ``{name: description}``,
    与 ``_entries`` 同步增删; ``_exists``: 记录 import 失败过的名字, 用作负缓存
    (进程内不再重试); ``_lock``: 保护全部读写的 ``RLock``。

    Examples
    --------
    >>> from Modal_Decomposition.Base.Cache import cache
    >>> import scipy.signal as ss
    >>> cache.add("scipy.signal", ss, description="signal processing submodule")
    True
    >>> cache.check("scipy.signal")
    True
    >>> cache.get("scipy.signal") is ss
    True
    >>> cache.describe("scipy.signal")
    'signal processing submodule'
    >>> cache.add("scipy.signal", ss)  # 重复注册: 永不覆盖
    False
    """

    # --- 进程级共享状态: 类属性, 实例也读写同一份 ---
    _entries: Dict[str, Any] = {}
    _exists: Dict[str, bool] = {}
    _descriptions: Dict[str, str] = {}
    _lock = threading.RLock()

    @staticmethod
    def _validate_name(name) -> str:
        """
        校验注册键。

        名字按原样保留 (精确字符串、区分大小写), 只拒绝极端情况后返回。

        Raises
        ------
        TypeError
            名字不是 str (None、int、bytes 等)。
        ValueError
            名字为空或纯空白。
        """
        if not isinstance(name, str):
            raise TypeError(
                f"cache name must be a str, got {type(name).__name__}"
            )
        if not name.strip():
            raise ValueError("cache name must not be empty or whitespace")
        return name

    @staticmethod
    def _validate_api(api) -> None:
        """拒绝 None api: 缓存 None 与"条目缺失"无法区分。"""
        if api is None:
            raise TypeError("cache api must not be None")

    @staticmethod
    def _validate_description(description: Optional[str]) -> Optional[str]:
        """描述必须是 str 或 None; 空串 / 纯空白等价于 None。"""
        if description is None:
            return None
        if not isinstance(description, str):
            raise TypeError(
                f"cache description must be a str or None, "
                f"got {type(description).__name__}"
            )
        stripped = description.strip()
        return stripped if stripped else None

    @staticmethod
    def _api_label(api) -> str:
        """缓存对象的人类可读标签 (模块名或类型名)。"""
        if isinstance(api, types.ModuleType):
            return getattr(api, "__name__", str(api))
        return f"{type(api).__module__}.{type(api).__qualname__}"

    @classmethod
    def get(cls, name):
        """
        取回已缓存的 api。

        Raises
        ------
        KeyError
            ``name`` 未注册; 错误信息里带上已注册的键。
        """
        name = cls._validate_name(name)
        with cls._lock:
            if name not in cls._entries:
                raise KeyError(
                    f"'{name}' is not registered in the cache; "
                    f"registered entries: {sorted(cls._entries)}"
                )
            return cls._entries[name]

    @classmethod
    def add(
            cls,
            name,
            api,
            verbose: bool = False,
            description: Optional[str] = None,
    ) -> bool:
        """
        把模块 / 函数 / 类的 api 注册进缓存。

        Parameters
        ----------
        name : str
            唯一注册键 (区分大小写, 非空)。
        api : object
            要缓存的已导入对象 (通常是模块); 不能是 None。
        verbose : bool
            为 True 时打印每条新注册与每次被拒的重复注册。
        description : str, optional
            该模块的一行描述, 之后用 :meth:`cache.describe` 取回; 省略时用自动
            生成的说明。

        Returns
        -------
        bool
            注册成功为 True; ``name`` 已注册时为 False (既有 api 原样保留)。
        """
        name = cls._validate_name(name)
        cls._validate_api(api)
        description = cls._validate_description(description)

        with cls._lock:
            if name in cls._entries:
                if verbose:
                    print(
                        f"[cache] '{name}' is already registered; "
                        f"duplicate registration ignored"
                    )
                return False

            cls._entries[name] = api
            cls._descriptions[name] = description if description is not None \
                else f"<no description provided at registration of '{name}'>"

            if verbose:
                print(f"[cache] registered '{name}' ({cls._api_label(api)})")
            return True

    @classmethod
    def import_module(
        cls,
        name: str,
        description: Optional[str] = None,
        verbose: bool = False,
    ):
        """
        抽象并集中管理经典惰性导入样板::

            if <flag> is None:
                try:
                    import <name> as <flag>
                except ImportError:
                    raise ImportError(...)

        - 首次调用: import 目标模块, 以模块名 ``name`` 注册进统一缓存 (持 ``RLock``, 恰好执行一次);
        - 之后调用: 直接返回缓存实例, 不再执行 import;
        - 导入失败: 名字记进 ``_exists`` 作负缓存, 返回 ``None`` (``verbose=True`` 时改抛
          ``ImportError``); 同一进程内不再重试, 不假设运行期会动态 pip install;
        - 键已存在时 (可能由另一条 import 路径先注册) 直接返回既有实例, 维持"永不覆盖"契约。

        Parameters
        ----------
        name : str
            要 import 的模块名, 同时作注册键 (sys.modules 风格, 区分大小写, 非空)。
        description : str, optional
            存入缓存、供 :meth:`cache.describe` 取回的一行描述。
        verbose : bool
            模块导入失败时抛 ``ImportError``; 为 False 时返回 ``None``。

        Returns
        -------
        module | None
            已 import 并注册的模块实例; 模块不存在且 ``verbose=False`` 时为 ``None``。
        """
        name = cls._validate_name(name)
        description = cls._validate_description(description)

        with cls._lock:
            if name in cls._entries:
                return cls._entries[name]

            if name not in cls._exists:
                from importlib import import_module as _import_module
                try:
                    api = _import_module(name)  # ImportError 原样传播, 不写入缓存
                except ImportError:
                    cls._exists[name] = False
                    api = None
            else:
                api = None

            if api is not None:
                cls._entries[name] = api
                cls._descriptions[name] = (
                    description if description is not None
                    else f"<imported module '{name}'>"
                )

            if verbose and api is None:
                raise ImportError(f"the {name} module is not pip!")

            return api

    @classmethod
    def lazy(cls, name: str, description: Optional[str] = None, verbose: bool = False):
        """
        装饰器: 把零参加载函数变成"进程内恰好执行一次"的惰性取用器。

        对调用方完全透明 (``functools.wraps`` 保留签名与 docstring), 缓存藏在背后: 调用者只看到一个普通函数调用。

        - 首次调用: 持 ``RLock`` 执行原函数, 结果以 ``name`` 注册进缓存并返回;
        - 之后调用: 返回缓存实例, 原函数不再执行;
        - 原函数抛异常: 原样传播且不缓存, 下次调用可重试;
        - 键已存在: 不执行原函数, 返回既有实例 (永不覆盖);
        - 带参调用或原函数返回 ``None``: 抛 ``TypeError``。

        Parameters
        ----------
        name : str
            Unique registry key (case-sensitive, non-empty).
        description : str, optional
            One-line description stored for :meth:`cache.describe`.
        verbose : bool
            Print a line when the entry is newly registered.
        """
        name = cls._validate_name(name)
        description = cls._validate_description(description)

        def dec(fn):
            @functools.wraps(fn)
            def wrapper(*args, **kwargs):
                if args or kwargs:
                    raise TypeError(
                        f"cache.lazy loader {fn.__qualname__!r} takes no "
                        f"arguments (got {len(args)} positional, "
                        f"{sorted(kwargs)} keyword)"
                    )
                with cls._lock:
                    if name in cls._entries:
                        return cls._entries[name]
                    api = fn()
                    cls._validate_api(api)
                    cls._entries[name] = api
                    cls._descriptions[name] = (
                        description if description is not None
                        else f"<lazy '{name}' via {fn.__qualname__}>"
                    )
                    if verbose:
                        print(
                            f"[cache] registered '{name}' "
                            f"via lazy loader {fn.__qualname__} "
                            f"({cls._api_label(api)})"
                        )
                    return api

            return wrapper

        return dec

    @classmethod
    def check(cls, name) -> bool:
        """
        查询 ``name`` 是否已注册。

        Returns
        -------
        bool
            已注册为 True, 否则 False。
        """
        name = cls._validate_name(name)
        with cls._lock:
            return name in cls._entries

    @classmethod
    def describe(cls, name) -> str:
        """
        返回已注册条目的描述。

        注册时显式给出的 ``description`` 原样返回; 未给出时返回注册时自动生成的
        说明 (如 ``<imported module 'x'>``)。

        Raises
        ------
        KeyError
            ``name`` 未注册。
        """
        name = cls._validate_name(name)
        with cls._lock:
            if name not in cls._descriptions:
                raise KeyError(
                    f"'{name}' is not registered in the cache; "
                    f"registered entries: {sorted(cls._entries)}"
                )
            return cls._descriptions[name]

    @classmethod
    def remove(cls, name) -> bool:
        """
        删除一个条目 (api 与描述) 并返回它原先是否存在。

        删除是唯一能为"替换注册"腾出位置的途径。
        """
        name = cls._validate_name(name)
        with cls._lock:
            existed = name in cls._entries
            cls._entries.pop(name, None)
            cls._descriptions.pop(name, None)
            cls._exists.pop(name, None)
            return existed

    @classmethod
    def clear(cls) -> None:
        """清空所有已注册条目。"""
        with cls._lock:
            cls._entries.clear()
            cls._descriptions.clear()
            cls._exists.clear()

    @classmethod
    def names(cls) -> List[str]:
        """已注册键的升序列表。"""
        with cls._lock:
            return sorted(cls._entries)

    @classmethod
    def keys(cls) -> List[str]:
        """``names`` 的别名。"""
        return cls.names()

    @classmethod
    def values(cls) -> List[Any]:
        """已缓存 api 的列表, 按键升序。"""
        with cls._lock:
            return [cls._entries[k] for k in sorted(cls._entries)]

    @classmethod
    def items(cls) -> List[Tuple[str, Any]]:
        """``(name, api)`` 对的列表, 按键升序。"""
        with cls._lock:
            return [(k, cls._entries[k]) for k in sorted(cls._entries)]

    @classmethod
    def descriptions(cls) -> Dict[str, str]:
        """``{name: description}`` 的快照 dict。"""
        with cls._lock:
            return dict(cls._descriptions)

    # ------------------------------------------------------------------ #
    # 便捷协议 (实例级: len(cache())、'x' in cache()、cache()['x'])
    # ------------------------------------------------------------------ #
    def __contains__(self, name) -> bool:
        return self.check(name)

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    def __getitem__(self, name):
        return self.get(name)

    def __repr__(self) -> str:
        with self._lock:
            return (
                f"<cache entries={sorted(self._entries)} "
                f"n={len(self._entries)}>"
            )


# 首字母大写别名: 与 Base 包其余类(Config / Decomposer)风格一致。
Cache = cache
