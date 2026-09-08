"""
store of the import cache

Process-wide, thread-safe registry for imported third-party submodules
(e.g. ``scipy.signal`` / ``scipy.interpolate``). A module is imported and
registered at most once per process; afterwards every consumer fetches it
through :meth:`cache.get`.

Uniqueness contract
-------------------
- Keys are **case-sensitive, non-empty strings** (module-style names such as
  ``"scipy.signal"`` are recommended). ``""``, whitespace-only names and
  non-string names are rejected up front.
- :meth:`cache.add` **never overwrites**: registering an already-known name
  returns ``False`` and keeps the original api. Replacement is only possible
  via an explicit :meth:`cache.remove` followed by a new :meth:`cache.add`.
- The backing store is shared by the class and by every instance, so the
  uniqueness guarantee holds process-wide no matter who registers.
- Every registered entry carries a description string (see
  :meth:`cache.describe`); the description is documented in a comment next to
  each ``cache.add(...)`` call site.

Thread safety
-------------
All mutating and reading operations are guarded by a single re-entrant lock,
so concurrent registration of distinct names is safe and concurrent
registration of the *same* name succeeds exactly once.
"""

import functools
import threading
import types
from typing import Any, Dict, List, Optional, Tuple

__all__ = ["cache", "Cache"]


class cache:
    """
    Import cache: unique-name registry of imported modules.

    The public API is implemented with classmethods, so both spellings are
    equivalent and share one process-wide store::

        cache.add("scipy.signal", scipy.signal)   # class-level call
        cache().add("scipy.signal", scipy.signal) # instance-level call

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
    >>> cache.add("scipy.signal", ss)  # duplicate: never overwrites
    False
    """

    # --- process-wide shared state (class attributes; instances share it) ---
    _entries: Dict[str, Any] = {}
    _descriptions: Dict[str, str] = {}
    _lock = threading.RLock()

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _validate_name(name) -> str:
        """
        Validate a registry key.

        Returns the name unchanged (keys are kept verbatim: exact string,
        case-sensitive) after rejecting the extreme cases.

        Raises
        ------
        TypeError
            On a non-string name (None, int, bytes, ...).
        ValueError
            On an empty or whitespace-only name.
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
        """Reject None apis: caching None is ambiguous with a missing entry."""
        if api is None:
            raise TypeError("cache api must not be None")

    @staticmethod
    def _validate_description(description: Optional[str]) -> Optional[str]:
        """Description must be a str (or None); empty/whitespace means None."""
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
        """Human-readable label of a cached api (module name or type name)."""
        if isinstance(api, types.ModuleType):
            return getattr(api, "__name__", str(api))
        return f"{type(api).__module__}.{type(api).__qualname__}"

    # ------------------------------------------------------------------ #
    # public API (classmethods: usable as ``cache.add(...)`` and
    # ``cache().add(...)`` alike)
    # ------------------------------------------------------------------ #
    @classmethod
    def add(
        cls,
        name,
        api,
        verbose: bool = False,
        description: Optional[str] = None,
    ) -> bool:
        """
        add the api of the module or function or class to the cache

        Parameters
        ----------
        name : str
            Unique registry key (case-sensitive, non-empty).
        api : object
            The imported module (or any object) to cache; must not be None.
        verbose : bool
            When True, print a line for every registration and for every
            rejected duplicate.
        description : str, optional
            One-line description of the registered module. Retrieved later
            with :meth:`cache.describe`. The same text is documented in a
            comment next to this ``add`` call. When omitted, ``describe``
            falls back to an auto-generated notice.

        Returns
        -------
        bool
            True when the entry was registered; False when ``name`` was
            already registered (the existing api is kept untouched).
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

        - 首次调用: import 目标模块, 以模块名 ``name`` 注册进统一缓存
          (持 ``RLock``, 恰好执行一次);
        - 之后调用: 直接返回缓存实例, 不再执行 import;
        - ``ImportError`` 原样传播且**不缓存** (标志保持"未就绪", 下次调用
          自动重试) —— try/except 只存在于这一处, 调用方零样板;
        - 键已存在时 (可能由另一条 import 路径先注册) 直接返回既有实例,
          维持"永不覆盖"契约。

        Parameters
        ----------
        name : str
            Module name to import, doubling as the registry key
            (sys.modules style, case-sensitive, non-empty).
        description : str, optional
            One-line description stored for :meth:`cache.describe`.
        verbose : bool
            Print a line when the entry is newly registered.

        Returns
        -------
        module
            The imported (and registered) module instance.
        """
        name = cls._validate_name(name)
        description = cls._validate_description(description)

        with cls._lock:
            if name in cls._entries:
                return cls._entries[name]

            from importlib import import_module as _import_module

            api = _import_module(name)  # ImportError propagates, nothing cached
            cls._validate_api(api)
            cls._entries[name] = api
            cls._descriptions[name] = (
                description if description is not None
                else f"<imported module '{name}'>"
            )

            if verbose:
                print(f"[cache] registered '{name}' ({cls._api_label(api)})")
            return api

    @classmethod
    def lazy(cls, name: str, description: Optional[str] = None, verbose: bool = False):
        """
        装饰器: 把零参加载函数变成"进程内恰好执行一次"的惰性取用器。

        对调用方完全透明 (``functools.wraps`` 保留签名与 docstring), 缓存
        藏在背后: 调用者只看到一个普通函数调用。

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
    def get(cls, name):
        """
        return the cached api

        Raises
        ------
        KeyError
            When ``name`` is not registered.
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
    def check(cls, name) -> bool:
        """
        return the result of the query to cache
        """
        name = cls._validate_name(name)
        with cls._lock:
            return name in cls._entries

    @classmethod
    def describe(cls, name) -> str:
        """
        Return the description of a registered module.

        Every entry registered with a ``description`` answers with exactly
        that text (mirrored in the comment next to the ``cache.add`` call);
        entries registered without one answer with an auto-generated notice.

        Raises
        ------
        KeyError
            When ``name`` is not registered.
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
        Remove one entry (api and description).

        Returns True when the entry existed. Removal is the only sanctioned
        way to make room for a *replacement* registration.
        """
        name = cls._validate_name(name)
        with cls._lock:
            existed = name in cls._entries
            cls._entries.pop(name, None)
            cls._descriptions.pop(name, None)
            return existed

    @classmethod
    def clear(cls) -> None:
        """Remove every registered entry."""
        with cls._lock:
            cls._entries.clear()
            cls._descriptions.clear()

    @classmethod
    def names(cls) -> List[str]:
        """Sorted list of all registered keys."""
        with cls._lock:
            return sorted(cls._entries)

    @classmethod
    def keys(cls) -> List[str]:
        """Alias of :meth:`cache.names`."""
        return cls.names()

    @classmethod
    def values(cls) -> List[Any]:
        """List of cached apis in sorted-key order."""
        with cls._lock:
            return [cls._entries[k] for k in sorted(cls._entries)]

    @classmethod
    def items(cls) -> List[Tuple[str, Any]]:
        """List of ``(name, api)`` pairs in sorted-key order."""
        with cls._lock:
            return [(k, cls._entries[k]) for k in sorted(cls._entries)]

    @classmethod
    def descriptions(cls) -> Dict[str, str]:
        """Snapshot dict of ``{name: description}``."""
        with cls._lock:
            return dict(cls._descriptions)

    # ------------------------------------------------------------------ #
    # convenience protocol (instance level: len(cache()), 'x' in cache(),
    # cache()['x'])
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
