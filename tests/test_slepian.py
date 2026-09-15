"""
Slepian (DPSS) 后端测试。

覆盖:
* 三后端 (numpy / scipy / C) 与 ``scipy.signal.windows.dpss`` 的数值一致性
  (序列逐元素含符号 + 集中比);
* 统一返回契约 (形状/dtype/正交性/集中比单调且 <= 1/默认条数);
* ``norm`` 与 ``sym`` 变体;
* 边界情形 (N=1,2,3; K=1; 极端 NW);
* 参数守卫 (非法 N / halfBW / nTapers / mod / norm; numpy 稠密预算);
* 后端解析与 import cache getter。
"""

import numpy as np
import pytest
from scipy.signal.windows import dpss

from Modal_Decomposition.Base.ConstDefine import (
    SLEPIAN_BACKEND,
    SLEPIAN_BACKEND_LIST,
    SLEPIAN_NUMPY_MAX_BYTES,
)
from Modal_Decomposition.Utils import get_slepian
from Modal_Decomposition.Utils._Slepian.C import available as c_available
from Modal_Decomposition.Utils._Slepian.numpy_slepian import generate_slepian_numpy
from Modal_Decomposition.Utils.Slepian import (
    _small_n_backend,
    available_backends,
    cache_info,
    clear_cache,
    resolve_backend,
    slepian,
)


@pytest.fixture(autouse=True)
def _reset_slepian_cache():
    """每个用例前后清空 Tier-1 缓存, 保证命中/未命中统计可确定。"""
    clear_cache()
    yield
    clear_cache()

#: 参与参数化测试的后端 (C 未编译时自动跳过)。
BACKENDS = [b for b in SLEPIAN_BACKEND_LIST if b != "C" or c_available()]

#: 小规模参数网格 (含奇/偶 N 与各种 NW)。K 上限取 N//2 以内:
#: 接近 N/2 时高阶特征值数值简并, 各实现的**排序/符号**可以不同 (集合仍一致,
#: 见 test_full_basis_matches_scipy_as_set)。
GRID = [
    (N, NW, K)
    for N in (16, 17, 64, 65, 129)
    for NW in (1.0, 2.0, 3.5, 6.0)
    for K in (1, 3, 5)
]

TOL_VEC = 1e-9
TOL_RATIO = 1e-9


def _scipy_ref(N, NW, K, **kw):
    """scipy 参考; scipy 自身在个别极小规模会抛异常, 跳过而非失败。"""
    try:
        return dpss(N, NW, K, **kw)
    except Exception as exc:                       # pragma: no cover - scipy 缺陷
        pytest.skip(f"scipy.signal.windows.dpss 不支持该组合: {exc}")


# --------------------------------------------------------------------------- #
# 正确性
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("N,NW,K", GRID)
def test_matches_scipy(backend, N, NW, K):
    """三后端与 scipy 一致: 序列逐元素 (含符号) 与集中比。"""
    mine, ratios = slepian(N, NW, K, mod=backend, return_ratios=True)
    ref, ref_ratios = _scipy_ref(N, NW, K, return_ratios=True)
    assert mine.shape == (K, N)
    assert np.max(np.abs(mine - ref)) < TOL_VEC
    assert np.max(np.abs(ratios - ref_ratios)) < TOL_RATIO


@pytest.mark.parametrize("backend", BACKENDS)
def test_orthonormal_and_dtype(backend):
    """返回 (K,N) float64 且各行两两正交、单位范数。"""
    tapers = slepian(256, 4.0, 5, mod=backend, norm=2)
    assert tapers.shape == (5, 256)
    assert tapers.dtype == np.float64
    gram = tapers @ tapers.T
    assert np.max(np.abs(gram - np.eye(5))) < 1e-12


@pytest.mark.parametrize("backend", BACKENDS)
def test_ratios_sorted_and_bounded(backend):
    """集中比降序且落在 (0, 1]; 且大 NW 时前 2NW-1 阶接近 1。"""
    tapers, ratios = slepian(1024, 4.0, 6, mod=backend, return_ratios=True)
    assert np.all(np.diff(ratios) <= 1e-12)
    assert np.all(ratios > 0.0) and np.all(ratios <= 1.0 + 1e-9)
    assert ratios[0] > 0.999999                    # 0 阶高度集中
    assert ratios[2] > 0.99                        # 2NW-1 = 7 阶以内仍高度集中


@pytest.mark.parametrize("backend", BACKENDS)
def test_default_taper_count(backend):
    """nTapers=None -> max(1, floor(2*NW-1)), 且不超过 solve_n。"""
    assert slepian(64, 4.0, None, mod=backend).shape == (7, 64)
    assert slepian(64, 1.2, None, mod=backend).shape == (1, 64)


@pytest.mark.parametrize("backend", BACKENDS)
def test_default_taper_count_clamped_to_length(backend):
    """NW 很大时条数截断到 N; scipy 因 NW >= N/2 不可用会自动换后端 (带 warning)。"""
    import warnings as _w

    with _w.catch_warnings():
        _w.simplefilter("ignore", UserWarning)
        assert slepian(3, 20.0, None, mod=backend).shape == (3, 3)


@pytest.mark.parametrize("backend", BACKENDS)
def test_full_basis_matches_scipy_as_set(backend):
    """
    K=N (请求全部阶) 时: 高阶/低阶存在数值简并, 逐元素可能只是排序+符号不同,
    但**序列集合**必须与 scipy 完全一致 (相关矩阵为置换矩阵), 集中比也必须一致。
    """
    N, NW = 16, 1.0
    mine, ratios = slepian(N, NW, N, mod=backend, return_ratios=True)
    ref, ref_ratios = dpss(N, NW, N, return_ratios=True)
    corr = np.abs(mine @ ref.T)
    best = np.argmax(corr, axis=1)
    assert len(set(best.tolist())) == N                      # 双射 (置换)
    assert corr[np.arange(N), best].min() > 0.999999         # 每个都精确对应
    assert np.max(np.abs(ratios - ref_ratios)) < TOL_RATIO


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("norm", [2, None, "approximate"])
def test_norm_variants(backend, norm):
    """norm=2/None (单位能量) 与 "approximate" (峰值 1 + 偶长度修正) 同 scipy。"""
    mine = slepian(65, 3.0, 4, mod=backend, norm=norm)
    ref = _scipy_ref(65, 3.0, 4, norm=norm)
    assert np.max(np.abs(mine - ref)) < TOL_VEC


@pytest.mark.parametrize("backend", BACKENDS)
def test_sym_false(backend):
    """sym=False (周期性) 与 scipy 一致, 且长度为 N。"""
    mine = slepian(64, 3.0, 4, mod=backend, sym=False)
    ref = _scipy_ref(64, 3.0, 4, sym=False)
    assert mine.shape == (4, 64)
    assert np.max(np.abs(mine - ref)) < TOL_VEC


# --------------------------------------------------------------------------- #
# 边界情形
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("N,NW,K", [(1, 1.0, 1), (2, 0.5, 1), (3, 1.0, 1), (5, 0.5, 2), (7, 3.0, 3)])
def test_edge_sizes(backend, N, NW, K):
    """极小规模 (含 scipy 会失败的 N=1/2) 必须能算, 且与 numpy 后端同谱同值。"""
    mine = slepian(N, NW, K, mod=backend)
    other = generate_slepian_numpy(N, NW, K)
    assert mine.shape == (K, N)
    if backend != "numpy":
        assert np.max(np.abs(mine - other)) < 1e-9


@pytest.mark.parametrize("backend", BACKENDS)
def test_tiny_and_huge_bandwidth(backend):
    """NW 极小 (近乎单点集中) 与很大 (整带集中) 都不崩。"""
    assert slepian(512, 1e-3, 2, mod=backend).shape == (2, 512)
    big = slepian(512, 200.0, 5, mod=backend)
    assert big.shape == (5, 512)
    assert np.max(np.abs(big @ big.T - np.eye(5))) < 1e-10


# --------------------------------------------------------------------------- #
# 守卫与解析
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("kwargs,exc", [
    (dict(N=0, halfBW=1.0), ValueError),
    (dict(N=-3, halfBW=1.0), ValueError),
    (dict(N=64, halfBW=0.0), ValueError),
    (dict(N=64, halfBW=-1.0), ValueError),
    (dict(N=64, halfBW=float("nan")), ValueError),
    (dict(N=64, halfBW=2.0, nTapers=0), ValueError),
    (dict(N=64, halfBW=2.0, norm="subsample"), ValueError),
    (dict(N=64, halfBW=2.0, sym=1), ValueError),
])
def test_invalid_args(backend, kwargs, exc):
    kwargs = dict(kwargs)
    N = kwargs.pop("N")
    halfBW = kwargs.pop("halfBW")
    with pytest.raises(exc):
        slepian(N, halfBW, mod=backend, **kwargs)


@pytest.mark.parametrize("bad", ["nope", "", "NumPy2", "cuda"])
def test_unknown_backend_rejected(bad):
    with pytest.raises(ValueError):
        slepian(64, 2.0, 3, mod=bad)


def test_numpy_dense_budget_guard():
    """numpy 后端超预算必须报错并提示换后端 (而不是默默分配 GB 级矩阵)。"""
    limit = 2 * int(np.sqrt(SLEPIAN_NUMPY_MAX_BYTES // 8))
    with pytest.raises(ValueError) as err:
        generate_slepian_numpy(limit * 2, 4.0, 3)
    assert "mod='C'" in str(err.value) or "mod='scipy'" in str(err.value)
    assert generate_slepian_numpy(limit, 4.0, 3).shape == (3, limit)


def test_available_backends_shape():
    avail = available_backends()
    assert set(avail) == {"numpy", "scipy", "C"}
    assert avail["numpy"] is True
    assert avail["scipy"] is True                  # 库的硬依赖
    assert isinstance(avail["C"], bool)


def test_default_backend_and_resolution():
    """默认后端取自 ConstDefine, 且默认路径不可用时降级 numpy (而非抛错)。"""
    assert SLEPIAN_BACKEND in SLEPIAN_BACKEND_LIST
    assert resolve_backend(256, None) in ("numpy", "C", "scipy")
    assert resolve_backend(256, "C") == ("C" if c_available() else "numpy")
    assert resolve_backend(1 << 20, "auto") in ("numpy", "C")
    assert slepian(256, 3.0, 3).shape == (3, 256)  # 默认路径永远可用


def test_c_backend_reports_missing(monkeypatch):
    """显式指定 "C" 而库不可用时, 报 RealizationError 且带编译提示。"""
    from Modal_Decomposition.Error import RealizationError
    from Modal_Decomposition.Utils import _Slepian as pkg
    from Modal_Decomposition.Utils._Slepian import C as cmod

    monkeypatch.setattr(cmod, "_LIB", None)
    monkeypatch.setattr(cmod, "library_path", lambda: None)
    with pytest.raises(RealizationError) as err:
        slepian(64, 3.0, 3, mod="C")
    assert "编译" in str(err.value) or "compile" in str(err.value).lower()
    assert pkg is not None


def test_getter_registers_module():
    """Utils.get_slepian() 惰性 import + 注册进程级缓存, 且第二次取同一实例。"""
    from Modal_Decomposition.Base.Cache import cache

    mod1 = get_slepian()
    mod2 = get_slepian()
    assert mod1 is mod2
    key = "Modal_Decomposition.Utils.Slepian"
    assert cache.check(key)
    assert cache.get(key) is mod1
    assert hasattr(mod1, "slepian")


# --------------------------------------------------------------------------- #
# scipy 的规模限制: NW >= N/2 (默认参数下 N <= 6) 自动换后端
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("N", [1, 2, 4, 6])
def test_small_n_switches_away_from_scipy(N):
    """N <= 6 (NW=3 时 scipy 必然报错) -> warning + 换后端, 不抛错。"""
    with pytest.warns(UserWarning, match="NW >= N/2"):
        tapers = slepian(N, 3.0, 5)                 # 默认后端即 scipy
    assert tapers.shape[1] == N
    assert resolve_backend(N, None, halfBW=3.0) != "scipy"


def test_small_n_explicit_scipy_also_switches():
    """即使显式 mod="scipy", 该规模下也换后端 (scipy 必错, 换比报错有用)。"""
    with pytest.warns(UserWarning, match="NW >= N/2"):
        tapers = slepian(5, 3.0, 5, mod="scipy")
    assert tapers.shape == (5, 5)


def test_scipy_used_when_length_allows():
    """N=8 (2*NW < N) 时 scipy 可用, 不应有任何 warning。"""
    import warnings as _w

    with _w.catch_warnings():
        _w.simplefilter("error")
        tapers = slepian(8, 3.0, 5, mod="scipy")
    assert tapers.shape == (5, 8)


def test_small_n_backend_prefers_c():
    """小 N 回退顺序首位是 C (实测快 7-10x、内存小 2.4x), C 缺失时退 numpy。"""
    from Modal_Decomposition.Base.ConstDefine import SLEPIAN_SMALL_N_ORDER

    assert SLEPIAN_SMALL_N_ORDER[0] == "C"
    assert _small_n_backend() == ("C" if c_available() else "numpy")


def test_small_n_result_matches_other_backends():
    """换后端不改变数值: 小 N 结果与 numpy 后端逐元素一致。"""
    for N in (2, 4, 6):
        with pytest.warns(UserWarning):
            got = slepian(N, 3.0, 5)
        assert np.max(np.abs(got - generate_slepian_numpy(N, 3.0, 5))) < 1e-12


# --------------------------------------------------------------------------- #
# Tier-1 进程内缓存
# --------------------------------------------------------------------------- #
def test_cache_hit_miss_and_equality():
    first = slepian(512, 3.0, 5, mod="C")
    second = slepian(512, 3.0, 5, mod="C")
    info = cache_info()
    assert np.array_equal(first, second)
    assert (info["size"], info["hits"], info["misses"]) == (1, 1, 1)


def test_cache_key_is_bit_exact():
    """NW 的位级差异必须视为不同键 (3.0 vs 3.0000000000000004 结果不同)。"""
    slepian(64, 3.0, 5, mod="C")
    slepian(64, 3.0000000000000004, 5, mod="C")
    assert cache_info()["size"] == 2


@pytest.mark.parametrize("kwargs", [
    dict(N=64, halfBW=3.0, nTapers=5, mod="numpy"),          # 后端
    dict(N=64, halfBW=3.0, nTapers=5, mod="C", return_ratios=True),
    dict(N=64, halfBW=3.0, nTapers=5, mod="C", sym=False),
    dict(N=64, halfBW=3.0, nTapers=5, mod="C", norm="approximate"),
    dict(N=64, halfBW=3.0, nTapers=3, mod="C"),              # 条数
    dict(N=64, halfBW=3.5, nTapers=5, mod="C"),              # 半带宽
    dict(N=65, halfBW=3.0, nTapers=5, mod="C"),              # 长度
])
def test_cache_key_includes_all_arguments(kwargs):
    """后端 / 集中比 / sym / norm / 条数 / 长度 / 半带宽 任一不同即为不同键。"""
    slepian(64, 3.0, 5, mod="C")
    slepian(**kwargs)
    assert cache_info()["size"] == 2


def test_cache_disabled_by_flag():
    slepian(64, 3.0, 5, mod="C", cache=False)
    assert cache_info()["size"] == 0


def test_cache_maxsize_eviction_is_lru():
    for N in (32, 33, 34):
        slepian(N, 3.0, 5, mod="C", cache=2)
    info = cache_info()
    assert (info["size"], info["evictions"]) == (2, 1)
    # 最近使用的 N=34 仍命中, 最早的 N=32 已被淘汰
    before = cache_info()["misses"]
    slepian(34, 3.0, 5, mod="C", cache=2)
    assert cache_info()["misses"] == before          # 命中, 未新增 miss
    slepian(32, 3.0, 5, mod="C", cache=2)
    assert cache_info()["misses"] == before + 1      # 已被淘汰 -> 重新计算


def test_cache_returns_independent_writable_copy():
    """命中返回副本: 可写且改写不会污染缓存主副本。"""
    a = slepian(256, 3.0, 5, mod="C")
    b = slepian(256, 3.0, 5, mod="C")                # 命中
    assert b.flags.writeable
    b[0, 0] = 12345.0
    c = slepian(256, 3.0, 5, mod="C")                # 再命中
    assert not np.any(c == 12345.0)
    assert np.array_equal(a, c)


def test_cache_master_is_readonly():
    """缓存内部主副本只读 (防止被外部就地改写)。"""
    from Modal_Decomposition.Utils.Slepian import _CACHE

    slepian(128, 3.0, 5, mod="C")
    key = list(_CACHE._data)[0]
    master = _CACHE._data[key][0]
    assert master.flags.writeable is False
    with pytest.raises(ValueError):
        master[0, 0] = 1.0


def test_cache_byte_budget_skips_oversized_entry(monkeypatch):
    """单条超过字节上限 -> 只算不存 (skipped 计数)。"""
    import Modal_Decomposition.Utils.Slepian as mod

    monkeypatch.setattr(mod, "SLEPIAN_CACHE_MAX_BYTES", 1024)
    slepian(4096, 3.0, 5, mod="C")
    info = cache_info()
    assert (info["size"], info["skipped"]) == (0, 1)


def test_cache_consistency_without_cache():
    """开缓存与关缓存结果逐位一致 (三个后端)。"""
    for backend in BACKENDS:
        for (N, NW, K) in ((64, 3.0, 5), (257, 2.5, 4)):
            plain = slepian(N, NW, K, mod=backend, return_ratios=True, cache=False)
            clear_cache()
            _ = slepian(N, NW, K, mod=backend, return_ratios=True)        # miss
            cached = slepian(N, NW, K, mod=backend, return_ratios=True)   # hit
            assert np.array_equal(plain[0], cached[0])
            assert np.array_equal(plain[1], cached[1])


@pytest.mark.parametrize("bad", [0, -1, "yes", 1.5])
def test_cache_invalid_argument(bad):
    with pytest.raises(ValueError):
        slepian(64, 3.0, 5, cache=bad)


def test_cache_clear_resets_stats():
    slepian(64, 3.0, 5, mod="C")
    slepian(64, 3.0, 5, mod="C")
    clear_cache()
    info = cache_info()
    assert info["size"] == 0 and info["hits"] == 0 and info["misses"] == 0
    assert "version" in info
