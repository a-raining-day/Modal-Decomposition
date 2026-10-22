"""``Utils.Check`` 深度测试（契约矩阵 + 特征化 + 参考实现比对）。

与 ``tests/utils/test_Check.py``（烟测）的分工：

* ``test_Check.py``   —— 公开面是否在、几个主路径是否通；
* **本文件** —— 逐 dtype / 逐 safe / 逐分支的**矩阵**，把每条实现分支都推一遍，
  并把"看起来反直觉但确实是现行行为"的点固化成 characterization 测试，
  以免后续重构把已知行为当 bug 改掉。

导出（写 ``table/utils/check/*.csv`` 与 ``fig/utils/check/*.png``）::

    .\\.venv\\Scripts\\python.exe tests/utils/test_Check_deep.py --export

只用 ≤ 1 MiB 的数组；大规模行为见 ``tests/size_axis/``。
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import os
import shutil
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pytest

_TESTS = Path(__file__).resolve().parents[1]
if str(_TESTS) not in sys.path:
    sys.path.insert(0, str(_TESTS))

import _bench as B  # noqa: E402

B.bootstrap_src()    # 允许 `python tests/utils/test_Check_deep.py` 直接运行

from Modal_Decomposition.Define import (  # noqa: E402
    DEFAULT_NUMPY_TYPE,
    SAFE_MODS,
    SIZE,
)
from Modal_Decomposition.Utils import (  # noqa: E402
    PEAKS_MODS,
    Check,
    Check_Time_and_Signal,
)
from Modal_Decomposition import Utils as U  # noqa: E402


# --------------------------------------------------------------------------- #
# 1. dtype 矩阵
# --------------------------------------------------------------------------- #
DTYPE_CASES = [
    (np.bool_, "b", True, False, "bool"),
    (np.int8, "i", True, True, "int"),
    (np.int32, "i", True, True, "int"),
    (np.int64, "i", True, True, "int"),
    (np.uint8, "u", True, True, "uint"),
    (np.uint64, "u", True, True, "uint"),
    (np.float16, "f", True, True, "float"),
    (np.float32, "f", True, True, "float"),
    (np.float64, "f", True, True, "float"),
    (np.complex64, "c", False, False, "complex"),
    (np.complex128, "c", False, False, "complex"),
    ("datetime64[ns]", "M", False, False, "datatime"),
    ("timedelta64[s]", "m", False, False, "timedelta"),
    (object, "O", False, False, "object"),
    ("<U4", "U", False, False, "unicode"),
    ("S4", "S", False, False, "string"),
    ("V4", "V", False, False, "void"),
]


def build_dtype_matrix() -> list[dict]:
    rows = []
    for dtype, kind, numeric, real, kind_name in DTYPE_CASES:
        if dtype is object:
            arr = np.array([None, 1], dtype=object)
        elif dtype == "datetime64[ns]":
            arr = np.arange(3).astype("datetime64[ns]")
        elif dtype == "timedelta64[s]":
            arr = np.arange(3).astype("timedelta64[s]")
        else:
            arr = np.zeros(3, dtype=dtype)
        try:
            got_kind = Check.return_dtype_kind(arr)
        except TypeError:
            got_kind = "TypeError"      # 非 biuf：实现先拒绝，match 分支不可达
        rows.append({
            "dtype": np.dtype(arr.dtype).name,
            "kind_char": arr.dtype.kind,
            "expected_kind_char": kind,
            "is_numeric": bool(Check.is_numeric(arr)),
            "expected_is_numeric": numeric,
            "is_real": bool(Check.is_real(arr)),
            "expected_is_real": real,
            "return_dtype_kind": got_kind,
            "expected_return_dtype_kind": (kind_name if kind in "biuf" else "TypeError"),
            "verdict": "PASS" if (
                arr.dtype.kind == kind
                and bool(Check.is_numeric(arr)) == numeric
                and bool(Check.is_real(arr)) == real
                and got_kind == (kind_name if kind in "biuf" else "TypeError")
            ) else "FAIL",
        })
    return rows


@pytest.mark.parametrize("dtype,kind,numeric,real,kind_name", DTYPE_CASES)
def test_return_dtype_kind(dtype, kind, numeric, real, kind_name):
    """只有 ``biuf`` 四类能取到名字；其余 8 个 ``match`` 分支**实际不可达**。

    ``return_dtype_kind`` 第一步就 ``if not cls.is_numeric(S): raise TypeError``，
    而 ``is_numeric`` 只认 ``biuf``。所以 complex / datetime64 / timedelta64 /
    object / str / unicode / void 一律 ``TypeError``，那 8 个分支属于死代码，
    且错误信息写成 "must be np.ndarray, not <class 'numpy.ndarray'>"（自相矛盾）。
    """
    if dtype is object:
        arr = np.array([None, 1], dtype=object)
    elif dtype == "datetime64[ns]":
        arr = np.arange(3).astype("datetime64[ns]")
    elif dtype == "timedelta64[s]":
        arr = np.arange(3).astype("timedelta64[s]")
    else:
        arr = np.zeros(3, dtype=dtype)
    assert arr.dtype.kind == kind
    assert bool(Check.is_numeric(arr)) is numeric
    assert bool(Check.is_real(arr)) is real
    if kind in "biuf":
        assert Check.return_dtype_kind(arr) == kind_name
    else:
        with pytest.raises(TypeError):
            Check.return_dtype_kind(arr)


def test_dtype_matrix_all_pass():
    rows = build_dtype_matrix()
    assert all(r["verdict"] == "PASS" for r in rows), [r for r in rows if r["verdict"] != "PASS"]


def test_return_dtype_kind_rejects_non_array():
    """特征化：docstring 说非数组抛 ``TypeError``，实际先炸 ``.dtype`` → ``AttributeError``。"""
    with pytest.raises(AttributeError):
        Check.return_dtype_kind([1.0, 2.0])


# --------------------------------------------------------------------------- #
# 2. 谓词矩阵
# --------------------------------------------------------------------------- #
PRED_CASES = [
    # name, array, is_0d, is_1d, is_2d, is_empty
    ("scalar", np.array(42.0), True, False, False, False),
    ("empty", np.array([], dtype=float), False, True, False, True),
    ("one", np.array([1.0]), False, True, False, False),
    ("1d", np.arange(5.0), False, True, False, False),
    ("2d", np.zeros((3, 4)), False, False, True, False),
    ("2d_empty", np.zeros((0, 4)), False, False, True, True),
    ("3d", np.zeros((2, 3, 4)), False, False, False, False),
]


def build_pred_matrix() -> list[dict]:
    rows = []
    for name, arr, e0, e1, e2, ee in PRED_CASES:
        got = (bool(Check.is_0d(arr)), bool(Check.is_1d(arr)),
               bool(Check.is_2d(arr)), bool(Check.is_empty(arr)))
        exp = (e0, e1, e2, ee)
        rows.append({
            "case": name, "shape": str(arr.shape),
            "is_0d": got[0], "is_1d": got[1], "is_2d": got[2], "is_empty": got[3],
            "expected": str(exp), "got": str(got),
            "verdict": "PASS" if got == exp else "FAIL",
        })
    return rows


def test_pred_matrix_all_pass():
    assert all(r["verdict"] == "PASS" for r in build_pred_matrix()), build_pred_matrix()


@contextlib.contextmanager
def temp_mm(shape=(8,), dtype=np.float64, fill=None):
    """临时 memmap 文件；退出时显式关句柄再删目录（Windows 上否则删不掉）。"""
    d = tempfile.mkdtemp(prefix="md_check_")
    path = os.path.join(d, "m.dat")
    mm = np.memmap(path, dtype=dtype, mode="w+", shape=shape)
    if fill is not None:
        mm[:] = fill
    mm.flush()
    try:
        yield mm, path, d
    finally:
        for attr in ("_mmap",):
            h = getattr(mm, attr, None)
            if h is not None:
                try:
                    h.close()
                except Exception:
                    pass
        del mm
        shutil.rmtree(d, ignore_errors=True)


# --------------------------------------------------------------------------- #
# 3. is_duplicates
# --------------------------------------------------------------------------- #
def test_is_duplicates_basic():
    assert Check.is_duplicates(np.array([1.0, 2.0, 3.0])) is False
    assert Check.is_duplicates(np.array([1.0, 1.0, 3.0])) is True
    assert Check.is_duplicates(np.array([1.0, 2.0, 1.0])) is False, "只检测相邻重复"
    assert Check.is_duplicates(np.array([5.0])) is False
    assert Check.is_duplicates(np.array(7.0)) is True, "0-d 视为重复（退化输入）"


def test_is_duplicates_nan_not_detected_as_duplicate():
    """特征化：``np.diff`` 得 NaN，``NaN == 0`` 为 False → 相邻 NaN **不**判重。"""
    x = np.array([1.0, np.nan, np.nan])
    assert Check.is_duplicates(x) is False


def test_is_duplicates_only_diffs_last_axis():
    """特征化：``axis`` 只控制**切块方向**，重复判定恒为 ``np.diff(chunk)``（最后一轴）。

    推论（都是实测）：
    * 行内相邻重复 → ``axis=0`` / ``axis=1`` 都判 True（方向不影响 diff）；
    * **相邻两行完全相同 → 判不出来**（diff 从不在行之间比较）；
    * 同一列出现相同值 → 也判不出来。
    """
    rows_equal = np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
    cols_equal = np.array([[1.0, 2.0], [1.0, 3.0]])
    inside_row = np.array([[1.0, 1.0, 3.0], [4.0, 5.0, 6.0]])
    assert Check.is_duplicates(rows_equal, axis=0) is False
    assert Check.is_duplicates(rows_equal, axis=1) is False
    assert Check.is_duplicates(cols_equal, axis=0) is False
    assert Check.is_duplicates(cols_equal, axis=1) is False
    assert Check.is_duplicates(inside_row, axis=0) is True
    assert Check.is_duplicates(inside_row, axis=1) is True


def test_is_duplicates_2d_multichunk_is_ambiguous(monkeypatch):
    """特征化（潜在缺陷）：2-D 输入一旦真的跨块，边界比较是**数组比较**，
    在 ``if`` 里触发 ``ValueError: truth value ... is ambiguous``。

    现行 ``chunk_size`` 恒 ≥ 元素总数，所以单块时走不到；一旦切分修好就会暴露。
    """
    C = importlib.import_module("Modal_Decomposition.Utils.Check")
    monkeypatch.setattr(C.Chunk, "chunk_size", staticmethod(lambda *a, **k: 2))
    with pytest.raises(ValueError, match="ambiguous"):
        Check.is_duplicates(np.arange(12.0).reshape(4, 3), axis=0)


def test_is_duplicates_memmap():
    with temp_mm(fill=np.arange(8.0)) as (mm, _path, d):
        mm[4] = 3.0
        mm.flush()
        assert Check.is_duplicates(mm, temp_dir=d) is True


def test_is_duplicates_cross_chunk(monkeypatch):
    """块边界两侧的重复必须能被发现（分块算法的核心不变量）。"""
    C = importlib.import_module("Modal_Decomposition.Utils.Check")

    monkeypatch.setattr(C.Chunk, "chunk_size", staticmethod(lambda *a, **k: 3))
    x = np.array([1.0, 2.0, 3.0, 3.0, 5.0, 6.0, 7.0])   # 3|3 落在第 1/2 块边界
    assert Check.is_duplicates(x) is True
    y = np.arange(8.0)
    assert Check.is_duplicates(y) is False


# --------------------------------------------------------------------------- #
# 4. is_uniform
# --------------------------------------------------------------------------- #
def test_is_uniform_basic():
    assert Check.is_uniform(np.arange(10.0)) is True
    assert Check.is_uniform(np.array([0.0, 1.0, 3.0])) is False
    assert Check.is_uniform(np.array([2.0])) is True
    assert Check.is_uniform(np.array(5.0)) is True, "0-d 直接为 True"
    with pytest.raises(ValueError, match="at least 1"):
        Check.is_uniform(np.array([])), "空 1-D 会在分块入口抛 ValueError（不是 True）"


def test_is_uniform_tolerance():
    x = np.arange(1000.0)
    assert Check.is_uniform(x) is True
    y = x.copy()
    y[-1] += 1e-6                                  # 绝对偏移 1e-6 > atol 1e-14 且 rtol 1e-10*1 ≈ 1e-10
    assert Check.is_uniform(y) is False
    assert Check.is_uniform(y, rtol=1e-3, atol=1e-6) is True


def test_is_uniform_2d_squeeze_paths():
    assert Check.is_uniform(np.arange(6.0).reshape(6, 1)) is True
    assert Check.is_uniform(np.arange(6.0).reshape(1, 6)) is True
    with pytest.raises(ValueError):
        Check.is_uniform(np.zeros((2, 3)))


def test_is_uniform_cross_chunk(monkeypatch):
    C = importlib.import_module("Modal_Decomposition.Utils.Check")

    monkeypatch.setattr(C.Chunk, "chunk_size", staticmethod(lambda *a, **k: 4))
    assert Check.is_uniform(np.arange(13.0)) is True
    bad = np.arange(13.0)
    bad[4] = 10.0                                   # 第 1/2 块边界的首元素
    assert Check.is_uniform(bad) is False


# --------------------------------------------------------------------------- #
# 5. is_infinite（注意：True = finite，名字与返回值相反）
# --------------------------------------------------------------------------- #
def test_is_infinite_name_inversion():
    """特征化：`is_infinite` 返回 **True 表示有限**（docstring 明写，但名字反直觉）。"""
    assert bool(Check.is_infinite(np.arange(4.0))) is True
    assert bool(Check.is_infinite(np.array([0.0, np.inf]))) is False
    assert bool(Check.is_infinite(np.array([0.0, np.nan]))) is False
    assert bool(Check.is_infinite(np.arange(4))) is True, "整型走 np.isfinite 同一分支"
    assert isinstance(Check.is_infinite(np.arange(4.0)), np.bool_), "返回 numpy.bool_ 而非 bool"


def test_is_infinite_chunked_branch(monkeypatch):
    """强制走 memmap 分支，验证分块检测与显式 chunk_size。"""
    C = importlib.import_module("Modal_Decomposition.Utils.Check")

    monkeypatch.setattr(C, "should_use_memmap", lambda *a, **k: True)
    assert bool(Check.is_infinite(np.arange(100.0), chunk_size=7)) is True
    x = np.arange(100.0)
    x[63] = np.inf                                  # 落在第 10 块
    assert bool(Check.is_infinite(x, chunk_size=7)) is False


# --------------------------------------------------------------------------- #
# 6. require_* 家族
# --------------------------------------------------------------------------- #
def test_require_numeric():
    Check.require_numeric(np.zeros(3))
    Check.require_numeric(np.zeros(3, dtype=bool))
    with pytest.raises(ValueError, match="expected np.ndarray"):
        Check.require_numeric([1, 2, 3])
    with pytest.raises(ValueError, match="non-numeric dtype"):
        Check.require_numeric(np.zeros(3, dtype=complex))
    with pytest.raises(ValueError, match="non-numeric dtype"):
        Check.require_numeric(np.zeros(3, dtype=bool), allow_bool=False)


def test_require_method_prefix():
    with pytest.raises(ValueError, match=r"^EMD: "):
        Check.require_nonempty(np.array([]), method="EMD")


def test_require_ndim_nonempty_1d():
    Check.require_ndim(np.zeros((2, 3)), {1, 2})
    with pytest.raises(ValueError):
        Check.require_ndim(np.zeros(3), {2})
    with pytest.raises(ValueError):
        Check.require_nonempty(np.array([]))
    Check.require_1d(np.zeros(3))
    with pytest.raises(ValueError):
        Check.require_1d(np.zeros((2, 3)))


def test_require_length_match():
    # 签名是 (T, S)：比较 T.shape[0] 与 S.shape[-1]
    Check.require_length_match(np.zeros(8), np.zeros((2, 8)))
    with pytest.raises(ValueError):
        Check.require_length_match(np.zeros(9), np.zeros((2, 8)))


def test_require_length_match_message_says_S_twice():
    """特征化：错误信息把两个入参都写成 ``S``（``T`` 应为前者），只影响可读性。"""
    with pytest.raises(ValueError) as ei:
        Check.require_length_match(np.zeros(3), np.zeros(4), method="M")
    assert "S (3)" in str(ei.value) and "S (4)" in str(ei.value)


# --------------------------------------------------------------------------- #
# 7. not_numpy_arr / not_numpy_memmap
# --------------------------------------------------------------------------- #
def test_not_numpy_arr_semantics():
    assert Check.not_numpy_arr(np.zeros(2)) is True
    assert Check.not_numpy_arr([0.0], raise_error=False) is False
    assert Check.not_numpy_arr("x", method="M", raise_error=False) is False
    with pytest.raises(TypeError, match=r"^M: "):
        Check.not_numpy_arr([0.0], method="M")


def test_not_numpy_memmap_semantics():
    assert Check.not_numpy_memmap(np.zeros(2)) is True
    with temp_mm(shape=(2,)) as (mm, _path, _d):
        assert Check.not_numpy_memmap(mm, raise_error=False) is False
        with pytest.raises(TypeError, match="must not be np.memmap"):
            Check.not_numpy_memmap(mm)


# --------------------------------------------------------------------------- #
# 8. validate_mod / validate_safe
# --------------------------------------------------------------------------- #
def test_validate_mod_roundtrip():
    for m in PEAKS_MODS:
        assert Check.validate_mod(m, PEAKS_MODS) is m
        assert Check.validate_mod(m.value, PEAKS_MODS) is m
    with pytest.raises(ValueError, match="mod"):
        Check.validate_mod("nope", PEAKS_MODS)


def test_validate_mod_message_hardcodes_mod():
    """特征化：错误信息里固定写 ``mod``，即使调用方校验的是别的参数名。"""
    with pytest.raises(ValueError, match="the parameter of `mod`"):
        Check.validate_mod("nope", SAFE_MODS)


def test_validate_safe_roundtrip():
    for s in SAFE_MODS:
        assert Check.validate_safe(s) is s
        assert Check.validate_safe(s.value) is s
    with pytest.raises(ValueError, match=r"\['safe', 'suspicious', 'unknown'\]"):
        Check.validate_safe("nope")


# --------------------------------------------------------------------------- #
# 9. to_array
# --------------------------------------------------------------------------- #
def test_to_array_identity_when_nothing_to_do():
    x = np.arange(4.0)
    out = Check.to_array(x, ndim=1)
    assert out is x, "无 dtype 转换 / 无 squeeze 时应返回同一对象（零拷贝）"


def test_to_array_accepts_sequences():
    for seq in ([1, 2, 3], (1, 2, 3), np.array([1, 2, 3])):
        out = Check.to_array(seq, ndim=1)
        assert isinstance(out, np.ndarray)
        assert out.shape == (3,)


def test_to_array_list_becomes_default_float():
    out = Check.to_array([1, 2, 3])
    assert out.dtype == DEFAULT_NUMPY_TYPE


def test_to_array_int_stays_int():
    out = Check.to_array(np.arange(4, dtype=np.int16))
    assert out.dtype == np.int16


def test_to_array_dtype_conversion():
    out = Check.to_array(np.arange(4), dtype=np.float32)
    assert out.dtype == np.float32
    out2 = Check.to_array(np.arange(4, dtype=np.int64), dtype="float64")
    assert out2.dtype == np.float64


def test_to_array_rejects_non_numeric():
    for bad in (np.array(["a", "b"]), np.zeros(2, dtype=complex), np.array([None], dtype=object)):
        with pytest.raises(ValueError, match="non-numeric dtype"):
            Check.to_array(bad)


def test_to_array_0d_scalar_reshaped():
    out = Check.to_array(np.array(42.0))
    assert out.shape == (1,)
    # 特征化：0-d 且 size==0 的分支在 NumPy 里不可达（0-d 数组 size 恒为 1），
    # 因此 ``raise ValueError("... 0-d array")`` 是死代码，仅作防御。 


def test_to_array_empty_1d_passes_through():
    """特征化：`to_array` **不**拒绝空数组（拒绝在 `Check_Time_and_Signal` 里）。"""
    out = Check.to_array(np.array([], dtype=float), ndim=1)
    assert out.size == 0


def test_to_array_nan_inf_rejected_only_for_float():
    with pytest.raises(ValueError, match="NaN or Inf"):
        Check.to_array(np.array([0.0, np.nan]))
    with pytest.raises(ValueError, match="NaN or Inf"):
        Check.to_array(np.array([0.0, np.inf]))
    # 整型没有 NaN，分支自然跳过；bool 同理
    assert Check.to_array(np.array([0, 1])).dtype.kind == "i"


def test_to_array_safe_levels():
    """三档 safe 的**实际**差异（设计意图 vs 实现）。"""
    x2 = np.zeros((2, 3))
    # unknown: 全量校验 → 维度不符即抛
    with pytest.raises(ValueError):
        Check.to_array(x2, ndim=1)
    # suspicious: 只做 array 化 / 数值性 / dtype，不做 ndim / 有限性 / squeeze
    out = Check.to_array(x2, safe="suspicious", ndim=1)
    assert out.shape == (2, 3)
    # safe: 原样返回，连 array 化都不做
    out_safe = Check.to_array([[1, 2]], safe="safe")
    assert out_safe == [[1, 2]], "safe 档不做任何规范化（把 list 原样返回）"


def test_to_array_safe_suspicious_skips_finite_check():
    with pytest.raises(ValueError):
        Check.to_array(np.array([np.nan]), ndim=1)
    out = Check.to_array(np.array([np.nan]), safe="suspicious", ndim=1)
    assert np.isnan(out[0])


def test_to_array_ndim_forms():
    x = np.zeros((2, 3))
    assert Check.to_array(x, ndim=(1, 2)).shape == (2, 3)
    assert Check.to_array(x, ndim={1, 2}).shape == (2, 3)
    assert Check.to_array(x, ndim=[1, 2]).shape == (2, 3)
    assert Check.to_array(x, ndim=None).shape == (2, 3)
    with pytest.raises(ValueError):
        Check.to_array(x, ndim=0)
    with pytest.raises(ValueError):
        Check.to_array(x, ndim=set())
    with pytest.raises(ValueError):
        Check.to_array(x, ndim={0})
    with pytest.raises(TypeError):
        Check.to_array(x, ndim=True)
    with pytest.raises(TypeError):
        Check.to_array(x, ndim="1")


def test_to_array_need_squeeze():
    x1 = np.zeros((1, 5))
    assert Check.to_array(x1, ndim={1, 2}, need_squeeze=True).shape == (5,)
    assert Check.to_array(x1, ndim={1, 2}, need_squeeze=False).shape == (1, 5)
    with pytest.raises(ValueError, match="need_squeeze must be bool"):
        Check.to_array(x1, ndim={1, 2}, need_squeeze=1)


def test_to_array_squeeze_to_0d_rejected():
    with pytest.raises(ValueError, match="at least 1 dimension after squeeze"):
        Check.to_array(np.zeros((1, 1)), ndim=None)


# --------------------------------------------------------------------------- #
# 10. to_dtype
# --------------------------------------------------------------------------- #
def test_to_dtype_same_dtype_returns_same_object():
    x = np.arange(4.0)
    assert Check.to_dtype(x, np.float64) is x


def test_to_dtype_ndarray():
    x = np.arange(4, dtype=np.int32)
    out = Check.to_dtype(x, np.float32)
    assert out.dtype == np.float32 and np.array_equal(out, x)


def test_to_dtype_bad_dtype():
    with pytest.raises(TypeError, match="dtype must be"):
        Check.to_dtype(np.zeros(3), "not-a-dtype")


def test_to_dtype_memmap_does_not_materialise():
    with temp_mm(shape=(6,), dtype=np.float32, fill=np.arange(6)) as (mm, _path, _d):
        out = Check.to_dtype(mm, np.float64)
        assert isinstance(out, np.memmap), "memmap 必须经 temp_memmap+fill 转换，不能 .astype"
        assert out.dtype == np.float64
        assert np.array_equal(np.asarray(out), np.arange(6.0))
        assert out.filename != mm.filename


# --------------------------------------------------------------------------- #
# 11. Check_Time_and_Signal
# --------------------------------------------------------------------------- #
def test_ctas_default_arange_dtype_follows_signal():
    S, T, N = Check_Time_and_Signal(np.arange(6, dtype=np.int32), ndim={1}, method="M")
    assert T.dtype == np.int32 and T.shape == (6,) and N == 6


def test_ctas_need_t_false_and_default_t_false():
    S, N = Check_Time_and_Signal(np.arange(6.0), ndim={1}, method="M", need_T=False)
    assert isinstance(S, np.ndarray) and N == 6
    S2, T2, N2 = Check_Time_and_Signal(np.arange(6.0), ndim={1}, method="M", default_T=False)
    assert T2 is None and N2 == 6


def test_ctas_copy_semantics():
    src = np.arange(8.0)
    out_copy, _ = Check_Time_and_Signal(src, ndim={1}, method="M", need_T=False, copy=True)
    out_alias, _ = Check_Time_and_Signal(src, ndim={1}, method="M", need_T=False, copy=False)
    assert out_copy is not src and out_alias is src
    out_copy += 1.0
    assert src[0] == 0.0


def test_ctas_rejects_empty():
    with pytest.raises(ValueError, match="must not be empty"):
        Check_Time_and_Signal(np.array([], dtype=float), ndim={1}, method="M", need_T=False)


def test_ctas_squeeze_singleton():
    S, _ = Check_Time_and_Signal(np.zeros((1, 6)), ndim={1, 2}, method="M", need_T=False)
    assert S.shape == (6,)


def test_ctas_rejects_wrong_ndim():
    with pytest.raises(ValueError):
        Check_Time_and_Signal(np.zeros((2, 6)), ndim={1}, method="M", need_T=False)


def test_ctas_t_ascending_kept():
    t = np.arange(6.0)
    S, T, N = Check_Time_and_Signal(np.arange(6.0), T=t, ndim={1}, method="M")
    assert np.array_equal(T, t)


def test_ctas_t_duplicates_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        Check_Time_and_Signal(np.arange(4.0), T=np.array([0.0, 1.0, 1.0, 2.0]),
                              ndim={1}, method="M")


def test_ctas_t_length_mismatch():
    with pytest.raises(ValueError, match="length mismatch"):
        Check_Time_and_Signal(np.arange(4.0), T=np.arange(5.0), ndim={1}, method="M")


def test_ctas_t_2d_squeezed():
    S, T, _ = Check_Time_and_Signal(np.arange(5.0), T=np.arange(5.0).reshape(1, 5),
                                    ndim={1}, method="M")
    assert T.ndim == 1


def test_ctas_t_nan_rejected():
    with pytest.raises(ValueError, match="NaN or Inf"):
        Check_Time_and_Signal(np.arange(3.0), T=np.array([0.0, np.nan, 2.0]),
                              ndim={1}, method="M")


def test_ctas_descending_t_warns_and_reorders():
    S_src = np.array([10.0, 20.0, 30.0, 40.0])
    T_src = np.array([3.0, 2.0, 1.0, 0.0])
    with pytest.warns(UserWarning, match="not monotonically increasing"):
        S, T, _ = Check_Time_and_Signal(S_src, T=T_src, ndim={1}, method="M")
    assert np.array_equal(T, np.array([0.0, 1.0, 2.0, 3.0]))
    assert np.array_equal(S, np.array([40.0, 30.0, 20.0, 10.0]))
    assert np.array_equal(S_src, np.array([10.0, 20.0, 30.0, 40.0])), "不得改调用方数组"


def test_ctas_non_monotonic_t_warns_and_reorders():
    S_src = np.array([0.0, 1.0, 2.0, 3.0])
    T_src = np.array([0.0, 1.0, 3.0, 2.0])
    with pytest.warns(UserWarning):
        S, T, _ = Check_Time_and_Signal(S_src, T=T_src, ndim={1}, method="M")
    assert np.array_equal(T, np.array([0.0, 1.0, 2.0, 3.0]))
    assert np.array_equal(S, np.array([0.0, 1.0, 3.0, 2.0]))


def test_ctas_int_signal_with_float_T_truncates_to_duplicates():
    """特征化：T 被强制转成 S 的 dtype，整型信号 + 小数时间轴 → 截断成重复值。"""
    with pytest.raises(ValueError, match="duplicate"):
        Check_Time_and_Signal(np.arange(4, dtype=np.int32),
                              T=np.array([0.0, 0.4, 0.8, 1.2]), ndim={1}, method="M")


def test_ctas_axis_out_of_range_raises_value_error():
    """特征化：T 恒为 1-D，``axis>=1`` 在 ``Chunk.chunk`` 里被判越界 → ``ValueError``。"""
    with pytest.raises(ValueError, match="out of bounds"):
        Check_Time_and_Signal(np.arange(4.0), T=np.arange(4.0), ndim={1},
                              method="M", axis=1)


def test_ctas_safe_suspicious_bypasses_ndim():
    """特征化：`safe="suspicious"` 时 ndim 契约被跳过，2-D 输入原样通过。"""
    S, _ = Check_Time_and_Signal(np.zeros((2, 5)), ndim={1}, method="M",
                                 need_T=False, safe="suspicious")
    assert S.shape == (2, 5)


def test_ctas_safe_safe_on_list_fails_late():
    """特征化：`safe="safe"` 不做 array 化，list 输入在取 ``.shape`` 时 AttributeError。"""
    with pytest.raises(AttributeError):
        Check_Time_and_Signal([1.0, 2.0, 3.0], ndim={1}, method="M", need_T=False, safe="safe")


def test_ctas_memmap_copy_true_materialises():
    with temp_mm(fill=np.arange(8.0)) as (mm, _path, _d):
        out, _ = Check_Time_and_Signal(mm, ndim={1}, method="M", need_T=False, copy=True)
        assert not isinstance(out, np.memmap) and isinstance(out, np.ndarray)
        out[0] = -1.0
        assert mm[0] == 0.0


def test_ctas_memmap_copy_false_keeps_mapping():
    with temp_mm(fill=np.arange(8.0)) as (mm, _path, _d):
        out, _ = Check_Time_and_Signal(mm, ndim={1}, method="M", need_T=False, copy=False)
        assert isinstance(out, np.memmap)


def test_ctas_npy_roundtrip(tmp_path):
    p = tmp_path / "s.npy"
    np.save(p, np.arange(5.0))
    S, N = Check_Time_and_Signal(str(p), ndim={1}, method="M", need_T=False)
    assert np.array_equal(S, np.arange(5.0)) and N == 5
    S2, N2 = Check_Time_and_Signal(p, ndim={1}, method="M", need_T=False)
    assert N2 == 5


def test_ctas_npy_memmap_branch(monkeypatch, tmp_path):
    C = importlib.import_module("Modal_Decomposition.Utils.Check")

    p = tmp_path / "s.npy"
    np.save(p, np.arange(5.0))
    monkeypatch.setattr(C, "should_use_memmap", lambda *a, **k: True)
    S, N = Check_Time_and_Signal(p, ndim={1}, method="M", need_T=False, copy=False)
    assert isinstance(S, np.memmap) and N == 5
    S2, _ = Check_Time_and_Signal(p, ndim={1}, method="M", need_T=False, copy=True)
    assert not isinstance(S2, np.memmap), "copy=True 必须把只读映射物化成内存数组"


def test_ctas_rejects_non_npy_path(tmp_path):
    p = tmp_path / "s.txt"
    p.write_text("1 2 3", encoding="utf-8")
    with pytest.raises(ValueError, match="only .npy"):
        Check_Time_and_Signal(str(p), ndim={1}, method="M", need_T=False)


def test_ctas_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        Check_Time_and_Signal(str(tmp_path / "nope.npy"), ndim={1}, method="M", need_T=False)


def test_ctas_long_list_branch_is_broken(monkeypatch):
    """特征化（缺陷）：``_load`` 想把超长 list 灌进临时 memmap，但把 list 原样交给
    ``Chunk.fill_memmap``，后者要求 ndarray/memmap → ``TypeError``。

    即 ``LIST_MEM_THRESHOLD`` 这条分支**只要命中就一定抛错**（小 list 走不到，
    所以长期没被发现）。正确行为应是先 ``np.asarray`` 再填充。
    """
    C = importlib.import_module("Modal_Decomposition.Utils.Check")

    monkeypatch.setattr(C, "LIST_MEM_THRESHOLD", 4)
    with pytest.raises(TypeError, match="src must be an ndarray"):
        Check_Time_and_Signal([0.0, 1.0, 2.0, 3.0, 4.0], ndim={1}, method="M", need_T=False)


def test_ctas_dtype_float64_request():
    S, _ = Check_Time_and_Signal(np.arange(4, dtype=np.int16), ndim={1}, method="M",
                                 need_T=False, dtype="float64")
    assert S.dtype == np.float64


def test_ctas_idempotent():
    x = np.arange(16.0)
    a, _, _ = Check_Time_and_Signal(x, ndim={1}, method="M")
    b, _, _ = Check_Time_and_Signal(a, ndim={1}, method="M")
    assert np.array_equal(a, b) and a is not b


# --------------------------------------------------------------------------- #
# 12. 契约面 / 性能烟测
# --------------------------------------------------------------------------- #
def test_public_surface_matches_all():
    assert Check.__module__.endswith("Utils.Check")
    assert U.Check is Check and U.Check_Time_and_Signal is Check_Time_and_Signal
    assert set(Check.__dict__) >= {
        "return_dtype_kind", "not_numpy_arr", "not_numpy_memmap", "is_numeric",
        "is_real", "is_0d", "is_1d", "is_2d", "is_empty", "is_duplicates",
        "is_uniform", "is_infinite", "require_numeric", "require_ndim",
        "require_nonempty", "require_1d", "require_length_match",
        "validate_safe", "validate_mod", "to_array", "to_dtype",
    }


def test_predicates_are_linear_not_quadratic():
    """1 Mi 元素上三个谓词都应远低于 1 s（防 O(N²) 回归）。"""
    x = B.make_signal(1, kind="ramp")
    assert bool(Check.is_uniform(x)) is True
    assert bool(Check.is_duplicates(x)) is False
    assert bool(Check.is_infinite(x)) is True
    r = B.timeit(lambda: (Check.is_uniform(x), Check.is_duplicates(x)), repeat=3, warmup=1)
    assert r["best_ms"] < 1000.0, r


def test_size_constant_is_1mib():
    assert SIZE["1MB"] == 1048576


# --------------------------------------------------------------------------- #
# 13. 导出（csv + fig）
# --------------------------------------------------------------------------- #
def build_ctas_matrix() -> list[dict]:
    """把 CTAS 的主要分支跑一遍，记录"输入 → 结果"而不作为断言。"""
    rows: list[dict] = []

    def rec(case, fn):
        try:
            fn()
            rows.append({"case": case, "outcome": "ok", "detail": ""})
        except Exception as exc:                    # noqa: BLE001 - 故意全捕
            rows.append({"case": case, "outcome": type(exc).__name__,
                         "detail": str(exc)[:90]})

    S1 = lambda: Check_Time_and_Signal(np.arange(8.0), ndim={1}, method="M", need_T=False)
    rec("1d float64, need_T=False", S1)
    rec("1d float64, need_T=True", lambda: Check_Time_and_Signal(
        np.arange(8.0), ndim={1}, method="M", need_T=True))
    rec("2-D (2,8), ndim={1}", lambda: Check_Time_and_Signal(
        np.zeros((2, 8)), ndim={1}, method="M", need_T=False))
    rec("2-D (1,8), ndim={1,2}", lambda: Check_Time_and_Signal(
        np.zeros((1, 8)), ndim={1, 2}, method="M", need_T=False))
    rec("empty", lambda: Check_Time_and_Signal(np.array([]), ndim={1}, method="M", need_T=False))
    rec("nan", lambda: Check_Time_and_Signal(np.array([0.0, np.nan]), ndim={1},
                                             method="M", need_T=False))
    rec("int32", lambda: Check_Time_and_Signal(np.arange(4, dtype=np.int32), ndim={1},
                                               method="M", need_T=False))
    rec("T ascending", lambda: Check_Time_and_Signal(
        np.arange(4.0), T=np.arange(4.0), ndim={1}, method="M"))
    rec("T duplicate", lambda: Check_Time_and_Signal(
        np.arange(4.0), T=np.array([0.0, 1.0, 1.0, 2.0]), ndim={1}, method="M"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rec("T descending (warn)",
            lambda: Check_Time_and_Signal(np.arange(4.0), T=np.array([3.0, 2.0, 1.0, 0.0]),
                                          ndim={1}, method="M"))
    rec("T wrong length", lambda: Check_Time_and_Signal(
        np.arange(4.0), T=np.arange(5.0), ndim={1}, method="M"))
    rec("axis=1", lambda: Check_Time_and_Signal(
        np.arange(4.0), T=np.arange(4.0), ndim={1}, method="M", axis=1))
    rec("safe=safe + list", lambda: Check_Time_and_Signal(
        [1.0, 2.0], ndim={1}, method="M", need_T=False, safe="safe"))
    rec("safe=suspicious + 2-D", lambda: Check_Time_and_Signal(
        np.zeros((2, 5)), ndim={1}, method="M", need_T=False, safe="suspicious"))
    rec("dtype=float64", lambda: Check_Time_and_Signal(
        np.arange(4), ndim={1}, method="M", need_T=False, dtype="float64"))
    rec("bad safe", lambda: Check_Time_and_Signal(
        np.arange(4.0), ndim={1}, method="M", need_T=False, safe="nope"))
    return rows


def export() -> None:
    B.bootstrap_src()

    dtype_rows = build_dtype_matrix()
    B.write_csv(B.TABLE / "utils" / "check" / "check_dtype_matrix.csv", dtype_rows)

    case_rows = []
    for r in build_pred_matrix():
        case_rows.append({"section": "predicate", "case": r["case"],
                          "expected": r["expected"], "got": r["got"],
                          "verdict": r["verdict"]})
    for r in dtype_rows:
        case_rows.append({"section": "dtype", "case": r["dtype"],
                          "expected": f"{r['expected_kind_char']}/{r['expected_return_dtype_kind']}",
                          "got": f"{r['kind_char']}/{r['return_dtype_kind']}",
                          "verdict": r["verdict"]})
    B.write_csv(B.TABLE / "utils" / "check" / "check_cases.csv", case_rows)

    ctas_rows = build_ctas_matrix()
    B.write_csv(B.TABLE / "utils" / "check" / "check_ctas_matrix.csv", ctas_rows)

    # ---- 图：三张矩阵的判定热图 ----
    B.plt_style()
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(13.5, max(3.2, 0.26 * max(
        len(dtype_rows), len(ctas_rows))) + 1.0))

    ax = axes[0]
    labels = ["is_numeric", "is_real", "return_kind"]
    grid = np.array([[int(r["is_numeric"] == r["expected_is_numeric"]),
                      int(r["is_real"] == r["expected_is_real"]),
                      int(r["return_dtype_kind"] == r["expected_return_dtype_kind"])]
                     for r in dtype_rows])
    ax.imshow(grid, cmap="Greens", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(3), labels, rotation=30, ha="right")
    ax.set_yticks(range(len(dtype_rows)), [r["dtype"] for r in dtype_rows], fontsize=7)
    ax.set_title("dtype predicates vs expectation")

    ax = axes[1]
    preds = build_pred_matrix()
    pgrid = np.array([[int(r[k]) for k in ("is_0d", "is_1d", "is_2d", "is_empty")]
                      for r in preds])
    ax.imshow(pgrid, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(4), ["is_0d", "is_1d", "is_2d", "is_empty"], rotation=30, ha="right")
    ax.set_yticks(range(len(preds)), [r["case"] for r in preds], fontsize=7)
    ax.set_title("shape predicates")

    ax = axes[2]
    ok = np.array([[0 if r["outcome"] == "ok" else 1] for r in ctas_rows])
    ax.imshow(ok, cmap="OrRd", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks([0], ["raised?"])
    ax.set_yticks(range(len(ctas_rows)), [r["case"] for r in ctas_rows], fontsize=6)
    ax.set_title("Check_Time_and_Signal: ok / raised")

    fig.suptitle("Utils.Check — contract matrix (exported by tests/utils/test_Check_deep.py)")
    fig.tight_layout()
    B.savefig(fig, B.FIG / "utils" / "check" / "check_matrix.png")
    plt.close(fig)

    print(f"wrote {B.TABLE / 'utils' / 'check'}/*.csv")
    print(f"wrote {B.FIG / 'utils' / 'check' / 'check_matrix.png'}")
    B.print_table(ctas_rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    B.add_export_flag(ap)
    args = ap.parse_args()
    if args.export:
        export()
    else:
        for r in build_dtype_matrix():
            print(r)
