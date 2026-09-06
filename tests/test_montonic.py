"""
analysis the speed of the 'monotonic' function and 'is_monotonic' function
"""

import pytest

from src.Modal_Decomposition.Utils.Monotonicity import *

import time
import math
import numpy as np
from typing import Literal, Callable, Dict

BYTE_SIZE = \
{
    "1MB": 1024 * 1024,
    "1GB": 1024 * 1024 * 1024
}

dtype_map = \
{
    "F16": np.float16,
    "F32": np.float32,
    "F64": np.float64,
}

def to_sci_str_rounded(x: float, ndigits: int = 6) -> str:
    if x == 0:
        return "0e+00"
    if math.isinf(x) or math.isnan(x):
        return str(x)

    exp = math.floor(math.log10(abs(x)))
    mant = x / (10 ** exp)

    if abs(mant) >= 10:
        mant /= 10
        exp += 1

    # 保留 ndigits 位有效数字
    mant_rounded = round(mant, ndigits - 1)
    # 四舍五入可能进位，例如 9.999 -> 10.0
    if abs(mant_rounded) >= 10:
        mant_rounded /= 10
        exp += 1

    if mant_rounded.is_integer():
        mant_str = str(int(mant_rounded))
    else:
        mant_str = f"{mant_rounded:.{ndigits-1}g}"

    return f"{mant_str}e{exp:+d}"


def data_loader \
(
    size_in_bytes: int,
    data_type: Literal["F16", "F32", "F64"] = "F16",
    mod: Literal["up", "down", "random"] = "up"
) -> np.ndarray:
    """
    生成指定字节大小、指定数据类型的数组。

    Parameters
    ----------
    size_in_bytes : int
        目标数据大小（字节）。如 1MB = 1024 * 1024
    data_type : {"F16", "F32", "F64"}
        数据类型：F16 → np.float16, F32 → np.float32, F64 → np.float64
    mod : {"up", "down", "random"}
        up = 单调递增, down = 单调递减, random = 随机

    Returns
    -------
    np.ndarray
        一维数组，元素类型为指定浮点类型。
    """

    dtype = dtype_map[data_type]
    itemsize = np.dtype(dtype).itemsize
    if size_in_bytes % itemsize != 0:
        raise ValueError(f"size_in_bytes ({size_in_bytes}) 必须是 itemsize ({itemsize}) 的整数倍")
    length = int(size_in_bytes // itemsize)

    if mod == "up":
        # 严格递增：用整数创建后转为浮点，避免浮点累积误差导致重复
        arr = np.linspace(start=-1000, stop=1000, num=length, dtype=dtype)
    elif mod == "down":
        # 严格递减：递增后反转
        arr = np.linspace(start=-1000, stop=1000, num=length, dtype=dtype)[::-1]
    elif mod == "random":
        # 随机均匀分布 [0, 1)
        arr = np.random.rand(length).astype(dtype)
    else:
        raise ValueError("mod 只能为 'up', 'down' 或 'random'")

    return arr

def time_analysis(
    func: Callable,
    size_in_bytes: int,
    data_type: Literal["F16", "F32", "F64"] = "F16",
    mod: Literal["up", "down", "random"] = "up",
    rounds: int = 5,
    **kwargs
) -> Dict[str, Dict[str, float]]:
    """
    测量函数平均耗时（多次运行，去除最大最小值后取平均）。

    Parameters
    ----------
    func : Callable
        被测试的函数。
    size_in_bytes : int
        生成的数据大小（字节）。
    data_type : {"F16", "F32", "F64"}
        数据类型。
    mod : {"up", "down", "random"}
        数据模式。
    rounds : int, default=5
        重复运行次数，必须 >= 3 才能剔除极值。
    **kwargs :
        传递给 func 的额外参数。

    Returns
    -------
    Dict[str, Dict[str, float]]
        结果字典，结构与原实现一致，新增了 `rounds` 字段。
    """
    data = data_loader(size_in_bytes, data_type, mod)

    # 收集所有轮次的时间
    times = []
    mode = None
    for _ in range(rounds):
        start = time.perf_counter()          # 使用高精度计时器
        mode = func(data, **kwargs)
        end = time.perf_counter()
        times.append(end - start)

    # 去除最大最小值后取平均（至少需要3个样本）
    if len(times) > 2:
        times.remove(max(times))
        times.remove(min(times))
    avg_time = sum(times) / len(times) if times else 0.0

    result = {
        f"{func.__name__}": {
            "time": to_sci_str_rounded(avg_time),   # 平均时间，科学计数法
            "mode": mode,
            "byte_size": size_in_bytes,
            "data_type": data_type,
            "mod": mod,
            "rounds": rounds,
            "raw_times": [to_sci_str_rounded(t) for t in times]  # 剔除后的各次耗时
        }
    }
    return result

def test_is_monotonic():
    # is_monotonic
    """
    repeats times -> 10
    byte size | data type | mod     | strict | chunk size | assign mod | mode | time (s)

    (before optimization: forced float64 upcast + np.diff)
    0.5 MB    | F16       | up      | False  | 1MB        | monotonic  | True | 1.0711e-3
    0.5 MB    | F32       | up      | False  | 1MB        | monotonic  | True | 6.5704e-4
    0.5 MB    | F64       | up      | False  | 1MB        | monotonic  | True | 3.2713e-5
    0.5 MB    | F64       | up      | True   | 1MB        | monotonic  | True | 3.9263e-5
    0.5 MB    | F64       | random  | False  | 1MB        | monotonic  | False| 4.3700e-5
    5 MB      | F16       | up      | False  | 1MB        | monotonic  | True | 1.6735e-2
    5 MB      | F32       | up      | False  | 1MB        | monotonic  | True | 7.6652e-3
    5 MB      | F64       | up      | False  | 1MB        | monotonic  | True | 1.5934e-3
    5 MB      | F64       | up      | True   | 1MB        | monotonic  | True | 1.9938e-3
    5 MB      | F64       | random  | False  | 1MB        | monotonic  | False| 1.5620e-3
    500 MB    | F64       | up      | False  | 10MB       | monotonic  | True | 1.6199e-3  (mis-measured; real ~2.3e-1)
    1 GB      | F16       | up      | False  | 100MB      | monotonic  | True | 3.2337e+0

    (after optimization: native dtype + adjacent comparisons, no np.diff)
    0.5 MB    | F16       | up      | False  | 1MB        | monotonic  | True | 5.45e-4
    0.5 MB    | F32       | up      | False  | 1MB        | monotonic  | True | 2.9e-5
    0.5 MB    | F64       | up      | False  | 1MB        | monotonic  | True | 2.4e-5
    0.5 MB    | F64       | up      | True   | 1MB        | monotonic  | True | 2.4e-5
    0.5 MB    | F64       | random  | False  | 1MB        | monotonic  | False| 4.2e-5
    5 MB      | F16       | up      | False  | 1MB        | monotonic  | True | 6.14e-3
    5 MB      | F32       | up      | False  | 1MB        | monotonic  | True | 7.06e-4
    5 MB      | F64       | up      | False  | 1MB        | monotonic  | True | 2.76e-4
    5 MB      | F64       | up      | True   | 1MB        | monotonic  | True | 2.43e-4
    5 MB      | F64       | random  | False  | 1MB        | monotonic  | False| 4.46e-4
    500 MB    | F64       | up      | False  | 10MB       | monotonic  | True | 8.46e-2
    1 GB      | F16       | up      | False  | 100MB      | monotonic  | True | 1.589e+0
    2 GB      | F16       | up      | False  | 500MB      | monotonic  | True | 3.23e+0
    3 GB      | F32       | up      | False  | 500MB      | monotonic  |

    Notes
    -----
    - The f16/f32 "up" data (np.linspace cast to f16/f32) contains many ties,
      so strict=True on those rows would return False (data issue, not code).
    - The 500 MB row above was re-measured: 1.6 ms is physically impossible for
      62.5M elements (the isfinite mask alone takes >= ~25 ms).
    """
    name = is_monotonic.__name__
    func = is_monotonic

    data_type = "F16"
    chunk_size = (BYTE_SIZE["1MB"] * 500) // np.dtype(dtype_map[data_type]).itemsize

    result = time_analysis(func, size_in_bytes=BYTE_SIZE["1GB"] * 3, data_type=data_type, mod="up", rounds=10, strict=False, chunk_size=chunk_size)
    result = result[name]

    print(f"\nresult for -> {name}")
    print(f"byte size = {result['byte_size']} | data type is {result['data_type']} | mod is {result['mod']}")
    print(f"the result is -> \\")
    print(f"time is {result['time']}s | mode is {result['mode']}")