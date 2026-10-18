import os

from ..Define import PYTHON_VERSION, DEFAULT_ALLOCABLE_CPU_RATIO

__all__ = ["cpu_num", "allocable_cpu_num", "total_cpu", "set_allocable"]

cpu_num = None
allocable_cpu_num = None


def total_cpu() -> int:
    """
    本机可用逻辑 CPU 数 (至少 1)

    如版本 >= 3.13, 则使用 os.process_cpu_count, 否则 os.cpu_count()
    """
    global cpu_num
    if cpu_num is None:
        if PYTHON_VERSION.major == 3 and PYTHON_VERSION.minor >= 13:
            try:
                cpu_num = os.process_cpu_count()
            except Exception:
                cpu_num = None

        if cpu_num is None:
            cpu_num = os.cpu_count()

    return cpu_num

def set_allocable(value: int | float) -> int:
    """
    set allocable_cpu_num.
    int means cpu num, [1, total];
    float means cpu ratio, (0, 1];
    -1 means all.
    :param value:
    :return:
    """
    global cpu_num, allocable_cpu_num

    if isinstance(value, bool):
        raise TypeError(f"value must be int or float, got {value!r}")

    if isinstance(value, int):
        # 整数是绝对核数：取值 0 < value <= cpu_num；-1 是"全部"哨兵，等价于 cpu_num。
        if value == -1:
            allocable_cpu_num = cpu_num
            return allocable_cpu_num

        if value <= 0 or value > cpu_num:
            raise ValueError(f"num of value must in (0, {cpu_num})]")

        allocable_cpu_num = value
        return allocable_cpu_num

    if isinstance(value, float):
        if not (0.0 < value <= 1.0):
            raise ValueError(f"ratio of value must in (0, 1], got {value!r}")

        allocable_cpu_num = int(cpu_num * value)
        if allocable_cpu_num < 1:
            allocable_cpu_num = 1

        allocable_cpu_num = allocable_cpu_num
        return allocable_cpu_num

    raise TypeError(f"value must be int or float, got {type(value)!r}")

if cpu_num is None:
    total_cpu()

if allocable_cpu_num is None:
    allocable_cpu_num = set_allocable(DEFAULT_ALLOCABLE_CPU_RATIO)