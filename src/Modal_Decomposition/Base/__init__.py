"""
Base: 项目底层结构 (Cache / CPU / Memory / Chunk / Seed)。

层级: ``Define < Base < Utils`` —— 本包只允许 import Define, 不得 import Utils
或更上层; 反向依赖由各层的 getter + 进程级 import 缓存惰性承担。

包内顺序: ``{CPU, Memory} < Chunk``; ``Seed`` 面向最高层, 其余层不使用;
``Cache`` 只用标准库, 故放最前。
"""

from .Cache import cache, Cache
from .CPU import (
    cpu_num,
    allocable_cpu_num,
    total_cpu,
    set_allocable
)
from .Memory import (
    should_use_memmap,

    get_memory_snapshot,
    get_available_memory,
    get_commit_available,
    get_process_memory,

    memory_budget,

    MemoryPolicy,

    temp_memmap,
    drop_memmap,
    cleanup_memmaps
)
from .Chunk import (
    Chunk,
    CHUNK_MOD,
    CHUNK_MOD_ARGS
)
from .Seed import (
    Seed,
    Global_Seed
)
