"""
Size constants shared by the whole package.

所有涉及分块 / 内存策略的大小常量统一从本字典取用
(见 ``Utils.Chunk`` / ``Utils.Check`` / ``Utils.Memory`` /
``Utils.Monotonicity``), 避免各模块各自书写魔数。
"""

__all__ = ["SIZE"]

#: 基本字节大小字典: 1KB / 1MB / 1GB 三类基本常量。
#: 值为字节数 (int); 以"元素数"为单位的常量同样可用其倍数表达
#: (如 8M 元素 = ``8 * SIZE["1MB"]``)。
SIZE = \
{
    "1KB": 1024,            # 2**10 bytes
    "1MB": 1024 ** 2,       # 2**20 bytes
    "1GB": 1024 ** 3,       # 2**30 bytes
}
