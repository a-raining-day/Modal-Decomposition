"""
Get the last memory of the environment.
"""

import psutil as ps

def EnvironmentMemory() -> tuple:
    """
    Get the last memory of the environment.
    :return: RSS, VMS
    """

    process = ps.Process()

    Memory = process.memory_info()

    RSS = Memory.rss
    VMS = Memory.vms

    return (RSS, VMS)