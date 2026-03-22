"""
Python version: 3.11

Lib and Version:
    colorama - 0.4.6

Only accessed by:
    __init__.py, colorful_print.py

Modify:
    2026.3.4
"""

import colorama as cm
from typing import Literal

__all__ = ["Color"]

Fore = cm.Fore

class Color:
    black = Fore.BLACK
    red = Fore.RED
    green = Fore.GREEN
    yellow = Fore.YELLOW
    blue = Fore.BLUE
    magenta = Fore.MAGENTA
    cyan = Fore.CYAN
    white = Fore.WHITE

