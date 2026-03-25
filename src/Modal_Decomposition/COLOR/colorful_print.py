"""
Python version: 3.11

Lib and Version:
    colorama - 0.4.6

Only accessed by:
    All

Modify:
    2026.3.4
"""

import colorama as cm
from typing import Literal, Any

__all__ = ["printc"]

Fore = cm.Fore

Color = \
{
    "black": Fore.BLACK,
    "red": Fore.RED,
    "green": Fore.GREEN,
    "yellow": Fore.YELLOW,
    "blue": Fore.BLUE,
    "magenta": Fore.MAGENTA,
    "cyan": Fore.CYAN,
    "white": Fore.WHITE,
}

def printc(*values: Any, color: Literal["black", "red", "green", "yellow", "blue", "magenta", "cyan", "white"] = None, sep: str = ' ', end: str = '\n') -> bool:
    """
    (don't think much, just use as print)\n
    A function that can make output be colorful.\n
    If the function is successful, will default return True. If 'R' parameter is not change, when fail will Raise.(make 'R'=False, will return False only)\n
    If you don't choose color or input a error parameter, default white.\n
    kwargs: you can input 'c' to cover the "color" parameter, and use 'R' to choose whether raise the error, default is raise(R=True).\n
    Reference Lib: colorama

    :param values: Any, like values in print
    :param color: color for output
    :param sep: print parameter
    :param end: print parameter
    :return: bool: the result of print
    """

    if not hasattr(printc, '_initialized'):
        cm.init(autoreset=True)
        printc._initialized = True

    # deal color
    if color is None:
        print(*values, sep=sep, end=end)
        return True

    c = Color.get(color, Color["white"])  # check and appoint color

    # for values' type
    if len(values) == 1:  # only one element been input
        text: str = str(values[0])

    text: str = sep.join(str(args) for args in values)

    print(c + text, end=end)
