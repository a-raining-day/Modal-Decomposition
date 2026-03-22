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
from typing import Literal, Any, Union
# from _typeshed import SupportsWrite
import enum
from .color_define import Color


__all__ = ["printc"]

cm.init(autoreset=True)

Fore = cm.Fore

Reset = cm.Style.RESET_ALL

def printc(*values: Any, color: Literal["black", "red", "green", "yellow", "blue", "magenta", "cyan", "white"] = None, sep: str = ' ', end: str = '\n', **kwargs) -> bool:
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
    :param kwargs: accept 'c' as 'color' parameter, accept 'R' as "if get exception raise"
    :return: bool: the result of print
    """

    # deal the kwargs
    if (_color := kwargs.get("c", None)) is not None:
        color = _color

    R = kwargs.get("R", True)

    # deal color
    if color is None:
        print(*values, sep=sep, end=end)
        return True

    c = getattr(Color, color, Color.white)  # check and appoint color

    # for values' type
    if len(values) == 1:  # only one element been input
        text: str = str(values[0])

    else:
        try:
            text: str = sep.join(str(args) for args in values)
        except:
            if R:
                raise
            else:
                return False

    try:
        print(c + text, end=end)
        return True

    except:
        if R:
            raise
        else:
            return False

if __name__ == '__main__':
    # print(Color.yellow + "hello world")
    printc([1, 2, 3], color="red")

"""
    BLACK           = 30
    RED             = 31
    GREEN           = 32
    YELLOW          = 33
    BLUE            = 34
    MAGENTA         = 35
    CYAN            = 36
    WHITE
"""