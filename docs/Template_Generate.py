"""
Python version:
    3.10.11

Role:  (if None write None)
    Help to generate the template page.

Lib and Version:  (if None write None)
    clipboard - 0.0.4

Only accessed by:  (must)
    All

Modify:  (must)
    2026.3.25

Description: (if None write None)
    You can write the template page with this file, and it will be copied to clipboard auto.
"""

from datetime import datetime
from typing import Dict
import platform
import subprocess
from src.Modal_Decomposition.COLOR import printc

Template = \
{
    "main":
    """
Python version:  
    {version}

Role:  (if None write None)
    {Role}

Lib and Version:  (if None write None)
    {package_version}

Only accessed by:  (must)
    {Object}

Modify:  (must)
    {time}
    
Description: (if None write None)
    {description}
    """,

    "sub-lib":
    """
Python version:  (must)
    {version}

Lib and Version:  (if None write None)
    {package_version}

Only accessed by:  (must)
    {Object}

Modify:  (must)
    {time}
    
Description: (if None write None)
    {description}
    """,

    "function":
    """
the description of function:
    {description}

    the *args and **kwargs:
        *args:
            {args}
            
        **kwargs:
            {kwargs}       
    """
}

def _input(string: str) -> str | None:
    inputs = input(string)
    if inputs == "..." or inputs == "pass":
        return None

    else:
        return inputs

def generate_template(template: Dict[str, str], verbose: bool = True, clip: bool = True) -> str:
    version = None
    Role = None
    package_version = None
    Object = None
    time = datetime.now()
    time = f"{time.year}.{time.month}.{time.day}"
    description = None
    args = None
    kwargs = None

    mode = input("please choose template mode: ('main' | 'sub-lib' | 'function')\n>> ")
    print("the next steps, if you write '...' or 'pass' will wrote by None auto...")

    match mode:
        case "main":

            print("\nthe total description of the entrance or main of lib (for example: the description of the __init__.py)\n")

            template: str = template["main"]

            version = _input("the python version: ('...' and 'pass' will write your default python version -> platform.python_version()) \n>> ")
            if version is None:
                version = platform.python_version()

            package_version = _input("the imported package and version: (template: package - version | package1 - version, use '|' split different libs)\n>> ")
            if package_version is not None:
                package_version = package_version.split('|')
                f = lambda x: x.strip()
                package_version = list(map(f, package_version))
                package = ""
                for idx, i in enumerate(package_version):
                    package += i
                    if idx != len(package_version) - 1:
                        package += '\n\t'

                package_version = package

            Role = _input("this file's role is:\n>> ")

            Object = _input("this file can be accessed by whom? (default by All)\n>> ")
            if Object is None:
                Object = "All"

            description = _input("the description of this file:\n>> ")

            formation = \
            {
                "version": f"{version}",
                "package_version": f"{package_version}",
                "Role": f"{Role}",
                "Object": f"{Object}",
                "time": f"{time}",
                "description": f"{description}"
            }

            template = template.format(**formation)

        case "sub-lib":
            template = template["sub-lib"]

            print("\nthe description of sub-lib (for example: the description of the color_define.py)\n")

            version = _input("the python version: ('...' and 'pass' will write your default python version -> platform.python_version())\n>> ")
            if version is None:
                version = platform.python_version()

            package_version = _input("the imported package and version: (template: package - version)\n>> ")
            if package_version is not None:
                package_version = package_version.split('|')
                f = lambda x: x.strip()
                package_version = list(map(f, package_version))
                package = ""
                for idx, i in enumerate(package_version):
                    package += i
                    if idx != len(package_version) - 1:
                        package += '\n\t'

                package_version = package

            # Role = _input("this file's role is:\n>> ")

            Object = _input("this file can be accessed by whom? (default by All)\n>> ")
            if Object is None:
                Object = "All"

            description = _input("the description of this file is: \n>> ")

            formation = \
            {
                "version": f"{version}",
                "package_version": f"{package_version}",
                # "Role": Role,
                "Object": f"{Object}",
                "time": f"{time}",
                "description": f"{description}"
            }

            template = template.format(**formation)

        case "function":
            template = template["function"]

            description = _input("the description of this function: \n>> ")

            args = _input("the effect of args:\n>> ")

            kwargs = _input("the effect of kwargs:\n>> ")

            formation = \
            {
                "description": f"{description}",
                "args": f"{args}",
                "kwargs": f"{kwargs}"
            }

            template = template.format(**formation)

        case x:
            raise ValueError(f"Hope get 'main', 'sub-lib' or 'function', But {x}!")

    if verbose:
        print(template)

    if clip:
        Clip(template)
        printc("\n\n(the generated template has been copied to clipboard)", color="magenta")

    return template

def Clip(text) -> None:
    system = platform.system()

    try:
        match system:
            case "Darwin":
                subprocess.run("pbcopy", text=True, input=text, check=True)

            case "Windows":
                subprocess.run("clip", text=True, input=text, check=True)

            case _:
                try:
                    subprocess.run("xclip -selection clipboard", shell=True, text=True,
                                   input=text, check=True)
                except:
                    subprocess.run("wl-copy", text=True, input=text, check=True)

    except Exception as e:
        print(f"Error in Clip: {e}")
        return

if __name__ == '__main__':
    returns = generate_template(Template)