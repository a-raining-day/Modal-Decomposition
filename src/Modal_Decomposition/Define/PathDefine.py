import os

__all__ = ["ROOT_DIR", "DEFINE_DIR", "BASE_DIR", "OPERATOR_DIR", "ERROR_DIR", "UTILS_DIR"]

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_DIR = os.path.join(ROOT_DIR, "Base")
OPERATOR_DIR = os.path.join(BASE_DIR, "Operator")
ERROR_DIR = os.path.join(BASE_DIR, "Error")
UTILS_DIR = os.path.join(BASE_DIR, "Utils")
DEFINE_DIR = os.path.join(BASE_DIR, "Define")
