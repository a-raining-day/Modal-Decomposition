import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_DIR = os.path.join(ROOT_DIR, "Base")
C_METHOD_DIR = os.path.join(BASE_DIR, "C_method")
ERROR_DIR = os.path.join(BASE_DIR, "Error")
UTILS_DIR = os.path.join(BASE_DIR, "Utils")
TEMP_DIR = os.path.join(BASE_DIR, "_temp")

if __name__ == "__main__":
    print(ROOT_DIR)