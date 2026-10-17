"""层次导入自检: 自动发现包目录名, 再验证 Define / Base / Utils 能否各自 import。

运行: .venv\\Scripts\\python.exe tests\\check_layers.py
"""
import importlib
import importlib.util
import os
import sys
import traceback
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
print("cwd    :", os.getcwd())
print("src dir:", SRC.is_dir())
dirs = sorted(p for p in SRC.iterdir() if p.is_dir())
print("src 下目录:", [(repr(p.name), (p / "__init__.py").is_file()) for p in dirs])

PKG = next((p for p in dirs if (p / "__init__.py").is_file()), None)
if PKG is None:
    raise SystemExit("!! src 下找不到带 __init__.py 的包目录")
NAME = PKG.name
print("包名   :", repr(NAME), "| pure-ascii:", NAME.isascii(), "| len:", len(NAME))

sys.path.insert(0, str(SRC))
parent = types.ModuleType(NAME)
parent.__path__ = [str(PKG)]
parent.__package__ = NAME
sys.modules[NAME] = parent

print("find_spec:", {n: importlib.util.find_spec(f"{NAME}.{n}") is not None
                     for n in ("Define", "Base", "Utils")})

report = {}
for layer in ("Define", "Base", "Utils"):
    try:
        importlib.import_module(f"{NAME}.{layer}")
        report[layer] = "OK"
    except Exception as exc:  # noqa: BLE001
        report[layer] = f"FAIL {type(exc).__name__}: {exc}"
        traceback.print_exc(limit=8)
print("import :", report)
print("loaded :", sorted(k for k in sys.modules if k.startswith(NAME)))

utils = sys.modules.get(f"{NAME}.Utils")
if utils is not None:
    import numpy as np
    from importlib import import_module

    mirror_new = import_module(f"{NAME}.Utils.Mirror").mirror
    print("Mirror smoke       :", mirror_new(np.arange(6.0), nbsym=2).tolist())
    print("find_peaks(mod=min):", utils.find_peaks(np.sin(np.arange(60.0) / 3), mod="min")[0][:4])
    print("Check.is_uniform   :", utils.Check.is_uniform(np.arange(10.0)))
    for name in ("mirror", "MIRROR_MODS", "is_monotonic", "is_uniform", "Chunk",
                 "temp_memmap", "find_peaks", "fast_poly", "slepian", "hilbert", "envelope"):
        print(f"  Utils.{name:<13}", hasattr(utils, name))
    print("  Utils.mirror 来自:", getattr(getattr(utils, "mirror", None), "__module__", None))
