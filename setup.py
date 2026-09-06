"""
Modal-Decomposition 项目根构建脚本 —— Cython 加速扩展(_fht_native)的正式入口。

- PyPI / CI 构建: 由 pyproject.toml 的 build-system(setuptools + Cython + numpy)
  驱动, 本文件是扩展模块的唯一配置处;
- 本地开发(在仓库根目录运行, 编译产物直接放进源码包):
      python setup.py build_ext --inplace

说明:
- 扩展链接的 C 内核来自 Smithsonian Astrophysical Observatory "am" 项目
  (见 src/Modal_Decomposition/Utils/Hilbert/_C/_fht/ 各文件头部出处声明);
- -O3 / -ffast-math 仅适用于 GCC/Clang(Linux/macOS), Windows MSVC 用 /O2;
- NPY_TARGET_VERSION 让用 NumPy 2 头文件编译出的扩展在 NumPy>=1.19 运行时上
  依然可加载;
- cythonize 采用惰性触发: metadata/sdist 等不需要编译的阶段不依赖 Cython。
"""

import sys

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext as _build_ext

_FHT_DIR = "src/Modal_Decomposition/Utils/Hilbert"
_FHT_C_DIR = _FHT_DIR + "/_C/_fht"


def _native_extension() -> Extension:
    """_fht_native: Cython 包装 SAO 的 FHT / Hilbert C 实现。

    纯 Python 版 src/Modal_Decomposition/Utils/Hilbert/_fht.py 会自动检测该
    编译模块并优先使用; 找不到时回退到自身实现, 因此本扩展对库的功能是
    可选加速而非必需依赖。
    """
    compile_args = ["/O2"] if sys.platform == "win32" else ["-O3", "-ffast-math"]
    return Extension(
        "Modal_Decomposition.Utils.Hilbert._fht_native",
        sources=[
            _FHT_DIR + "/_fht_native.pyx",
            _FHT_C_DIR + "/transforms.c",
            _FHT_C_DIR + "/am_sysdep.c",
        ],
        include_dirs=[_FHT_C_DIR, _FHT_DIR],
        define_macros=[
            ("L1_CACHE_BYTES", "0x8000"),                     # am_sysdep.h: L1 = 32KB
            ("NPY_TARGET_VERSION", "NPY_1_19_API_VERSION"),   # 兼容 numpy>=1.19 运行时
        ],
        extra_compile_args=compile_args,
    )


class _LazyCythonBuildExt(_build_ext):
    """只有在真正编译(build_ext)时才执行 cythonize, 并注入 NumPy 头文件路径。"""

    def run(self):
        import numpy as np
        from Cython.Build import cythonize

        exts = cythonize(
            self.distribution.ext_modules,
            compiler_directives={"language_level": "3"},
        )
        for ext in exts:
            ext.include_dirs.append(np.get_include())
        self.distribution.ext_modules = exts
        super().run()


setup(
    ext_modules=[_native_extension()],
    cmdclass={"build_ext": _LazyCythonBuildExt},
)
