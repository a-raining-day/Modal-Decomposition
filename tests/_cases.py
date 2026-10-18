"""
Shared method list and minimal runtime parameter sets for the test suite.

参数名与 ``src/Modal_Decomposition`` 的当前实现保持一致:
旧别名 ``trials`` / ``ensemble_size`` / ``N_whitenoise`` / ``max_imfs`` 已废弃,
改为 ``M`` / ``num_imf`` / ``num_imfs``。
"""

METHODS = [
    "EMD", "EEMD", "CEEMD", "CEEMDAN", "ICEEMDAN", "RPSEMD",
    "EFD", "CEEFD", "MEMD", "SSA", "FMD", "LMD", "VMD", "EWT", "SVMD",
]

#: Minimal parameters keeping the suite fast (random methods are seeded by the
#: caller through ``Base.Seed``).
CASES = {
    "EMD": {"num_imf": 3},
    "EEMD": {"num_imf": 3, "M": 5},
    "CEEMD": {"num_imf": 3, "M": 5},
    "CEEMDAN": {"num_imf": 3, "M": 5},
    "ICEEMDAN": {"num_imf": 3, "M": 5},
    "RPSEMD": {"num_imf": 3, "sin_smaple_num": 4},

    "EFD": {"max_IMFs": 3},
    "CEEFD": {},
    "MEMD": {"k": 32, "max_imf": 3},
    "SSA": {"window_size": 64},
    "FMD": {"K": 2, "max_iter": 5},
    "LMD": {"max_pf": 3},
    "VMD": {"num_imf": 3, "n": 100},
    "EWT": {"num_imfs": 3},
    "SVMD": {"num_modes": 3, "max_iter": 50},
}

#: 依赖**可选**第三方包的方法 -> 包名。缺包时 conftest 自动跳过该方法的全部用例。
#: 当前 15 个方法全部是自研实现, 无外部硬依赖 (scipy 属必装)。
OPTIONAL_DEPENDENCIES: dict[str, str] = {}
