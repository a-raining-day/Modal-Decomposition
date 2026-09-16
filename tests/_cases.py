"""
Shared method list and minimal runtime parameter sets for the test suite.
"""

METHODS = [
    "CEEFD", "CEEMD", "CEEMDAN", "CEEMDAN", "EEMD", "EEMD", "EFD", "EMD",
    "EWT", "EWTpy", "FMD", "ICEEMDAN", "LMD", "MEMD", "RPSEMD", "SSA", "SVMD", "VMD",
]

# Minimal parameters keeping the suite fast; random methods are seeded.
CASES = {
    "CEEFD": {},
    "CEEMD": {"N_whitenoise": 5, "seed": 0},
    "CEEMDAN": {"trials": 5, "seed": 0},
    "CEEMDAN": {"trials": 5, "seed": 0},
    "EEMD": {"trials": 5, "seed": 0},
    "EEMD": {"trials": 5, "seed": 0},
    "EFD": {},
    "EMD": {},
    "EWT": {},
    "EWTpy": {},
    "FMD": {"K": 2, "max_iter": 5, "seed": 0},
    "ICEEMDAN": {"ensemble_size": 5, "max_imfs": 3, "seed": 0},
    "LMD": {"max_pf": 3},
    "MEMD": {"k": 32, "max_imf": 3},
    "RPSEMD": {"M": 2, "max_imf": 3},
    "SSA": {},
    "SVMD": {"num_modes": 3, "max_iter": 50},
    "VMD": {"num_imf": 3, "n": 100},
}

#: 依赖**可选**第三方包的方法 → 包名。缺包时 conftest 自动跳过该方法的全部用例
#: (EWTpy 走 ewtpy; EWT 是自研实现, 不依赖任何东西)。
OPTIONAL_DEPENDENCIES = {
    "EWTpy": "ewtpy",
}
