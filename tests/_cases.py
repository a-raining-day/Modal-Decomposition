"""
Shared method list and minimal runtime parameter sets for the test suite.
"""

METHODS = [
    "CEEFD", "CEEMD", "CEEMDAN", "EEMD", "EFD", "EMD", "EWT", "FMD",
    "ICEEMDAN", "LMD", "MEMD", "RPSEMD", "SSA", "SVMD", "VMD",
]

# Minimal parameters keeping the suite fast; random methods are seeded.
CASES = {
    "CEEFD": {},
    "CEEMD": {"N_whitenoise": 5, "seed": 0},
    "CEEMDAN": {"trials": 5, "seed": 0},
    "EEMD": {"trials": 5, "seed": 0},
    "EFD": {},
    "EMD": {},
    "EWT": {},
    "FMD": {"K": 2, "max_iter": 5, "seed": 0},
    "ICEEMDAN": {"ensemble_size": 5, "max_imfs": 3, "seed": 0},
    "LMD": {"max_pf": 3},
    "MEMD": {"k": 32, "max_imf": 3},
    "RPSEMD": {"M": 2, "max_imf": 3},
    "SSA": {},
    "SVMD": {"num_modes": 3, "max_iter": 50},
    "VMD": {"K": 3},
}
