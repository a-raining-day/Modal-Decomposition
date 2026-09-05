from dataclasses import dataclass
from typing import Literal

# __all__ = ["Config", "CEEFDConfig", "CEEMDConfig"]

class Config:
    pass

@dataclass(frozen=True)
class CEEFDConfig(Config):
    fs: int | float
    min_peak_distance: int
    envelop_iter: int

@dataclass(frozen=True)
class CEEMDConfig(Config):
    N_whitenoise: int
    beta: float
    max_imf: int
    dead_line: int

@dataclass(frozen=True)
class CEEMDANConfig(Config):
    trials: int
    noise_scale: int | float
    nbsym: int
    max_imf: int
    range_thr: float
    total_power_thr: float
    noise_kind: Literal["normal", "uniform"]
    noise_seed: int
    spline_kind: Literal["akima", "cubic", "pchip", "cubic_hermite", "slinear", "quadratic", "linear"]
    extrema_detection: Literal["simple", "parabol"]

@dataclass(frozen=True)
class EEMDConfig(Config):
    trials: int
    noise_width: float
    max_imf: int
    noise_seed: int

@dataclass(frozen=True)
class EFDConfig(Config):
    max_imf: int

@dataclass(frozen=True)
class EMDConfig(Config):
    nbsym: int
    spline_kind: Literal["akima", "cubic", "pchip", "cubic_hermite", "slinear", "quadratic", "linear"]
    max_imf: int