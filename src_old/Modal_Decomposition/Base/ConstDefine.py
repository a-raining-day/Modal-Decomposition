from ._Define_Struct import *
from enum import Enum

__all__ = ['ConstDefine', 'Reference', 'Name']

# ------ Const Define ------
class Monotony(Enum):
    Monotonic = 0

    Increasing = 1
    Decreasing = 2

    StrictIncreasing = 3
    StrictDecreasing = 4

Reference = \
{
    "CEEMDAN": "10.1109/ICASSP.2011.5947265",
    "CEEFD": "10.3969/j.issn.1001-4551.2023.07.001",
    "CEEMD": "10.1016/j.jhydrol.2020.124647",
    "EEMD": "10.1142/S1793536909000047",
    "EFD": "10.1016/j.ymssp.2021.108155",
    "EMD": "10.1098/rspa.1998.0193",
    "EWT": "10.48550/arXiv.2304.06274",
    "FMD": "10.1109/TIE.2022.3156156",
    "ICEEMDAN": "10.1007/s10470-021-01901-3",
    "LMD": "10.1098/rsif.2005.0058",
    "MEMD": "10.48550/arXiv.2206.00926",
    "RPSEMD": "10.1109/LSP.2016.2537376",
    "SSA": "10.1016/j.mex.2020.101015",
    "SVMD": "10.1016/j.sigpro.2020.107610",
    "VMD": "10.1109/TSP.2013.2288675",
}

Name = \
{
    "CEEMDAN": "Complete Ensemble Empirical Mode Decomposition with Adaptive Noise",
    "CEEFD": "Cyclic Envelop Empirical Fourier Decomposition",
    "CEEMD": "Complementary Ensemble Empirical Mode Decomposition",
    "EEMD": "Ensemble Empirical Mode Decomposition	",
    "EFD": "Empirical Fourier Decomposition",
    "EMD": "Empirical Mode Decomposition",
    "EWT": "Empirical Wavelet Transform",
    "FMD": "Filtered Mode Decomposition",
    "ICEEMDAN": "Improved Complete Ensemble Empirical Mode Decomposition with Adaptive Noise",
    "LMD": "Local Mean Decomposition",
    "MEMD": "Multivariate Empirical Mode Decomposition",
    "RPSEMD": "Random Phase Sinusoidal Assisted Empirical Mode Decomposition",
    "SSA": "Singular Spectrum Analysis",
    "SVMD": "Successive Variational Mode Decomposition",
    "VMD": "Variational Mode Decomposition",
}