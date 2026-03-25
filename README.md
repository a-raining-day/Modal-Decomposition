# Modal Decomposition

## Introduction

There are many methods of medal decomposition, but there are not a lib can include them all yet.

In order to integrate the modal decomposition method as comprehensive as possible, I make this lib.

Hope my lib can help you.

## Entrance

All entrance of functions or class are stored in `Modal_Decomposition/__init__.py`

## Modal Decomposition

| method   | description                                                                 |             use             |                                                                                                                                                                     resource(doi and link)                                                                                                                                                                      |
|----------|:----------------------------------------------------------------------------|:---------------------------:|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
| CEEMDAN  | Complete Ensemble Empirical Mode Decomposition with Adaptive Noise          | `Function.CEEMDAN(siganl)`  |                                                                                                                                      [10.1109/ICASSP.2011.5947265](https://ieeexplore.ieee.org/abstract/document/5947265)                                                                                                                                       |
| CEEFD    | Complementary Ensemble Empirical Fourier Decomposition                      |  `Function.CEEFD(signal)`   | [10.27623/d.cnki.gzkyu.2024.000865](https://kns.cnki.net/kcms2/article/abstract?v=ZBmzy5eCHcaCVlEIopIAIsrkbpuktZOqmNTgm0KCGgooXOhG-T125e5DHg42YfQeZGE2n7IJBcR4uG6l7zEZRrYO8hwN1pNXAXORSZFZox26KvLzulXRctDPfZRS-d40m7Ntx-o4tnGzqAeud3gb4MMXSwWOXDKd0PV_c8K_yO_XZdwkhX5Q-_-CxmpLYlY_xAXga0yW9v4&uniplatform=NZKPT&captchaId=bc290be6-a948-48a8-a7ea-20cc128b686a) |
| CEEMD    | Complementary Ensemble Empirical Mode Decomposition                         |  `Function.CEEMD(siganl)`   |                                                                                                                            [10.1016/j.jhydrol.2020.124647](https://www.sciencedirect.com/science/article/abs/pii/S0022169420301074)                                                                                                                             |
| EEMD     | Ensemble Empirical Mode Decomposition                                       |   `Function.EEMD(signal)`   |                                                                                              [10.1142/S1793536909000047](https://www.semanticscholar.org/paper/Ensemble-Empirical-Mode-Decomposition%3A-a-Data-Wu-Huang/a97ee1d4a15c04160c323bd650e9cb9dff9dfced)                                                                                               |
| EFD      | Empirical Fourier Decomposition                                             |   `Function.EFD(signal)`    |                                                                                                                        [10.1016/j.ymssp.2021.108155](https://www.sciencedirect.com/science/article/abs/pii/S0888327021005355?via%3Dihub)                                                                                                                        |
| EMD      | Empirical Mode Decomposition                                                |   `Function.EMD(signal)`    |                                                                                              [10.1098/rspa.1998.0193](https://www.semanticscholar.org/paper/The-empirical-mode-decomposition-and-the-Hilbert-Huang-Shen/3842d81b0375dae8ae92734aa2a5d4aeed7a91d1)                                                                                               |
| EWT      | Empirical Wavelet Transform                                                 |   `Function.EWT(signal)`    |                                                                                                                                                  [10.48550/arXiv.2304.06274](https://arxiv.org/abs/2304.06274)                                                                                                                                                  |
| FMD      | Filtered Mode Decomposition                                                 |   `Function.MEMD(signal)`   |                                                                                                                                            [10.1109/TIE.2022.3156156](https://ieeexplore.ieee.org/document/9732251)                                                                                                                                             |
| ICEEMDAN | Improved Complete Ensemble Empirical Mode Decomposition with Adaptive Noise | `Function.ICEEMDAN(signal)` |                                                                                                                                [10.1007/s10470-021-01901-3](https://link.springer.com/article/10.1007/s10470-021-01901-3#citeas)                                                                                                                                |
| LMD      | Local Mean Decomposition                                                    |   `Function.LMD(signal)`    |                                                                                                                                     [10.1098/rsif.2005.0058](https://royalsocietypublishing.org/doi/10.1098/rsif.2005.0058)                                                                                                                                     |
| MEMD     | Multivariate Empirical Mode Decomposition                                   |   `Function.MEMD(signal)`   |                                                                                                                                                  [10.48550/arXiv.2206.00926](https://arxiv.org/abs/2206.00926)                                                                                                                                                  |
| RPSEMD   | Random Phase Sinusoidal Assisted Empirical Mode Decomposition               |  `Function.RPSEMD(signal)`  |                                                                                                                                            [10.1109/LSP.2016.2537376](https://ieeexplore.ieee.org/document/7423702)                                                                                                                                             |
| SSA      | Singular Spectrum Analysis                                                  |   `Function.SSA(signal)`    |                                                                                                                                [10.1016/j.mex.2020.101015](https://www.sciencedirect.com/science/article/pii/S2215016120302351)                                                                                                                                 |
| SVMD     | Sequential Variational Mode Decomposition                                   |   `Function.SVMD(signal)`   |                                                                                                                             [10.1016/j.sigpro.2020.107610](https://www.sciencedirect.com/science/article/abs/pii/S0165168420301535)                                                                                                                             |
| VMD      | Variational Mode Decomposition                                              |   `Function.VMD(signal)`    |                                                                                                                             [10.1016/j.sigpro.2020.107610](https://www.sciencedirect.com/science/article/abs/pii/S0165168420301535)                                                                                                                             |

## Install

You can install by:
```shell
git clone https://github.com/a-raining-day/Motal-Decomposition.git
cd Motal-Decomposition
pip install -r requirements.txt
```

*Well, some libs should installed by yourself, don't use `pip install -r requirements.txt`, because `torch, torchvision` need installed on Official Web*


## Dependence

This lib's dependence are:

***Python: 3.10***

- antropy
- colorama (for printc)
- einops
- EMD-signal
- ewtpy
- numba
- numpy
- scipy
- vmdpy

*Other dependence please read "requirements.txt"*

## Codes Resource

All codes from :

- Github:
  - EMD-signal -> EMD, CEEFD, CEEMDAN, EEMD
  - ewtpy -> EWT
  - vmdpy -> VMD


- Myself:
  - CEEMD, EFD, FMD, ICEEMDAN, LMD, MEMD, RPSEMD, SSA, SVMD

## Url

This lib's url is: https://github.com/a-raining-day/Modal-Decomposition