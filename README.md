# Modal Decomposition

## Introduction

There are many methods of medal decomposition, but there are not a lib can include them all yet.

In order to integrate the modal decomposition method as comprehensive as possible, I make this lib.

Hope my lib can help you.

## Entrance

All entrance of functions or class are stored in `Modal_Decomposition/__init__.py`

## API

There two classification in `__init__.py`: `Class` and `Function`.

- ### Function:

    Use `Function.method` to choose the following mode decomposition method.

- ### Class:

    Use `Class.class` will give you a class. `Class.class.decompose` will decompose the signal.

### Quick Start

```python
from Modal_Decomposition import Function as f
import numpy as np

S = np.random.random(10)

IMFs, Res, Info = f.EMD(S)
```

## Modal Decomposition

| method   | description                                                                 |             use             |                                                                        resource(doi and link)                                                                        |
|----------|:----------------------------------------------------------------------------|:---------------------------:|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
| CEEMDAN  | Complete Ensemble Empirical Mode Decomposition with Adaptive Noise          | `Function.CEEMDAN(siganl)`  |                                         [10.1109/ICASSP.2011.5947265](https://ieeexplore.ieee.org/abstract/document/5947265)                                         |
| CEEFD    | Cyclic Envelop Empirical Fourier Decomposition                              |  `Function.CEEFD(signal)`   |                                    [10.3969/j.issn.1001-4551.2023.07.001](https://d.wanfangdata.com.cn/periodical/jdgc202307001)                                     |
| CEEMD    | Complementary Ensemble Empirical Mode Decomposition                         |  `Function.CEEMD(siganl)`   |                               [10.1016/j.jhydrol.2020.124647](https://www.sciencedirect.com/science/article/abs/pii/S0022169420301074)                               |
| EEMD     | Ensemble Empirical Mode Decomposition                                       |   `Function.EEMD(signal)`   | [10.1142/S1793536909000047](https://www.semanticscholar.org/paper/Ensemble-Empirical-Mode-Decomposition%3A-a-Data-Wu-Huang/a97ee1d4a15c04160c323bd650e9cb9dff9dfced) |
| EFD      | Empirical Fourier Decomposition                                             |   `Function.EFD(signal)`    |                          [10.1016/j.ymssp.2021.108155](https://www.sciencedirect.com/science/article/abs/pii/S0888327021005355?via%3Dihub)                           |
| EMD      | Empirical Mode Decomposition                                                |   `Function.EMD(signal)`    | [10.1098/rspa.1998.0193](https://www.semanticscholar.org/paper/The-empirical-mode-decomposition-and-the-Hilbert-Huang-Shen/3842d81b0375dae8ae92734aa2a5d4aeed7a91d1) |
| EWT      | Empirical Wavelet Transform                                                 |   `Function.EWT(signal)`    |                                                    [10.48550/arXiv.2304.06274](https://arxiv.org/abs/2304.06274)                                                     |
| FMD      | Filtered Mode Decomposition                                                 |   `Function.FMD(signal)`    |                                               [10.1109/TIE.2022.3156156](https://ieeexplore.ieee.org/document/9732251)                                               |
| ICEEMDAN | Improved Complete Ensemble Empirical Mode Decomposition with Adaptive Noise | `Function.ICEEMDAN(signal)` |                                  [10.1007/s10470-021-01901-3](https://link.springer.com/article/10.1007/s10470-021-01901-3#citeas)                                   |
| LMD      | Local Mean Decomposition                                                    |   `Function.LMD(signal)`    |                                       [10.1098/rsif.2005.0058](https://royalsocietypublishing.org/doi/10.1098/rsif.2005.0058)                                        |
| MEMD     | Multivariate Empirical Mode Decomposition                                   |   `Function.MEMD(signal)`   |                                                    [10.48550/arXiv.2206.00926](https://arxiv.org/abs/2206.00926)                                                     |
| RPSEMD   | Random Phase Sinusoidal Assisted Empirical Mode Decomposition               |  `Function.RPSEMD(signal)`  |                                               [10.1109/LSP.2016.2537376](https://ieeexplore.ieee.org/document/7423702)                                               |
| SSA      | Singular Spectrum Analysis                                                  |   `Function.SSA(signal)`    |                                   [10.1016/j.mex.2020.101015](https://www.sciencedirect.com/science/article/pii/S2215016120302351)                                   |
| SVMD     | Successive Variational Mode Decomposition                                   |   `Function.SVMD(signal)`   |                               [10.1016/j.sigpro.2020.107610](https://www.sciencedirect.com/science/article/abs/pii/S0165168420301535)                                |
| VMD      | Variational Mode Decomposition                                              |   `Function.VMD(signal)`    |                                               [10.1109/TSP.2013.2288675](https://ieeexplore.ieee.org/document/6655981)                                               |

## Install

You can install by:
```shell
git clone https://github.com/a-raining-day/Modal-Decomposition.git
cd Motal-Decomposition
pip install -r requirements.txt
```

***Or***:

```shell
pip install Modal-Decomposition
```

> 发布版的平台 wheel (Windows / macOS / Linux) 已内置 Cython 编译的 FHT 加速
> 内核 `_fht_native` (C 源码源自 Smithsonian *am* 项目, 见下方
> "Acknowledgement" 章节), 安装后自动启用; 找不到编译内核时自动回退到纯
> NumPy 实现, 因此任何环境都可以运行。若想在本地源码树手动编译加速内核:
> ```shell
> python setup.py build_ext --inplace   # 仓库根目录
> ```

## Dependence

This lib's dependence are:

***Python: 3.10***

- [EMD-signal](https://github.com/laszukdawid/PyEMD)
- [ewtpy](https://github.com/vrcarva/ewtpy)
- [vmdpy](https://github.com/vrcarva/vmdpy)

*Other dependence please read "requirements.txt"*

*Please pip `EMD-signal`, not `PyEMD`*

## Url

This lib's url is: https://github.com/a-raining-day/Modal-Decomposition

## Acknowledgement

The `FHT` Hilbert backend under
`src/Modal_Decomposition/Utils/_Hilbert/_C/_fht/` contains **third-party C code**:

* Originally written by the **Smithsonian Astrophysical Observatory**,
  Submillimeter Receiver Laboratory (Scott Paine), as part of the *am*
  atmospheric model: <https://www.cfa.harvard.edu/~spaine/am/>
* Acquired from
  [waddafunk/Smithsonians_Discrete_Hilbert_Fourier_Hartley_Transforms](https://github.com/waddafunk/Smithsonians_Discrete_Hilbert_Fourier_Hartley_Transforms)
  (Jacopo Piccirillo, 13/10/2020), where the transform routines were isolated
  for standalone C/C++ use.

Smithsonian *am* license notice:

> This computer program containing an atmospheric propagation model for the
> submillimeter band is a work of the United States and may be used freely,
> with attribution and credit to the Smithsonian Astrophysical Observatory.
> The program is intended for educational, scholarly or research purposes. In
> connection with any commercial use of the program, the user should disclose
> clearly and conspicuously all of the information contained in the first
> sentence of this notice.

**Hilbert-transform phase convention:** the SAO implementation shifts the
phase by **+90°** — *not* **−90°** as in MATLAB/SciPy. To obtain the
MATLAB-compatible result, multiply the component orthogonal to the input by
`exp(j·π) = −1`, i.e. multiply the imaginary part of the analytic signal by
−1 for a real input (`matlab_phase=True` in
`Modal_Decomposition.Utils._Hilbert._fht`).