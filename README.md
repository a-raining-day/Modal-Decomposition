# Modal-Decomposition

> A unified library of modal decomposition methods — 15 classical and modern
> signal-decomposition algorithms behind one consistent API.

There are many methods of modal decomposition, but there is no library that
covers them all with a single, coherent interface. **Modal-Decomposition**
integrates the most commonly used decomposition algorithms into one package:
every method shares the same class contract, the same return type, and the
same validation / seeding / memory conventions, so switching between
algorithms (or benchmarking them against each other) is a one-line change.

## Methods

| Method   | Description                                                              | Use                        | Reference (DOI) |
|----------|--------------------------------------------------------------------------|----------------------------|-----------------|
| CEEMDAN  | Complete Ensemble Empirical Mode Decomposition with Adaptive Noise       | `Function.CEEMDAN(S)`      | [10.1109/ICASSP.2011.5947265](https://ieeexplore.ieee.org/abstract/document/5947265) |
| CEEFD    | Cyclic Envelop Empirical Fourier Decomposition                           | `Function.CEEFD(S)`        | [10.3969/j.issn.1001-4551.2023.07.001](https://d.wanfangdata.com.cn/periodical/jdgc202307001) |
| CEEMD    | Complementary Ensemble Empirical Mode Decomposition                      | `Function.CEEMD(S)`        | [10.1016/j.jhydrol.2020.124647](https://www.sciencedirect.com/science/article/abs/pii/S0022169420301074) |
| EEMD     | Ensemble Empirical Mode Decomposition                                    | `Function.EEMD(S)`         | [10.1142/S1793536909000047](https://www.semanticscholar.org/paper/Ensemble-Empirical-Mode-Decomposition%3A-a-Data-Wu-Huang/a97ee1d4a15c04160c323bd650e9cb9dff9dfced) |
| EFD      | Empirical Fourier Decomposition                                          | `Function.EFD(S)`          | [10.1016/j.ymssp.2021.108155](https://www.sciencedirect.com/science/article/abs/pii/S0888327021005355?via%3Dihub) |
| EMD      | Empirical Mode Decomposition                                             | `Function.EMD(S)`          | [10.1098/rspa.1998.0193](https://www.semanticscholar.org/paper/The-empirical-mode-decomposition-and-the-Hilbert-Huang-Shen/3842d81b0375dae8ae92734aa2a5d4aeed7a91d1) |
| EWT      | Empirical Wavelet Transform                                              | `Function.EWT(S)`          | [10.48550/arXiv.2304.06274](https://arxiv.org/abs/2304.06274) |
| FMD      | Filtered Mode Decomposition                                              | `Function.FMD(S)`          | [10.1109/TIE.2022.3156156](https://ieeexplore.ieee.org/document/9732251) |
| ICEEMDAN | Improved Complete Ensemble Empirical Mode Decomposition with Adaptive Noise | `Function.ICEEMDAN(S)`  | [10.1007/s10470-021-01901-3](https://link.springer.com/article/10.1007/s10470-021-01901-3#citeas) |
| LMD      | Local Mean Decomposition                                                  | `Function.LMD(S)`          | [10.1098/rsif.2005.0058](https://royalsocietypublishing.org/doi/10.1098/rsif.2005.0058) |
| MEMD     | Multivariate Empirical Mode Decomposition                                | `Function.MEMD(S)`         | [10.48550/arXiv.2206.00926](https://arxiv.org/abs/2206.00926) |
| RPSEMD   | Random Phase Sinusoidal Assisted Empirical Mode Decomposition            | `Function.RPSEMD(S)`       | [10.1109/LSP.2016.2537376](https://ieeexplore.ieee.org/document/7423702) |
| SSA      | Singular Spectrum Analysis                                               | `Function.SSA(S)`          | [10.1016/j.mex.2020.101015](https://www.sciencedirect.com/science/article/pii/S2215016120302351) |
| SVMD     | Successive Variational Mode Decomposition                                | `Function.SVMD(S)`         | [10.1016/j.sigpro.2020.107610](https://www.sciencedirect.com/science/article/abs/pii/S0165168420301535) |
| VMD      | Variational Mode Decomposition                                           | `Function.VMD(S)`          | [10.1109/TSP.2013.2288675](https://ieeexplore.ieee.org/document/6655981) |

## Install

```shell
pip install Modal-Decomposition
```

Or install from source:

```shell
git clone https://github.com/a-raining-day/Modal-Decomposition.git
cd Modal-Decomposition
pip install -r requirements.txt      # pinned build/dev environment
pip install -e .                     # editable install (compiles the FHT kernel)
```

The published platform wheels (Windows / macOS / Linux) ship the optional
Cython-compiled FHT acceleration kernel `_fht_native` (C source from the
Smithsonian *am* project, see [Acknowledgement](#acknowledgement)); it is used
automatically when present and the library falls back to the pure NumPy
implementation otherwise, so it runs in any environment. To compile the
kernel manually inside a source tree:

```shell
python setup.py build_ext --inplace   # repository root
```

Requires **Python >= 3.10**.

### Dependencies

| Package     | Used by / for                                       |
|-------------|-----------------------------------------------------|
| numpy       | core arrays and vectorized kernels                 |
| scipy       | splines, envelope / Hilbert, filtering, peaks, FFT |
| EMD-signal (PyEMD) | EMD, EEMD, CEEMDAN (and ensemble chains)  |
| ewtpy       | EWT                                               |
| vmdpy       | VMD                                               |
| psutil      | available-memory reading for the memmap policy     |

> Please install `EMD-signal`, not `PyEMD` (the latter is an unrelated older package).

Optional extras (not required at runtime): `[dev]` (pytest, black) and
`[plot]` (matplotlib). `numba` may be installed to activate the optional
`"numba"` backend of `Utils.Peaks` / `SVMD`; everything degrades gracefully
without it.

## Quick Start

```python
import numpy as np
from Modal_Decomposition import Function as f

fs = 1000
t = np.linspace(0, 1, 1000, endpoint=False)
S = np.sin(2 * np.pi * 37 * t) + 0.5 * np.sin(2 * np.pi * 180 * t)

IMFs, Res, Info = f.EMD(S)          # returns a DecompositionResult tuple
```

### API — two entrances

All entrances live in `Modal_Decomposition/__init__.py`, exposed as two
read-only namespaces plus a small set of global helpers:

- **`Class`** — decomposer classes. Construct with parameters, then call
  `.decompose(S, T=None)` (also available as `__call__`):

  ```python
  from Modal_Decomposition import Class

  r = Class.VMD(alpha=3000, K=4).decompose(S)
  ```

- **`Function`** — function facades, strictly equivalent to the class form
  (`Function.X(S, T=None, **params)` = `Class.X(**params).decompose(S, T)`):

  ```python
  r = f.LMD(S)          # r is a DecompositionResult, not a tuple
  ```

- **Global helpers** — `set_seed(seed)` / `get_seed()`,
  `set_absolute_limit(bytes)` / `set_memmap_ratio(ratio)` (see
  [Memory policy](#memory-policy-for-very-large-signals)).

### Return contract

Every method returns a **`DecompositionResult`** (also unpackable as a
3-tuple `IMFs, Res, info`):

| Field    | Meaning                                                            |
|----------|--------------------------------------------------------------------|
| `.IMFs`  | decomposed modes, shape `(K, N)` (univariate) or `(K, d, N)` (MEMD)|
| `.Res`   | residual, or `None` for methods without a residual concept (SSA, VMD) |
| `.info`  | method-specific diagnostics dict (e.g. FFT spectra, boundaries)     |
| `.config`| frozen dataclass snapshot of the *effective* parameters of this run |

Plus helpers: `.n_imfs`, `.shape` and `.reconstruct()` (`sum(IMFs) + Res`).

```python
r = f.CEEMDAN(S, trials=30, seed=0)
print(r.n_imfs, r.reconstruct().shape)
print(r.config)                 # effective parameter snapshot
```

### Random seed

Methods that use randomness accept a local `seed` parameter. A process-level
global seed overrides every local seed (with a `UserWarning`) — call
`Modal_Decomposition.set_seed(n)` once at the start of a script to make all
downstream runs reproducible.

### Time axis

`T` is optional; when omitted an index axis is used. Pass a non-uniformly
sampled axis when the method needs physical frequencies (e.g. `fs`-related
parameters); duplicate or descending axes are validated (descending axes are
reordered with a warning).

## Memory policy for very large signals

Large-input support is built into the input layer
(`Utils.Check_Time_and_Signal`): inputs above the policy threshold are served
from disk-backed `np.memmap` working copies instead of being materialized in
RAM, so a machine can process signals far larger than its physical memory.

- `set_memmap_ratio(ratio)` — memmap when the projected usage reaches
  `ratio` (default 0.6) of the remaining available memory;
- `set_absolute_limit(n_bytes)` — memmap above a fixed byte limit (default 2 GiB).

The two strategies are mutually exclusive; the latest call wins. Monotonicity
checks and chunked utilities (`Utils.Chunk`) follow the same policy.

## Repository layout

```
src/Modal_Decomposition/
├── __init__.py      Class / Function namespaces + global API
├── <METHOD>.py      1 Config dataclass + 1 Decomposer per method
├── EMD.py           native EMD engine (PyEMD-free; original EMD_new,
│                    registered as the public "EMD")
├── _Registry.py     class registry (registration at import time)
├── Base/            Decomposer ABC, DecompositionResult, Config,
│                    import Cache, metadata tables, size constants
└── Utils/           Check / Chunk / Peaks / Mirror / Spline / Envelope /
                     Hilbert / Monotonicity / Memory / Seed
                     (+ _Hilbert FHT backends)
tests/               pytest suite + benchmarking harnesses (comparison/,
                     ssa/, test_memory/) against PyEMD and PySDKit
```

## Changelog

### Unreleased (0.3.0)

- `EMD` is now the native self-implemented sifting engine (ex-`EMD_new`,
  registered as the public `EMD`; the former PyEMD wrapper was removed) —
  faster than PyEMD at equal mode quality, cold start ~ms vs ~0.5 s
  (`docs/EMD_vs_EMD_new_Performance_Report.md`).
- `EMD(faster=...)` two-branch stopping policy: `faster=False` (default,
  quality branch) additionally requires the classic narrowband balance
  `|zc - ext| <= 1` before accepting each IMF (clean signals cost nothing;
  noisy / long signals trade speed for row-level purity comparable with
  PyEMD); `faster=True` keeps the legacy fast branch (energy Cauchy SD
  stops sifting). Full four-way comparison (MD-quality / MD-fast / PyEMD /
  PySDKit) in `docs/EMD_faster_Branch_Comparison_Report.md`. Speed ratios
  are always quoted same-process/same-script (median timing in the bench
  scripts), never cross-day absolute timings.
- `EMD` defaults (CubicSpline envelope + `sd_thr=0.01` + `faster=False`)
  chosen by the mode-level validation (`docs/EMD_Validation_and_Comparison_Report.md`)
  and the mechanism/optimization analysis (`docs/EMD_Quality_Gap_and_Optimization.md`);
  `spline_kind` accepts the canonical `Utils.Spline` kinds.
- LMD defaults to the extrema (Spline) interpolation envelope; the analytic
  Hilbert envelope stays as an explicit opt-in only. The Hilbert-vs-Spline
  envelope concepts are separated again (`Utils.Envelope` restored as its own
  module).
- New `Utils.Mirror` (endpoint mirror extension, EMD nbsym + LMD boundary
  modes) shared by EMD / LMD / future consumers.

### 0.2.0

- Unified utility layer (`Utils.Chunk` / `Peaks` / `Check` / `Cache` /
  `Memory` / `Spline` / `Envelope`) with documented conventions and speed
  reports under `docs/`.
- Memory-bounded processing for very large signals: policy-driven `memmap`
  input layer, adaptive chunking, dtype-preserving pipelines.
- Optional C-accelerated FHT/Hilbert kernel (`_fht_native`, compiled wheels
  for Windows / macOS / Linux, cp310–cp313) with automatic pure-NumPy
  fallback; CI verifies native-vs-pure numerical identity.
- `EMD_new`: experimental self-implemented EMD engine (prototype of the
  PyEMD-free roadmap).
- Dependency cleanup: removed unused packages (`calorine`, `tqdm` and stale
  pinned entries); unified result `config` snapshots and seed handling.

## Acknowledgement

The `FHT` Hilbert backend under
`src/Modal_Decomposition/Utils/_Hilbert/_C/_fht/` contains **third-party C
code**:

- Originally written by the **Smithsonian Astrophysical Observatory**,
  Submillimeter Receiver Laboratory (Scott Paine), as part of the *am*
  atmospheric model: <https://www.cfa.harvard.edu/~spaine/am/>
- Acquired from
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

## License

Apache-2.0 — see [LICENSE](LICENSE).

## URL

Project homepage / repository:
<https://github.com/a-raining-day/Modal-Decomposition>
