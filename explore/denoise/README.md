# Denoising benchmark for `NoisyEnvelopeMeanIteration`

This directory contains the experiment code for the denoising study of the
experimental `NoisyEnvelopeMeanIteration` (with the M-realisation average).
The conclusions and figures are in
`docs/explore/denoise/Denoise_Performance_Report.md`.

## Layout

| file | role |
|---|---|
| `signals.py` | seven synthetic cases (two tones, trend+tone, AM-FM, impacts, steps, spikes, pure noise) and noise injection |
| `metrics.py` | generic metrics (RMSE, SNR, correlation, edge error, band energy, LSD) and case-specific metrics (trend, impact envelope, step edge, spike leakage) |
| `methods.py` | classical baselines, oracle low-pass, EEMD-corr reference, proposed-method grid, and the wrapper that seeds `np.random` |
| `run_benchmark.py` | main denoising runner; writes raw records to `results/benchmark.json` |
| `analyze_results.py` | aggregates raw records to `summary.csv`, `ranking.csv`, `summary.json`, and optional Markdown tables |
| `make_figures.py` | regenerates the denoising figures under `docs/explore/denoise/figures/` |
| `morphology.py` | 1-D morphological opening/closing/open-close-average filters |
| `run_high_noise.py` | focused high-noise main-morphology runner (morphology vs SG vs Butter vs proposed) |
| `analyze_high_noise.py` | aggregates high-noise records; writes `high_noise_summary.csv` / `high_noise_ranking.csv` |
| `make_high_noise_figures.py` | high-noise rank/heatmap/example figures |
| `run_noise_amp.py` | high-noise auxiliary-noise amplitude sweep (`noise_amp` 0.1-1.2) |
| `analyze_noise_amp.py` | aggregates the amplitude sweep; writes `high_noise_noise_amp_*.csv/json` |
| `make_noise_amp_figures.py` | shape/SNR vs amplitude and per-case figures |
| `run_joint_grid.py` | joint K x M x amp grid + per-shape oracle baselines |
| `analyze_joint_grid.py` | global/LOSO/per-shape oracle cost analysis + P0 CI |
| `make_joint_grid_figures.py` | joint-grid heatmap, cost, marginal figures |
| `run_affine_scale.py` | P1: affine scale `a` distribution for 3 shapes/config |
| `make_affine_scale_figure.py` | P1: `a` distribution figure |
| `run_real_eeg.py` | P2: real EEG sample + synthetic high noise demo |
| `analyze_real_eeg.py` | P2: real EEG summary |
| `make_real_eeg_figure.py` | P2: real EEG example figure |
| `run_cwru.py` | P2: real CWRU bearing fault segment, envelope SNR |
| `make_cwru_figure.py` | P2: CWRU envelope-spectrum figure |
| `run_reliability.py` | peak / realization-std reliability experiment (CWRU + synthetic impact) |
| `analyze_reliability.py` | aggregates reliability R vs SNR |
| `make_reliability_figure.py` | CWRU mean±std and reliability-ratio figure |
| `run_reliability_amp.py` | CWRU reliability vs auxiliary-noise amplitude (0.01-0.5) |
| `analyze_reliability_amp.py` | aggregates the amp sweep |
| `make_reliability_amp_figure.py` | R/SNR vs auxiliary-noise amplitude figure |
| `data/cwru/IR007_105_0.mat` | real CWRU IR007, 1797 rpm, 12 kHz drive-end (local copy; see data/cwru/README.md) |
| `image_2d/run_image_demo.py` | 2-D morphological analogue on a real image |
| `image_2d/make_figure.py` | 2-D image demo figure |

## Quick start

From the repository root, with the project virtual environment:

```powershell
# full main grid + EEMD reference on a small subset + noise_amp sensitivity
.venv\Scripts\python.exe -B -m explore.denoise.run_benchmark `
    --reps 5 --include-eemd --include-sensitivity

# aggregate
.venv\Scripts\python.exe -B -m explore.denoise.analyze_results --markdown

# figures
.venv\Scripts\python.exe -B -m explore.denoise.make_figures

# high-noise main-morphology comparison: morph OC/CO/avg + SG + Butter + proposed
.venv\Scripts\python.exe -B -m explore.denoise.run_high_noise --reps 5
.venv\Scripts\python.exe -B -m explore.denoise.analyze_high_noise --markdown
.venv\Scripts\python.exe -B -m explore.denoise.make_high_noise_figures

# auxiliary-noise amplitude sweep at high noise
.venv\Scripts\python.exe -B -m explore.denoise.run_noise_amp --reps 5
.venv\Scripts\python.exe -B -m explore.denoise.analyze_noise_amp --markdown
.venv\Scripts\python.exe -B -m explore.denoise.make_noise_amp_figures

# joint K x M x amp grid + per-shape oracle
.venv\Scripts\python.exe -B -m explore.denoise.run_joint_grid --reps 5
.venv\Scripts\python.exe -B -m explore.denoise.analyze_joint_grid --markdown
.venv\Scripts\python.exe -B -m explore.denoise.make_joint_grid_figures

# P1: affine scale a
.venv\Scripts\python.exe -B -m explore.denoise.run_affine_scale
.venv\Scripts\python.exe -B -m explore.denoise.make_affine_scale_figure

# P2: real EEG sanity check
.venv\Scripts\python.exe -B -m explore.denoise.run_real_eeg --reps 10
.venv\Scripts\python.exe -B -m explore.denoise.analyze_real_eeg
.venv\Scripts\python.exe -B -m explore.denoise.make_real_eeg_figure

# P2: real CWRU bearing envelope check
.venv\Scripts\python.exe -B -m explore.denoise.run_cwru
.venv\Scripts\python.exe -B -m explore.denoise.make_cwru_figure

# reliability score: peak / realization std
.venv\Scripts\python.exe -B -m explore.denoise.run_reliability
.venv\Scripts\python.exe -B -m explore.denoise.analyze_reliability
.venv\Scripts\python.exe -B -m explore.denoise.make_reliability_figure

# reliability vs auxiliary-noise amplitude
.venv\Scripts\python.exe -B -m explore.denoise.run_reliability_amp
.venv\Scripts\python.exe -B -m explore.denoise.analyze_reliability_amp
.venv\Scripts\python.exe -B -m explore.denoise.make_reliability_amp_figure

# 2-D image analogue
.venv\Scripts\python.exe -B -m explore.denoise.image_2d.run_image_demo
.venv\Scripts\python.exe -B -m explore.denoise.image_2d.make_figure
```

Smoke test:

```powershell
.venv\Scripts\python.exe -B -m explore.denoise.run_benchmark `
    --cases two_tone trend_tone --noise-levels 0.3 --reps 1 `
    --out explore/denoise/results/smoke.json
```

## Design

* **Cases** (7): `two_tone`, `trend_tone`, `am_fm`, `impact`, `step`,
  `spike_tone`, `pure_noise`. All are `float64`, `fs = 1000 Hz`,
  `n = 2048` unless `--n` is changed.
* **Noise levels** (ordinary cases): `0.1, 0.3, 0.6` times the clean signal's
  standard deviation. `pure_noise` uses a unit-variance absolute noise level.
* **Repetitions**: 5 independent noise realisations per case/level. For a
  fixed observation, every method receives the same noisy array; stochastic
  methods get one fixed RNG seed per repetition.
* **Proposed grid**: `K in {1, 2, 3}` × `M in {1, 32}` at
  `noise_amp = 0.2` (six configurations). The auxiliary-noise sensitivity
  varies `noise_amp in {0.05, 0.1, 0.3, 0.5}` at `K=2, M=32` on
  `trend_tone` and `impact`.
* **Baselines**: raw, MA(5), Savitzky-Golay(11, 3), median(5),
  Butterworth low-pass (4th order, 150 Hz, zero-phase), Wiener(11).
  These are fixed, representative settings, not tuned per case.
* **References**: `IdealLP` is an oracle zero-phase low-pass at the 99%
  clean-energy bandwidth; it uses the clean spectrum by construction.
  `EEMD-corr` is a correlation-thresholded EEMD reconstruction, run only on
  `two_tone`, `trend_tone`, and `impact` at noise `0.3` (five repetitions).

## High-noise main-morphology experiment

`run_high_noise.py` is a focused follow-up for the clarified role of the
method: **high-noise main-shape extraction, not general denoising**.  It uses
noise levels `0.5/0.7/0.9 x std`, and compares:

* `MorphOC5/11/21` (opening -> closing), `MorphCO5/11/21` (closing -> opening),
  `MorphAvg5/11/21` (their average), flat SE with reflect boundaries;
* `SG11`, `SG21`; `Butter100`, `Butter150`, `Butter250`;
* `P-K1-M32`, `P-K1-M64`, `P-K2-M32`, `P-K2-M64`;
* `raw` and the oracle `IdealLP`.

The primary metric is **shape NRMSE after best affine rescaling** (amplitude
attenuation is allowed); SNR and feature metrics are secondary.  See
`docs/explore/denoise/High_Noise_Morphology_Report.md`.

`run_noise_amp.py` additionally sweeps `noise_amp` from 0.1 to 1.2 at high
noise.  Result summary: `noise_amp=0.5-0.6` is better than the old fixed 0.2
overall, but the per-shape optimum differs; see
`docs/explore/denoise/Noise_Amplitude_Matching_Report.md`.

## Outputs

* `results/benchmark.json` — one record per
  `(case, noise_level, repetition, method)` with metrics and runtime.
* `results/summary.csv` — median metrics per `(case, noise_level, method)`.
* `results/ranking.csv` — median relative RMSE, SNR improvement, edge max and
  runtime per method (over all observations).
* `results/summary.json` — the same summary/ranking in JSON.
* `results/high_noise_morph.json` — high-noise comparison records.
* `results/high_noise_summary.csv` / `high_noise_ranking.csv` — high-noise
  median metrics and cross-shape ranking.
* `results/high_noise_noise_amp.json` — auxiliary-noise amplitude sweep.
* `results/high_noise_noise_amp_summary.csv` / `..._ranking.csv` — amplitude
  sweep medians and rankings.
* `results/joint_grid.json` — joint K x M x amp grid + baseline oracle data.
* `results/joint_grid_configs.csv` / `joint_grid_cases.csv` — per-config
  rankings and per-shape oracle/LOSO cost + top-3 CI.
* `results/affine_scale.json/csv` — P1 affine scale distribution.
* `results/real_eeg.json` / `real_eeg_summary.csv` — P2 real EEG demo.
* `results/cwru.json` / `cwru_summary.csv` — P2 real CWRU bearing envelope.
* `results/reliability.json` / `reliability_summary.csv` — peak/std reliability experiment.
* `results/reliability_amp.json` / `reliability_amp_summary.csv` — amp sweep.
* `results/image_2d.json` / `image_2d_summary.csv` — 2-D image demo.
* `docs/explore/denoise/figures/*.png` — example waveforms, SNR vs noise,
  K/M sensitivity, impact envelope spectrum, high-noise rank/heatmap,
  amplitude-sweep figures, joint-grid heatmap/cost, affine-a distribution,
  real-EEG example, CWRU envelope spectrum, 2-D image example, and
  2-D mean-surface/subtraction comparison.

## Notes

* The reported runtime is a single-call wall time measured in the same
  process; it includes Python overhead and is not a controlled benchmark.
* `NoisyEnvelopeMeanIteration` still has the boundary/extrema issues
  described in the report; the M average reduces random variance but does not
  fix them. The benchmark keeps those behaviours because the experiment is
  meant to measure the method as implemented.
* The `IdealLP` row is an oracle and must not be read as a deployable method.
