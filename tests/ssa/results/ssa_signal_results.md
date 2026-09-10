# SSA on multi-tone / wideband / non-stationary signals

N=2048, fs=1000.0 Hz, full decomposition; measured by test_ssa_signal.py::test_ssa_signal_results_recorded.

| case | stride | n_comp | recon max abs | first-RC metric | wall (s) |
|------|-------:|-------:|--------------:|-----------------|---------:|
| multitone | 1 | 682 | 1.31e-14 | corr 0.9965 | 8.4453 |
| multitone | 4 | 343 | 6.38e-15 | corr 0.9957 | 0.9293 |
| multitone | 16 | 87 | 3.55e-15 | corr 0.9834 | 0.0560 |
| wideband | 1 | 682 | 5.77e-15 | energy 0.0028 | 7.0621 |
| wideband | 4 | 343 | 4.96e-15 | energy 0.0038 | 0.9805 |
| wideband | 16 | 87 | 5.77e-15 | energy 0.0059 | 0.0582 |
| nonstationary | 1 | 682 | 5.77e-15 | corr 0.9517 | 7.3971 |
| nonstationary | 4 | 343 | 8.55e-15 | corr 0.9525 | 0.9815 |
| nonstationary | 16 | 87 | 7.55e-15 | corr 0.9339 | 0.0521 |

Notes: reconstruction is exact (~1e-14) for every case/stride; multitone tones are recovered by grouped sine/cosine pairs (corr > 0.95); the wideband (white-noise) case spreads energy across components; the non-stationary case (37 -> 90 Hz jump at mid-signal) is separated into two pairs tracking each segment (corr > 0.9).
