# LMD comparison summary (single run, --repeats 1)

Four columns: LMD-H(scipy) = Hilbert-envelope LMD (scipy analytic signal); LMD-midpoint = shipped extrema-midpoint LMD; LMD-H(FHT) = Hilbert-envelope LMD via the compiled C (SAO FHT) kernel; PySDKit = pysdkit.LMD (classical Smith moving-average). All my variants capped at max_pf=5 (pysdkit default K=5).

## case A

| n | LMD-H(scipy) | LMD-midpoint | LMD-H(FHT) | PySDKit |
|---|-------------:|-------------:|-----------:|--------:|
| 256 | 0.0038 | 0.0338 | 0.0024 | 0.0334 |
| 1024 | 0.0029 | 0.0362 | 0.0030 | 0.0566 |
| 4096 | 0.0036 | 0.1710 | 0.0142 | 0.4746 |
| 16384 | 0.0208 | 0.4997 | 0.0320 | 1.4686 |

Quality (n=16384):

| impl | n_rows | recon max abs | best mode corr |
|------|-------:|--------------:|----------------|
| LMD-H(scipy) | 6 | 1.5543122344752192e-15 | {'tone_37hz': 0.824, 'tone_113hz': 0.8425} |
| LMD-midpoint | 6 | 3.302524920201222e-12 | {'tone_37hz': 0.0056, 'tone_113hz': 0.0123} |
| LMD-H(FHT) | 6 | 2.1094237467877974e-15 | {'tone_37hz': 0.824, 'tone_113hz': 0.8425} |
| PySDKit | 6 | 4.440892098500626e-16 | {'tone_37hz': 0.815, 'tone_113hz': 0.7391} |

## case B

| n | LMD-H(scipy) | LMD-midpoint | LMD-H(FHT) | PySDKit |
|---|-------------:|-------------:|-----------:|--------:|
| 256 | 0.0061 | 0.1019 | 0.0101 | 0.0630 |
| 1024 | 0.0068 | 0.0821 | 0.0105 | 0.4005 |
| 4096 | 0.0129 | 0.1830 | 0.0103 | 2.0441 |
| 16384 | 0.0338 | 0.3448 | 0.0181 | 1.4779 |

Quality (n=16384):

| impl | n_rows | recon max abs | best mode corr |
|------|-------:|--------------:|----------------|
| LMD-H(scipy) | 6 | 4.440892098500626e-16 | {'amfm': 0.9986, 'tone_89hz': 0.9866, 'trend': 0.6809} |
| LMD-midpoint | 6 | 4.440892098500626e-16 | {'amfm': 0.9937, 'tone_89hz': 0.9413, 'trend': 0.9024} |
| LMD-H(FHT) | 6 | 4.440892098500626e-16 | {'amfm': 0.9986, 'tone_89hz': 0.9866, 'trend': 0.6809} |
| PySDKit | 6 | 4.440892098500626e-16 | {'amfm': 0.9884, 'tone_89hz': 0.9687, 'trend': 0.997} |

## case C

| n | LMD-H(scipy) | LMD-midpoint | LMD-H(FHT) | PySDKit |
|---|-------------:|-------------:|-----------:|--------:|
| 256 | 0.0104 | 0.1575 | 0.0104 | 0.0369 |
| 1024 | 0.0126 | 0.1779 | 0.0094 | 0.1007 |
| 4096 | 0.0158 | 0.2488 | 0.0199 | 0.2667 |
| 16384 | 0.0195 | 0.5788 | 0.0208 | 1.2085 |

Quality (n=16384):

| impl | n_rows | recon max abs | best mode corr |
|------|-------:|--------------:|----------------|
| LMD-H(scipy) | 6 | 1.3322676295501878e-15 | - |
| LMD-midpoint | 6 | 6.27697893662571e-12 | - |
| LMD-H(FHT) | 6 | 1.1796119636642288e-15 | - |
| PySDKit | 6 | 8.881784197001252e-16 | - |

## notes

- Hilbert variants use single-shot demodulation per PF (a = |H(h)|, PF = h - m_t); the iterated Hilbert sift does NOT converge on multi-component signals (extrema count 77 -> 201 and growing on case A) - the historical `compute_envelope` in the old LMD was dead code for the same reason.
- Compiled C kernel `_fht_native` is active; its envelope matches scipy to ~1e-14, and standalone microbenchmark at n=16384 shows the C FHT kernel itself is ~1.9x SLOWER per call than scipy's FFT-based hilbert (0.36 vs 0.19 ms); inside LMD the difference is hidden because each PF only needs one Hilbert call.
- PySDKit's moving-average LMD is pure-Python in the inner loops (per-sample extrema staircase + smoothing), so it scales the worst with n.
