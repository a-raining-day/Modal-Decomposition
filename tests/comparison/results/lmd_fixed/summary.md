# LMD comparison summary (single run, --repeats 1)

Four columns: LMD-H(scipy) = Hilbert-envelope LMD (scipy analytic signal); LMD-midpoint = shipped extrema-midpoint LMD; LMD-H(FHT) = Hilbert-envelope LMD via the compiled C (SAO FHT) kernel; PySDKit = pysdkit.LMD (classical Smith moving-average). All my variants capped at max_pf=5 (pysdkit default K=5).

## case A

| n | LMD-H(scipy) | LMD-midpoint | LMD-H(FHT) | PySDKit |
|---|-------------:|-------------:|-----------:|--------:|
| 256 | 0.0028 | 0.0314 | 0.0026 | 0.0328 |
| 1024 | 0.0030 | 0.0348 | 0.0034 | 0.0537 |
| 4096 | 0.0043 | 0.0686 | 0.0039 | 0.1544 |
| 16384 | 0.0087 | 0.1201 | 0.0090 | 0.6999 |

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
| 256 | 0.0102 | 0.0975 | 0.0071 | 0.0493 |
| 1024 | 0.1313 | 0.0436 | 0.0439 | 0.4549 |
| 4096 | 0.0107 | 0.1211 | 0.0109 | 1.2345 |
| 16384 | 0.0107 | 0.1250 | 0.0103 | 0.5775 |

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
| 256 | 0.0027 | 0.0412 | 0.0023 | 0.0085 |
| 1024 | 0.0026 | 0.0497 | 0.0026 | 0.0323 |
| 4096 | 0.0040 | 0.0577 | 0.0043 | 0.0990 |
| 16384 | 0.0123 | 0.1751 | 0.0085 | 0.4912 |

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
