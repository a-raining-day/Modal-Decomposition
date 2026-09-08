# LMD comparison summary (single run, --repeats 1)

Four columns: LMD-H(scipy) = Hilbert-envelope LMD (scipy analytic signal); LMD-midpoint = shipped extrema-midpoint LMD; LMD-H(FHT) = Hilbert-envelope LMD via the compiled C (SAO FHT) kernel; PySDKit = pysdkit.LMD (classical Smith moving-average). All my variants capped at max_pf=5 (pysdkit default K=5).

## case A

| n | LMD-H(scipy) | LMD-midpoint | LMD-H(FHT) | PySDKit |
|---|-------------:|-------------:|-----------:|--------:|
| 256 | 0.0027 | 0.0390 | 0.0022 | 0.0304 |
| 1024 | 0.0028 | 0.0236 | 0.0024 | 0.0464 |
| 4096 | 0.0042 | 0.0385 | 0.0038 | 0.1447 |
| 16384 | 0.0078 | 0.1927 | 0.0087 | 0.8745 |

Quality (n=16384):

| impl | n_rows | recon max abs | best mode corr |
|------|-------:|--------------:|----------------|
| LMD-H(scipy) | 6 | 1.5543122344752192e-15 | {'tone_37hz': 0.824, 'tone_113hz': 0.8425} |
| LMD-midpoint | 6 | 0.00032452970974006234 | {'tone_37hz': 0.0197, 'tone_113hz': 0.0223} |
| LMD-H(FHT) | 6 | 2.1094237467877974e-15 | {'tone_37hz': 0.824, 'tone_113hz': 0.8425} |
| PySDKit | 6 | 4.440892098500626e-16 | {'tone_37hz': 0.815, 'tone_113hz': 0.7391} |

## case B

| n | LMD-H(scipy) | LMD-midpoint | LMD-H(FHT) | PySDKit |
|---|-------------:|-------------:|-----------:|--------:|
| 256 | 0.0051 | 0.0651 | 0.0039 | 0.0189 |
| 1024 | 0.0071 | 0.0551 | 0.0034 | 0.1586 |
| 4096 | 0.0047 | 0.0967 | 0.0081 | 0.9562 |
| 16384 | 0.0091 | 0.1334 | 0.0091 | 0.5145 |

Quality (n=16384):

| impl | n_rows | recon max abs | best mode corr |
|------|-------:|--------------:|----------------|
| LMD-H(scipy) | 6 | 4.440892098500626e-16 | {'amfm': 0.9986, 'tone_89hz': 0.9866, 'trend': 0.6809} |
| LMD-midpoint | 6 | 0.0003171012203454504 | {'amfm': 0.4418, 'tone_89hz': 0.8853, 'trend': 0.0106} |
| LMD-H(FHT) | 6 | 4.440892098500626e-16 | {'amfm': 0.9986, 'tone_89hz': 0.9866, 'trend': 0.6809} |
| PySDKit | 6 | 4.440892098500626e-16 | {'amfm': 0.9884, 'tone_89hz': 0.9687, 'trend': 0.997} |

## case C

| n | LMD-H(scipy) | LMD-midpoint | LMD-H(FHT) | PySDKit |
|---|-------------:|-------------:|-----------:|--------:|
| 256 | 0.0023 | 0.0368 | 0.0026 | 0.0079 |
| 1024 | 0.0035 | 0.0434 | 0.0032 | 0.0291 |
| 4096 | 0.0066 | 0.1362 | 0.0068 | 0.0823 |
| 16384 | 0.0088 | 0.1683 | 0.0088 | 0.4303 |

Quality (n=16384):

| impl | n_rows | recon max abs | best mode corr |
|------|-------:|--------------:|----------------|
| LMD-H(scipy) | 6 | 1.3322676295501878e-15 | - |
| LMD-midpoint | 6 | 0.0003565299844253411 | - |
| LMD-H(FHT) | 6 | 1.1796119636642288e-15 | - |
| PySDKit | 6 | 8.881784197001252e-16 | - |

## notes

- Hilbert variants use single-shot demodulation per PF (a = |H(h)|, PF = h - m_t); the iterated Hilbert sift does NOT converge on multi-component signals (extrema count 77 -> 201 and growing on case A) - the historical `compute_envelope` in the old LMD was dead code for the same reason.
- Compiled C kernel `_fht_native` is active; its envelope matches scipy to ~1e-14, and standalone microbenchmark at n=16384 shows the C FHT kernel itself is ~1.9x SLOWER per call than scipy's FFT-based hilbert (0.36 vs 0.19 ms); inside LMD the difference is hidden because each PF only needs one Hilbert call.
- PySDKit's moving-average LMD is pure-Python in the inner loops (per-sample extrema staircase + smoothing), so it scales the worst with n.
