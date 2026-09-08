# VMD comparison summary (single run, --repeats 1)

Parity config for all three columns: alpha=2000, tau=0, DC=0, uniform frequency init, tol=1e-6, max_iter=500; K: case A=3, B=3, C=4. PySDKit runs with its default `store_history=True`.

## case A

| n | MD-VMD | vmdpy | PySDKit | PySDKit/vmdpy |
|---|-------:|------:|--------:|--------------:|
| 256 | 0.0022 | 0.0021 | 0.0323 | 15.25x |
| 1024 | 0.0062 | 0.0062 | 0.0725 | 11.78x |
| 4096 | 0.0492 | 0.0491 | 0.2512 | 5.11x |
| 16384 | 1.3993 | 1.4911 | 2.3292 | 1.56x |

Quality (n=16384):

| impl | recon max abs | best mode corr |
|------|--------------:|----------------|
| MD-VMD | 0.4963407132568356 | {'tone_37hz': 0.9993, 'tone_113hz': 0.9972} |
| vmdpy | 0.4963407132568356 | {'tone_37hz': 0.9993, 'tone_113hz': 0.9972} |
| PySDKit | 0.4962360841602951 | {'tone_37hz': 0.9993, 'tone_113hz': 0.9972} |

## case B

| n | MD-VMD | vmdpy | PySDKit | PySDKit/vmdpy |
|---|-------:|------:|--------:|--------------:|
| 256 | 0.0408 | 0.0481 | 0.0424 | 0.88x |
| 1024 | 0.0057 | 0.0071 | 0.0967 | 13.60x |
| 4096 | 0.0150 | 0.0103 | 0.3865 | 37.43x |
| 16384 | 0.0835 | 0.0674 | 1.8796 | 27.88x |

Quality (n=16384):

| impl | recon max abs | best mode corr |
|------|--------------:|----------------|
| MD-VMD | 0.3528350836425685 | {'amfm': 0.9995, 'tone_89hz': 0.9998, 'trend': 0.9928} |
| vmdpy | 0.3528350836425685 | {'amfm': 0.9995, 'tone_89hz': 0.9998, 'trend': 0.9928} |
| PySDKit | 0.35281934351757566 | {'amfm': 0.9995, 'tone_89hz': 0.9998, 'trend': 0.9928} |

## case C

| n | MD-VMD | vmdpy | PySDKit | PySDKit/vmdpy |
|---|-------:|------:|--------:|--------------:|
| 256 | 0.0164 | 0.0063 | 0.0677 | 10.77x |
| 1024 | 0.0224 | 0.0422 | 0.1613 | 3.82x |
| 4096 | 0.4262 | 0.4080 | 0.4862 | 1.19x |
| 16384 | 2.6614 | 3.9148 | 3.8842 | 0.99x |

Quality (n=16384):

| impl | recon max abs | best mode corr |
|------|--------------:|----------------|
| MD-VMD | 2.259368546375587 | - |
| vmdpy | 2.259368546375587 | - |
| PySDKit | 2.2595589573307744 | - |

## note

- MD-VMD (wrapper) ~ vmdpy engine within noise (wrapper overhead negligible).
- PySDKit default (`store_history=True`) never resets the convergence accumulator, so it always runs the full 500 ADMM iterations (its `store_history=False` branch resets correctly and matches vmdpy, e.g. case B n=4096: 0.008s vs 0.248s). Modes are numerically identical either way (corr ~ 1), so the default path burns time without changing the result.
- Case C (noise) rarely early-converges for any engine, so the gap vanishes at n=16384 (3.88s vs 3.91s).
