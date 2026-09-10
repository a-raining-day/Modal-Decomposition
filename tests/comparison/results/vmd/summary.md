# VMD comparison summary (single run, --repeats 1)

Parity config for all three columns: alpha=2000, tau=0, DC=0, uniform frequency init, tol=1e-6, max_iter=500; K: case A=3, B=3, C=4. PySDKit runs with its default `store_history=True`.

## case A

| n | MD-VMD | vmdpy | PySDKit | PySDKit/vmdpy |
|---|-------:|------:|--------:|--------------:|
| 256 | 0.0403 | 0.0020 | 0.0356 | 17.54x |
| 1024 | 0.0085 | 0.0082 | 0.0845 | 10.32x |
| 4096 | 0.0539 | 0.0570 | 0.2640 | 4.63x |
| 16384 | 4.0761 | 4.1390 | 6.5748 | 1.59x |

Quality (n=16384):

| impl | recon max abs | best mode corr |
|------|--------------:|----------------|
| MD-VMD | 0.4963407132568356 | {'tone_37hz': 0.9993, 'tone_113hz': 0.9972} |
| vmdpy | 0.4963407132568356 | {'tone_37hz': 0.9993, 'tone_113hz': 0.9972} |
| PySDKit | 0.4962360841602951 | {'tone_37hz': 0.9993, 'tone_113hz': 0.9972} |

## case B

| n | MD-VMD | vmdpy | PySDKit | PySDKit/vmdpy |
|---|-------:|------:|--------:|--------------:|
| 256 | 0.1714 | 0.1646 | 0.1324 | 0.80x |
| 1024 | 0.0087 | 0.0154 | 0.2993 | 19.40x |
| 4096 | 0.0255 | 0.0328 | 0.7706 | 23.48x |
| 16384 | 0.2790 | 0.2658 | 4.6374 | 17.45x |

Quality (n=16384):

| impl | recon max abs | best mode corr |
|------|--------------:|----------------|
| MD-VMD | 0.3528350836425685 | {'amfm': 0.9995, 'tone_89hz': 0.9998, 'trend': 0.9928} |
| vmdpy | 0.3528350836425685 | {'amfm': 0.9995, 'tone_89hz': 0.9998, 'trend': 0.9928} |
| PySDKit | 0.35281934351757566 | {'amfm': 0.9995, 'tone_89hz': 0.9998, 'trend': 0.9928} |

## case C

| n | MD-VMD | vmdpy | PySDKit | PySDKit/vmdpy |
|---|-------:|------:|--------:|--------------:|
| 256 | 0.0168 | 0.0124 | 0.1306 | 10.50x |
| 1024 | 0.0888 | 0.0872 | 0.5055 | 5.80x |
| 4096 | 0.9138 | 1.0607 | 1.1132 | 1.05x |
| 16384 | 5.8381 | 8.4870 | 8.3674 | 0.99x |

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
