# FMD comparison (after K=-1 hang fix)

MD-FMD = Class.FMD(K=3, max_iter=10, num_hand=3, seed=0); PySDKit-FMD = pysdkit.FMD(mode_num=3, fs=1000).

| case | n | MD-FMD | PySDKit-FMD | ratio |
|---|---|-------:|------------:|------:|
| A | 256 | 0.464 | 0.237 | 2.0x |
| A | 1024 | 8.704 | 0.636 | 13.7x |
| A | 4096 | 30.011 | 1.872 | 16.0x |
| B | 256 | 0.429 | 0.234 | 1.8x |
| B | 1024 | 5.421 | 0.587 | 9.2x |
| B | 4096 | 23.336 | 1.893 | 12.3x |
| C | 256 | 0.447 | 0.229 | 1.9x |
| C | 1024 | 1.417 | 0.580 | 2.4x |
| C | 4096 | 5.695 | 1.884 | 3.0x |

PySDKit-FMD @16384: A=1.79s, B=9.46s (MD-FMD SVD-based does not scale there).

Quality (n=4096): MD-FMD tone corr 0.41/0.21 vs PySDKit 0.87/0.44; trend 0.16 vs 1.00.
