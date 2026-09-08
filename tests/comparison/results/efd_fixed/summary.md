# EFD comparison (after max_IMFs contract fix)

MD-EFD = Class.EFD(max_IMFs=3); PySDKit-EFD = pysdkit.EFD(max_imfs=3). Both ~ms (FFT-based).

| case | n | MD-EFD | PySDKit-EFD |
|---|---|-------:|------------:|
| A | 256 | 0.0018 | 0.0007 |
| A | 1024 | 0.0019 | 0.0017 |
| A | 4096 | 0.0048 | 0.0048 |
| A | 16384 | 0.0179 | 0.0093 |
| B | 256 | 0.0010 | 0.0007 |
| B | 1024 | 0.0015 | 0.0015 |
| B | 4096 | 0.0036 | 0.0023 |
| B | 16384 | 0.0147 | 0.0074 |
| C | 256 | 0.0009 | 0.0012 |
| C | 1024 | 0.0016 | 0.0009 |
| C | 4096 | 0.0043 | 0.0026 |
| C | 16384 | 0.0176 | 0.0108 |

Quality (n=16384): case A tones: PySDKit 0.998/0.994 vs MD 0.888/0.443; case B amfm: MD 0.979 vs PySDKit 0.872; trend: PySDKit 0.989 vs MD 0.201.
