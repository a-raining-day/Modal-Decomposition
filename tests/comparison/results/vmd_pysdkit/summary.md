# VMD three-way summary (MD-VMD / vmdpy / PySDKit)

Parity config for all three columns: `alpha=2000`, `tau=0`, `DC=False`, uniform frequency init, `tol=1e-6`, `max_iter=500`; K: case A=3, B=3, C=4. Values are the **median of 3 repeats** (same process, warm-up excluded).

## case A

| n | MD-VMD (s) | vmdpy (s) | PySDKit (s) | vmdpy/MD | PySDKit/MD | PySDKit/vmdpy |
|---:|---:|---:|---:|---:|---:|---:|
| 256 | 0.0015 | 0.0019 | 0.0330 | 1.3x | 22.3x | 17.4x |
| 1024 | 0.0024 | 0.0066 | 0.0782 | 2.7x | 32.0x | 11.9x |
| 4096 | 0.0114 | 0.0529 | 0.2523 | 4.7x | 22.2x | 4.8x |
| 16384 | 0.7295 | 3.1071 | 4.3707 | 4.3x | 6.0x | 1.4x |
| 65536 | 2.5232 | 19.5730 | 34.6092 | 7.8x | 13.7x | 1.8x |

Quality:

| impl | rows | recon max abs err | orthogonality idx | mode recovery (best abs corr) |
|---|---:|---:|---:|---|
| MD-VMD (n=65536) | 4 | 2.78e-17 | 4.52e-03 | tone_37hz=0.9993, tone_113hz=0.9973 |
| vmdpy (n=65536) | 4 | 2.78e-17 | 4.52e-03 | tone_37hz=0.9993, tone_113hz=0.9973 |
| PySDKit (n=65536) | 4 | 2.78e-17 | 4.52e-03 | tone_37hz=0.9993, tone_113hz=0.9973 |

## case B

| n | MD-VMD (s) | vmdpy (s) | PySDKit (s) | vmdpy/MD | PySDKit/MD | PySDKit/vmdpy |
|---:|---:|---:|---:|---:|---:|---:|
| 256 | 0.0754 | 0.1092 | 0.1069 | 1.4x | 1.4x | 1.0x |
| 1024 | 0.0048 | 0.0099 | 0.2275 | 2.0x | 47.1x | 23.0x |
| 4096 | 0.0095 | 0.0239 | 0.6145 | 2.5x | 64.7x | 25.7x |
| 16384 | 0.0663 | 0.1847 | 4.1380 | 2.8x | 62.4x | 22.4x |
| 65536 | 0.1691 | 1.1177 | 25.9028 | 6.6x | 153.2x | 23.2x |

Quality:

| impl | rows | recon max abs err | orthogonality idx | mode recovery (best abs corr) |
|---|---:|---:|---:|---|
| MD-VMD (n=65536) | 4 | 1.73e-18 | 4.45e-02 | amfm=0.9996, tone_89hz=0.9999, trend=0.9948 |
| vmdpy (n=65536) | 4 | 1.73e-18 | 4.45e-02 | amfm=0.9996, tone_89hz=0.9999, trend=0.9948 |
| PySDKit (n=65536) | 4 | 1.73e-18 | 4.45e-02 | amfm=0.9996, tone_89hz=0.9999, trend=0.9948 |

## case C

| n | MD-VMD (s) | vmdpy (s) | PySDKit (s) | vmdpy/MD | PySDKit/MD | PySDKit/vmdpy |
|---:|---:|---:|---:|---:|---:|---:|
| 256 | 0.0061 | 0.0131 | 0.1730 | 2.1x | 28.2x | 13.2x |
| 1024 | 0.0368 | 0.0475 | 0.2870 | 1.3x | 7.8x | 6.0x |
| 4096 | 0.1535 | 0.6446 | 0.8755 | 4.2x | 5.7x | 1.4x |
| 16384 | 1.8711 | 5.8079 | 6.5198 | 3.1x | 3.5x | 1.1x |
| 65536 | - | - | - | - | - | - |

Quality:

| impl | rows | recon max abs err | orthogonality idx | mode recovery (best abs corr) |
|---|---:|---:|---:|---|
| MD-VMD (n=16384) | 5 | 4.44e-16 | 5.96e-01 | - |
| vmdpy (n=16384) | 5 | 4.44e-16 | 5.96e-01 | - |
| PySDKit (n=16384) | 5 | 4.44e-16 | 5.96e-01 | - |
