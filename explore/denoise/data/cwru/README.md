# CWRU bearing data (local copy)

- Source: Case Western Reserve University Bearing Data Center.
- Mirror used for download: `https://github.com/s-whynot/CWRU-dataset`
  (path `12k_Drive_End_Bearing_Fault_Data/IR/007/105_0.mat`).
- File: `IR007_105_0.mat` — 12 kHz drive-end acceleration, inner-race fault
  (0.007 in), load 0, shaft speed 1797 rpm.
- Variable used: `X105_DE_time` (121,265 samples), `X105RPM` (1797).
- Characteristic inner-race frequency used in the report:
  BPFI ≈ 5.4152 × fr, fr = 1797/60 = 29.95 Hz → 162.2 Hz.

If you redistribute this file, cite the CWRU Bearing Data Center and keep the
dataset's attribution/terms. The experiment code only needs the `.mat` file;
`run_cwru.py` reads it relative to the repository root.
