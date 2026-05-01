"""
Modal Decomposition Test
"""
import numpy as np
from src.Modal_Decomposition import Function

def gen_sinusoidal(length=512, fs=1.0, seed=42):
    rng = np.random.default_rng(seed)
    t = np.arange(length) / fs
    comps = [
        1.0 * np.sin(2 * np.pi * 5 * t),
        0.7 * np.sin(2 * np.pi * 20 * t),
        0.4 * np.sin(2 * np.pi * 50 * t)
    ]
    true = np.array(comps)
    signal = np.sum(true, axis=0) + rng.normal(0, 0.05, length)
    return t, signal, true

def gen_chirp(length=512, fs=1.0, seed=42):
    rng = np.random.default_rng(seed)
    t = np.arange(length) / fs
    chirp = np.sin(2*np.pi*(0.01*t + (0.2-0.01)/(2*length)*t**2))
    sine  = 0.5 * np.sin(2*np.pi*0.35*t)
    true = np.array([chirp, sine])
    signal = chirp + sine + rng.normal(0, 0.02, length)
    return t, signal, true

def gen_intermittent(length=512, fs=1.0, seed=42):
    rng = np.random.default_rng(seed)
    t = np.arange(length) / fs
    burst = np.sin(2*np.pi*0.12*t) * (np.abs(t-256/fs)<100/fs).astype(float)
    sine  = np.sin(2*np.pi*0.03*t)
    true = np.array([burst, sine])
    signal = burst + sine + rng.normal(0, 0.02, length)
    return t, signal, true

def gen_amfm(length=512, fs=1.0, seed=42):
    rng = np.random.default_rng(seed)
    t = np.arange(length) / fs
    am = 1.0 + 0.5*np.cos(2*np.pi*0.02*t)
    fc = 0.1 + 0.05*np.sin(2*np.pi*0.03*t)
    phase = 2*np.pi*np.cumsum(fc)/fs
    comp = am * np.sin(phase)
    true = np.array([comp])
    signal = comp + rng.normal(0, 0.02, length)
    return t, signal, true

def compute_metrics(orig, imfs, res):
    orig = np.asarray(orig)
    imfs = np.asarray(imfs)
    res = np.asarray(res)

    if imfs.ndim == 3:
        recon = np.sum(imfs, axis=0) + res  # shape (d, N)
        if orig.ndim == 1:
            orig = orig.reshape(1, -1)
        rmse = np.sqrt(np.mean((orig - recon) ** 2))
        sig_pow = np.mean(orig ** 2)
    else:
        recon = np.sum(imfs, axis=0) + res
        rmse = np.sqrt(np.mean((orig - recon) ** 2))
        sig_pow = np.mean(orig ** 2)

    if rmse < 1e-12:
        psnr = 100.0
    else:
        psnr = 10 * np.log10(sig_pow / (rmse ** 2))
    return {"rmse": rmse, "psnr": psnr, "num_imfs": imfs.shape[0]}


tests = [
    ("EMD",       lambda S, T, **kw: Function.EMD(S), {}),
    ("EEMD",      lambda S, T, **kw: Function.EEMD(S), {"trials": 50}),
    ("CEEMD",     lambda S, T, **kw: Function.CEEMD(S), {}),
    ("CEEMDAN",   lambda S, T, **kw: Function.CEEMDAN(S), {}),
    ("ICEEMDAN",  lambda S, T, **kw: Function.ICEEMDAN(S), {"ensemble_size": 50, "verbose": False}),
    ("LMD",       lambda S, T, **kw: Function.LMD(S), {}),
    ("EFD",       lambda S, T, **kw: Function.EFD(S), {}),
    ("CEEFD",     lambda S, T, **kw: Function.CEEFD.decompose(S), {}),
    ("EWT",       lambda S, T, **kw: Function.EWT(S), {}),
    ("VMD",       lambda S, T, **kw: Function.VMD(S), {"K": 3}),
    ("SVMD",      lambda S, T, **kw: Function.SVMD(S), {"num_modes": 3}),
    ("FMD",       lambda S, T, **kw: Function.FMD(S), {"K": 3}),
    ("SSA",       lambda S, T, **kw: Function.SSA(S), {}),
    ("RPSEMD",    lambda S, T, **kw: Function.RPSEMD(S), {}),
    ("MEMD",      lambda S, T, **kw: Function.MEMD(S.reshape(1, -1)), {}),
]

signal_suites = {
    "Sinusoidal":   gen_sinusoidal,
    "Chirp":        gen_chirp,
    "Intermittent": gen_intermittent,
    "AM-FM":        gen_amfm,
}

if __name__ == "__main__":
    print("="*60)
    print("Modal Decomposition Test")
    print("="*60)

    all_res = []
    for m_name, wrapper, params in tests:
        print(f"\nName -> {m_name}")
        for s_name, s_func in signal_suites.items():
            t, sig, _ = s_func()
            imfs, res, info = wrapper(sig, t, **params)
            met = compute_metrics(sig, imfs, res)
            met["method"] = m_name
            met["signal"] = s_name

            print(f"{m_name:10s} | {s_name:15s} | RMSE={met['rmse']:.2e}  "
                  f"PSNR={met['psnr']:.1f} dB  IMFs={met['num_imfs']}")

            all_res.append(met)

    failed = [r for r in all_res if r["rmse"] > 0.1]
    if failed:
        print("\nThere're some errors or big construction error in following:")
        for f in failed:
            err = f.get("error", "")
            print(f"  {f['method']:10s} on {f['signal']:15s} | RMSE={f['rmse']:.2e}  "
                  f"{('| '+err) if err else ''}")
    else:
        print("\nAll Test Pass (RMSE < 0.1)")


"""
Name -> EMD
EMD        | Sinusoidal      | RMSE=1.20e-20  PSNR=100.0 dB  IMFs=7
EMD        | Chirp           | RMSE=5.54e-19  PSNR=100.0 dB  IMFs=6
EMD        | Intermittent    | RMSE=4.28e-19  PSNR=100.0 dB  IMFs=5
EMD        | AM-FM           | RMSE=6.91e-20  PSNR=100.0 dB  IMFs=5

Name -> EEMD
EEMD       | Sinusoidal      | RMSE=1.54e-03  PSNR=30.0 dB  IMFs=7
EEMD       | Chirp           | RMSE=4.59e-02  PSNR=24.7 dB  IMFs=7
EEMD       | Intermittent    | RMSE=3.23e-02  PSNR=28.2 dB  IMFs=7
EEMD       | AM-FM           | RMSE=2.03e-02  PSNR=31.4 dB  IMFs=7

Name -> CEEMD
CEEMD      | Sinusoidal      | RMSE=7.49e-18  PSNR=100.0 dB  IMFs=11
CEEMD      | Chirp           | RMSE=1.19e-16  PSNR=100.0 dB  IMFs=11
CEEMD      | Intermittent    | RMSE=1.01e-16  PSNR=100.0 dB  IMFs=11
CEEMD      | AM-FM           | RMSE=9.90e-17  PSNR=100.0 dB  IMFs=11

Name -> CEEMDAN
CEEMDAN    | Sinusoidal      | RMSE=8.11e-18  PSNR=100.0 dB  IMFs=6
CEEMDAN    | Chirp           | RMSE=1.29e-16  PSNR=100.0 dB  IMFs=6
CEEMDAN    | Intermittent    | RMSE=1.27e-16  PSNR=100.0 dB  IMFs=5
CEEMDAN    | AM-FM           | RMSE=1.22e-16  PSNR=100.0 dB  IMFs=6

Name -> ICEEMDAN
ICEEMDAN   | Sinusoidal      | RMSE=4.99e-18  PSNR=100.0 dB  IMFs=5
ICEEMDAN   | Chirp           | RMSE=9.58e-17  PSNR=100.0 dB  IMFs=5
ICEEMDAN   | Intermittent    | RMSE=7.99e-17  PSNR=100.0 dB  IMFs=5
ICEEMDAN   | AM-FM           | RMSE=8.06e-17  PSNR=100.0 dB  IMFs=5

Name -> LMD
LMD        | Sinusoidal      | RMSE=2.83e-05  PSNR=64.7 dB  IMFs=9
LMD        | Chirp           | RMSE=6.18e-05  PSNR=82.1 dB  IMFs=9
LMD        | Intermittent    | RMSE=5.64e-05  PSNR=83.4 dB  IMFs=9
LMD        | AM-FM           | RMSE=6.39e-05  PSNR=81.4 dB  IMFs=9

Name -> EFD
EFD        | Sinusoidal      | RMSE=1.34e-18  PSNR=100.0 dB  IMFs=84
EFD        | Chirp           | RMSE=5.99e-17  PSNR=100.0 dB  IMFs=72
EFD        | Intermittent    | RMSE=1.46e-17  PSNR=100.0 dB  IMFs=80
EFD        | AM-FM           | RMSE=3.82e-17  PSNR=100.0 dB  IMFs=77

Name -> CEEFD
CEEFD      | Sinusoidal      | RMSE=3.82e-04  PSNR=42.1 dB  IMFs=20
CEEFD      | Chirp           | RMSE=1.43e-05  PSNR=94.9 dB  IMFs=17
CEEFD      | Intermittent    | RMSE=1.17e-03  PSNR=57.1 dB  IMFs=21
CEEFD      | AM-FM           | RMSE=1.66e-03  PSNR=53.2 dB  IMFs=21

Name -> EWT (from ewtpy)
EWT        | Sinusoidal      | RMSE=5.72e-03  PSNR=18.6 dB  IMFs=4
EWT        | Chirp           | RMSE=1.36e-01  PSNR=15.2 dB  IMFs=4
EWT        | Intermittent    | RMSE=1.27e-01  PSNR=16.3 dB  IMFs=4
EWT        | AM-FM           | RMSE=2.19e-01  PSNR=10.8 dB  IMFs=4

Name -> VMD (from vmdpy)
VMD        | Sinusoidal      | RMSE=3.79e-02  PSNR=2.2 dB  IMFs=2
VMD        | Chirp           | RMSE=5.01e-01  PSNR=3.9 dB  IMFs=2
VMD        | Intermittent    | RMSE=7.25e-02  PSNR=21.2 dB  IMFs=2
VMD        | AM-FM           | RMSE=2.61e-01  PSNR=9.2 dB  IMFs=2

Name -> SVMD
SVMD       | Sinusoidal      | RMSE=1.03e-14  PSNR=100.0 dB  IMFs=3
SVMD       | Chirp           | RMSE=2.68e-11  PSNR=209.4 dB  IMFs=3
SVMD       | Intermittent    | RMSE=8.33e-11  PSNR=200.0 dB  IMFs=3
SVMD       | AM-FM           | RMSE=1.15e-10  PSNR=196.4 dB  IMFs=3

Name -> FMD
FMD        | Sinusoidal      | RMSE=5.59e-17  PSNR=100.0 dB  IMFs=10
FMD        | Chirp           | RMSE=6.01e-16  PSNR=100.0 dB  IMFs=10
FMD        | Intermittent    | RMSE=5.40e-16  PSNR=100.0 dB  IMFs=10
FMD        | AM-FM           | RMSE=8.92e-15  PSNR=100.0 dB  IMFs=10

Name -> SSA
SSA        | Sinusoidal      | RMSE=3.51e-17  PSNR=100.0 dB  IMFs=170
SSA        | Chirp           | RMSE=8.10e-16  PSNR=100.0 dB  IMFs=170
SSA        | Intermittent    | RMSE=2.22e-15  PSNR=100.0 dB  IMFs=170
SSA        | AM-FM           | RMSE=8.41e-16  PSNR=100.0 dB  IMFs=170

Name -> RPSEMD
RPSEMD     | Sinusoidal      | RMSE=2.52e-18  PSNR=100.0 dB  IMFs=4
RPSEMD     | Chirp           | RMSE=8.92e-17  PSNR=100.0 dB  IMFs=6
RPSEMD     | Intermittent    | RMSE=7.97e-17  PSNR=100.0 dB  IMFs=5
RPSEMD     | AM-FM           | RMSE=7.34e-17  PSNR=100.0 dB  IMFs=5

MEMD       | Sinusoidal      | RMSE=6.73e-18  PSNR=100.0 dB  IMFs=9
MEMD       | Chirp           | RMSE=1.07e-16  PSNR=100.0 dB  IMFs=7
MEMD       | Intermittent    | RMSE=9.16e-17  PSNR=100.0 dB  IMFs=5
MEMD       | AM-FM           | RMSE=1.04e-16  PSNR=100.0 dB  IMFs=8
"""