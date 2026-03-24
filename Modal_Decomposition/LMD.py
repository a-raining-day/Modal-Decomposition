import numpy as np
import scipy.signal as sg
import scipy.interpolate as ip
from .help_function import is_increasing
from scipy.signal import argrelextrema
from scipy.signal import hilbert


def lmd(S, max_pf=None, max_iter=37, eps=0.05):
    """
    :param S: Signal (1-dim)
    :param max_pf: max num of pfs
    :param max_iter: max iterations of each pf
    :param eps: therhold
    :return: PFs, Res
    """
    if not isinstance(S, np.ndarray):
        S = np.array(S)

    T = len(S)
    t = np.arange(0, T, 1)

    if max_pf is None:
        max_pf = int(np.log2(T))

    PFs = []
    residue = S.copy()

    for pf_idx in range(max_pf):
        h = residue.copy()
        a_total = np.ones(T)
        s = h.copy()

        for iter_num in range(max_iter):
            max_idx = argrelextrema(h, np.greater)[0]
            min_idx = argrelextrema(h, np.less)[0]

            extrema_idx = np.sort(np.concatenate([max_idx, min_idx]))

            if len(extrema_idx) < 4:
                break

            m_values = []
            a_values = []
            t_mid = []

            for i in range(len(extrema_idx) - 1):
                t1, t2 = extrema_idx[i], extrema_idx[i + 1]
                v1, v2 = h[t1], h[t2]

                mid_t = (t1 + t2) / 2.0
                t_mid.append(mid_t)

                m = (v1 + v2) / 2.0
                m_values.append(m)

                a = np.abs(v1 - v2) / 2.0
                a_values.append(a)

            t_mid = np.array(t_mid)
            m_values = np.array(m_values)
            a_values = np.array(a_values)

            t_mid = np.clip(t_mid, 0, T - 1)
            a_values = np.maximum(a_values, 1e-10)
            a_values = np.minimum(a_values, 1e10)

            try:
                m_interp = ip.PchipInterpolator(t_mid, m_values, extrapolate=False)
                a_interp = ip.PchipInterpolator(t_mid, a_values, extrapolate=False)

                m_t = m_interp(t)
                a_t = a_interp(t)

                if np.any(np.isnan(m_t)) or np.any(np.isnan(a_t)):
                    raise ValueError("NaN in interpolation")

            except:
                m_t = np.interp(t, t_mid, m_values, left=m_values[0], right=m_values[-1])
                a_t = np.interp(t, t_mid, a_values, left=a_values[0], right=a_values[-1])

            a_t = np.maximum(a_t, 1e-10)
            a_t = np.minimum(a_t, 1e10)

            h_new = h - m_t
            s_new = h_new / a_t

            a_total = a_total * a_t
            a_total = np.clip(a_total, 1e-20, 1e20)

            a_deviation = np.max(np.abs(a_t - 1.0))

            if np.var(a_t) < 1e-6:
                s = s_new
                break

            if a_deviation < eps:
                s = s_new
                break

            h = s_new
            s = s_new

        current_pf = a_total * s

        if np.any(np.isnan(current_pf)) or np.any(np.isinf(current_pf)):
            print(f"Warning: PF {pf_idx + 1} contains invalid values. Stopping decomposition.")
            break

        PFs.append(current_pf)

        residue = residue - current_pf

        if is_increasing(residue) or is_increasing(-residue):
            break

        residue_energy = np.sum(residue ** 2)
        if residue_energy < 1e-10:
            break

        max_idx = argrelextrema(residue, np.greater)[0]
        min_idx = argrelextrema(residue, np.less)[0]
        if len(max_idx) + len(min_idx) <= 2:
            break

        pf_energy = np.sum(current_pf ** 2)
        if pf_energy < 1e-10:
            PFs.pop()  # remove
            break

    return np.array(PFs, dtype=np.float64), residue

def compute_envelope(signal):
    analytic_signal = hilbert(signal)
    amplitude_envelope = np.abs(analytic_signal)
    return amplitude_envelope


if __name__ == "__main__":
    t = np.linspace(0, 10, 1000)
    S = np.sin(2 * np.pi * 5 * t) * (1 + 0.5 * np.sin(2 * np.pi * 0.5 * t))

    PFs, residue = lmd(S, max_pf=5, eps=0.01)

    reconstructed = np.sum(PFs, axis=0) + residue
    error = np.max(np.abs(S - reconstructed))
    print(f"error: {error:.6f}")