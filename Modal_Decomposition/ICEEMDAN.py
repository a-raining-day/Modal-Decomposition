import numpy as np
from .EMD import emd
from .help_function import is_increasing


def iceemdan(S, T=None, Ne=300, epsilon_0=None, max_imf=None, verbose: bool=False, spline_kind: str = "cubic", nbsym: int = 2, emd_max_imf=-1, fs=None):
    """
    :param verbose: is print formation
    :param S: Signal (1-dim)
    :param Ne: total num of samples (times of add noise), default 300
    :param epsilon_0: initial amplitude of noise，default: 0.2 * std(S) / std(noise)
    :param max_imf: max num of IMFs
    :return: IMFs, Res
    """

    if not isinstance(S, np.ndarray):
        S = np.array(S)

    T_len = len(S)
    if T is None:
        T = np.arange(0, T_len, 1)

    white_noise = np.random.randn(Ne, T_len)  # generate Ne white nose

    if verbose:
        print(f"decomposition {Ne} of white noise ...")
    white_imfs_list = []

    for i in range(Ne):
        imfs, _ = emd(white_noise[i, :], T, spline_kind=spline_kind, nbsym=nbsym, max_imf=emd_max_imf)
        white_imfs_list.append(imfs)

    max_k = max(len(imfs) for imfs in white_imfs_list)

    std_Ek = np.zeros(max_k)
    for k in range(max_k):
        ek_values = []
        for i in range(Ne):
            imfs = white_imfs_list[i]
            if k < len(imfs):
                ek_values.append(np.std(imfs[k]))
        if ek_values:
            std_Ek[k] = np.mean(ek_values)
        else:
            std_Ek[k] = 0

    if epsilon_0 is None:
        epsilon_0 = 0.2 * np.std(S) / std_Ek[0] if std_Ek[0] > 0 else 0.2

    residue = S.copy()
    imfs_list = []

    if max_imf is None:
        max_imf = int(np.log2(T_len))

    for k in range(max_imf):
        if verbose:
            print(f"get {k + 1} IMF")

        if k == 0:
            epsilon_k = epsilon_0
        else:
            if std_Ek[0] > 0 and k < len(std_Ek):
                epsilon_k = epsilon_0 * std_Ek[k] / std_Ek[0] * np.sqrt(Ne)
            else:
                epsilon_k = epsilon_0 * np.sqrt(Ne)

        imf_candidates = np.zeros((Ne, T_len))

        valid_count = 0
        for i in range(Ne):
            white_imfs = white_imfs_list[i]

            if k >= len(white_imfs):
                continue

            Ek_wi = white_imfs[k]

            noisy_signal = residue + epsilon_k * Ek_wi

            imfs_noisy, _ = emd(noisy_signal, T, spline_kind=spline_kind, nbsym=nbsym, max_imf=emd_max_imf)

            if len(imfs_noisy) > 0:
                imf_candidates[valid_count] = imfs_noisy[0]
                valid_count += 1

        if valid_count == 0:
            if verbose:
                print(f"Warn: the {k + 1}-order IMF has no valid candidates, decomposition stop")
            break

        imf_candidates = imf_candidates[:valid_count]

        current_imf = np.mean(imf_candidates, axis=0)

        imfs_list.append(current_imf)

        residue = residue - current_imf

        if k > 0:
            if len(imfs_list) >= 2:
                prev_imf = imfs_list[-2]
                correlation = np.corrcoef(current_imf, prev_imf)[0, 1]
                if abs(correlation) > 0.95:
                    if verbose:
                        print(f"IMF{k + 1} and IMF{k} are highly correlated (corr={correlation:.3f}), decomposition stop")
                    imfs_list.pop()
                    residue = residue + current_imf
                    break

        if is_increasing(residue) or is_increasing(-residue):
            break

        residue_energy = np.sum(residue ** 2)
        if residue_energy < 1e-10:
            break

        imf_energy = np.sum(current_imf ** 2)
        if imf_energy < 1e-10:
            imfs_list.pop()  # 移除这个IMF
            break

        from scipy.signal import argrelextrema
        maxima = argrelextrema(residue, np.greater)[0]
        minima = argrelextrema(residue, np.less)[0]
        if len(maxima) + len(minima) <= 2:
            break

    imfs_array = np.array(imfs_list)

    return imfs_array, np.array(residue)