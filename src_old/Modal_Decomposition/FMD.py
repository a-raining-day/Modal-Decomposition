"""
Python version:  (must)
    3.10.11

Lib and Version:  (if None write None)
    scipy - 1.15.3
	matplotlib - 3.10.8
	numpy - 2.2.6

Only accessed by:  (must)
    Only __init__.py

Description: (if None write None)
    Realize the FMD

Modify:  (must)
    2026.3.25 - Create.
    2026.4.3  - Finish the Optimization of the FMD.
    2026.4.3  - Make the num_hand parameter can be generated auto.
"""

import numpy as np
from typing import Union, Tuple, Optional
from .Utils import Check_Time_and_Signal

def fmd(
    S: Union[list, np.ndarray],
    T: Optional[Union[list, np.ndarray]] = None,
    M: Optional[int] = None,
    L_factor: int = 10,
    K: int = 10,
    max_iter: int = 50,
    tol: float = 1e-6,
    num_hand: int=None,
    seed: Optional[int] = 42
) -> Tuple[np.ndarray, np.ndarray, None]:
    """
    FMD: Feature Mode Decomposition

    :param S: Signal (N,)
    :param T: Time-axis, None means default fs = 1.
    :param M: the period.
    :param L_factor: the factor of the L, default 10, means 10 * M.
    :param K: the max num of the IMFs. -1 means decompose completely.
    :param max_iter: the max num of the iterations. 10 is enough to converge.
    :param tol: the epsilon of the energy of the Ress.
    :param num_hand: the candidacy of the filter.
    :param seed: random seed
    :return: IMFs(max_IMFs, N), Res(N,), None
    """
    from scipy.signal import argrelextrema
    from scipy.linalg import eigh, eig

    S, T, N = Check_Time_and_Signal(S, T)

    S = np.asarray(S, dtype=np.float64).ravel()
    N = len(S)
    S_original = S.copy()

    if N < 10 or np.allclose(S, S[0], atol=1e-8):
        return np.empty((0, N)), S_original, None

    if seed is not None:
        np.random.seed(seed)

    mean_S = np.mean(S)
    std_S = np.std(S) + 1e-12
    S_norm = (S - mean_S) / std_S

    if M is None:
        M = _robust_period_estimation(S_norm, 1.0)
    M = int(np.clip(M, 2, N // 4))
    L = int(np.clip(L_factor * M, 10, N // 2))

    if num_hand is None or num_hand <= 0:
        num_hand = _auto_estimate_num_candidates(S_norm, M, L, 1.0)

    residual = S_norm.copy()
    modes_list = []

    if K != -1:
        for k in range(K):
            x = residual.copy()
            Nx = len(x)

            if Nx - L + 1 <= 2 * M + 2:
                break

            try:
                X = np.lib.stride_tricks.sliding_window_view(x, L)
            except AttributeError:
                rows = Nx - L + 1
                X = np.zeros((rows, L))
                for i in range(rows):
                    X[i] = x[i:i+L]

            K_len = X.shape[0]
            M_eff = min(M, K_len - 1)

            X_short = X[:K_len - M_eff, :]
            X_shift = X[M_eff:, :]

            h = _advanced_filter_init(X_short, X_shift, num_hand=num_hand)
            h_prev = h.copy()

            for it in range(max_iter):
                y = X @ h
                y_short = y[:K_len - M_eff]
                y_shift = y[M_eff:]

                y2_sum = np.sum(y_short ** 2)
                if y2_sum < 1e-12:
                    break

                ck = np.sum((y_short * y_shift) ** 2) / (y2_sum ** 2)

                w = y_short * y_shift
                y2 = y_short ** 2

                A = (X_short.T * w) @ X_shift
                B = (X_short.T * y2) @ X_short

                A = 0.5 * (A + A.T)
                B = 0.5 * (B + B.T)

                B_trace = np.trace(B)
                reg = 1e-6 * B_trace / L if B_trace > 0 else 1e-6
                B_reg = B + reg * np.eye(L) + 1e-12 * np.eye(L)

                try:
                    eigvals, eigvecs = eigh(A, B_reg, subset_by_index=[L-1, L-1])
                    h_new = eigvecs[:, 0].real
                except np.linalg.LinAlgError:
                    try:
                        B_inv = np.linalg.pinv(B_reg, rcond=1e-12)
                        eig_vals, eig_vecs = eig(B_inv @ A)
                        max_idx = np.argmax(np.real(eig_vals))
                        h_new = np.real(eig_vecs[:, max_idx])
                    except np.linalg.LinAlgError:
                        h_new = h_prev.copy()

                h_norm = np.linalg.norm(h_new)
                if h_norm > 1e-12:
                    h_new = h_new / h_norm

                h_diff = np.linalg.norm(h_new - h)
                if h_diff < tol or (it > 0 and h_diff < 0.01):
                    h = h_new
                    break

                h_prev = h.copy()
                h = h_new

            mode = _bounded_convolution(x, h)

            mode_norm = np.linalg.norm(mode)
            if mode_norm < 1e-12 or np.std(mode) < 1e-8:
                break

            modes_list.append(mode)
            residual = residual - mode

            if np.sum(residual ** 2) < 1e-10 * np.sum(S_norm ** 2):
                break

            peaks_max = argrelextrema(residual, np.greater)[0]
            peaks_min = argrelextrema(residual, np.less)[0]
            if len(peaks_max) + len(peaks_min) < 3:
                break

    else:
        k = 0
        while True:
            x = residual.copy()
            Nx = len(x)

            if Nx - L + 1 <= 2 * M + 2:
                break

            try:
                X = np.lib.stride_tricks.sliding_window_view(x, L)
            except AttributeError:
                rows = Nx - L + 1
                X = np.zeros((rows, L))
                for i in range(rows):
                    X[i] = x[i:i + L]

            K_len = X.shape[0]
            M_eff = min(M, K_len - 1)

            X_short = X[:K_len - M_eff, :]
            X_shift = X[M_eff:, :]

            h = _advanced_filter_init(X_short, X_shift, num_hand=num_hand)
            h_prev = h.copy()

            for it in range(max_iter):
                y = X @ h
                y_short = y[:K_len - M_eff]
                y_shift = y[M_eff:]

                y2_sum = np.sum(y_short ** 2)
                if y2_sum < 1e-12:
                    break

                ck = np.sum((y_short * y_shift) ** 2) / (y2_sum ** 2)

                w = y_short * y_shift
                y2 = y_short ** 2

                A = (X_short.T * w) @ X_shift
                B = (X_short.T * y2) @ X_short

                A = 0.5 * (A + A.T)
                B = 0.5 * (B + B.T)

                B_trace = np.trace(B)
                reg = 1e-6 * B_trace / L if B_trace > 0 else 1e-6
                B_reg = B + reg * np.eye(L) + 1e-12 * np.eye(L)

                try:
                    eigvals, eigvecs = eigh(A, B_reg, subset_by_index=[L - 1, L - 1])
                    h_new = eigvecs[:, 0].real
                except np.linalg.LinAlgError:
                    try:
                        B_inv = np.linalg.pinv(B_reg, rcond=1e-12)
                        eig_vals, eig_vecs = eig(B_inv @ A)
                        max_idx = np.argmax(np.real(eig_vals))
                        h_new = np.real(eig_vecs[:, max_idx])
                    except np.linalg.LinAlgError:
                        h_new = h_prev.copy()

                h_norm = np.linalg.norm(h_new)
                if h_norm > 1e-12:
                    h_new = h_new / h_norm

                h_diff = np.linalg.norm(h_new - h)
                if h_diff < tol or (it > 0 and h_diff < 0.01):
                    h = h_new
                    break

                h_prev = h.copy()
                h = h_new

            mode = _bounded_convolution(x, h)

            mode_norm = np.linalg.norm(mode)
            if mode_norm < 1e-12 or np.std(mode) < 1e-8:
                break

            modes_list.append(mode)
            residual = residual - mode
            k += 1

            if np.sum(residual ** 2) < 1e-10 * np.sum(S_norm ** 2):
                break

            peaks_max = argrelextrema(residual, np.greater)[0]
            peaks_min = argrelextrema(residual, np.less)[0]
            if len(peaks_max) + len(peaks_min) < 3:
                break

    modes_array = np.array(modes_list, dtype=np.float64) * std_S if modes_list else np.empty((0, N))
    residual_final = residual * std_S + mean_S

    return modes_array, residual_final, None

def _auto_estimate_num_candidates(signal: np.ndarray, M: int, L: int, fs: float) -> int:
    N = len(signal)

    snr_estimate = _estimate_snr(signal)

    complexity = _estimate_signal_complexity(signal)

    base_num = 10  # base num

    if snr_estimate < 10:
        base_num += 20
    elif snr_estimate < 20:
        base_num += 10

    if complexity > 0.7:
        base_num += 15
    elif complexity > 0.4:
        base_num += 8

    if N > 5000:
        base_num += 5
    elif N > 1000:
        base_num += 2

    if M < 50:
        base_num += 5

    return int(np.clip(base_num, 5, 70))

def _estimate_snr(signal: np.ndarray) -> float:
    from scipy.signal import hilbert

    analytic = hilbert(signal)
    envelope = np.abs(analytic)

    from scipy.signal import savgol_filter
    envelope_smooth = savgol_filter(envelope, window_length=51, polyorder=3)
    noise_est = np.std(envelope - envelope_smooth)
    signal_est = np.std(envelope_smooth)

    if noise_est < 1e-12:
        return 100.0

    snr = 20 * np.log10(signal_est / noise_est)
    return max(0, snr)

def _estimate_signal_complexity(signal: np.ndarray) -> float:
    N = len(signal)

    m = 2
    r = 0.2 * np.std(signal)

    def _maxdist(xi, xj):
        return np.max(np.abs(xi - xj))

    def _phi(m):
        x = np.array([signal[i:i + m] for i in range(N - m + 1)])
        C = np.zeros(len(x))

        for i in range(len(x)):
            d = [_maxdist(x[i], x[j]) for j in range(len(x)) if j != i]
            C[i] = np.sum(np.array(d) <= r) / (len(x) - 1)

        return np.sum(np.log(C + 1e-12)) / len(x)

    if N > 2 * m + 1:
        apen = _phi(m) - _phi(m + 1)
        apen_norm = min(max(apen, 0), 2) / 2
    else:
        apen_norm = 0.5

    spec = np.abs(np.fft.rfft(signal))
    spec = spec[spec > 0]
    p = spec / np.sum(spec)
    entropy = -np.sum(p * np.log2(p + 1e-12))
    max_entropy = np.log2(len(p))
    spectral_entropy_norm = entropy / max_entropy if max_entropy > 0 else 0

    zero_crossings = np.sum(np.diff(np.signbit(signal)))
    zcr_norm = zero_crossings / (N - 1)

    complexity = 0.4 * apen_norm + 0.4 * spectral_entropy_norm + 0.2 * zcr_norm
    return np.clip(complexity, 0, 1)

def _robust_period_estimation(x: np.ndarray, fs: float) -> int:
    from scipy.signal import find_peaks, hilbert

    n = len(x)
    x = x - np.mean(x)

    max_lag = min(1000, n // 2)
    autocorr = np.correlate(x, x, mode='full')[n-1:n-1+max_lag]
    peaks, _ = find_peaks(autocorr, distance=5, prominence=0.1*np.max(autocorr))
    period_ac = peaks[0] if len(peaks) > 0 and peaks[0] > 0 else 10

    analytic = hilbert(x)
    envelope = np.abs(analytic)
    n_fft = 2**int(np.ceil(np.log2(len(envelope))))
    spec = np.abs(np.fft.rfft(envelope, n=n_fft))
    freqs = np.fft.rfftfreq(n_fft, 1/fs)

    valid_freqs = (freqs > 0.01*fs) & (freqs < 0.45*fs)
    if np.any(valid_freqs):
        f_main = freqs[valid_freqs][np.argmax(spec[valid_freqs])]
        period_env = int(round(fs / f_main)) if f_main > 0 else 10
    else:
        period_env = 10

    period = int(round(0.7*period_ac + 0.3*period_env))
    return max(2, min(period, n // 4))

def _advanced_filter_init(X_short: np.ndarray, X_shift: np.ndarray, num_hand: int=10) -> np.ndarray:
    from scipy.linalg import svd

    L = X_short.shape[1]

    candidates = []

    for i in range(-2, 3):
        h = np.zeros(L)
        pos = L//2 + i
        if 0 <= pos < L:
            h[pos] = 1.0
        candidates.append(h)

    try:
        _, _, Vh = svd(X_short, full_matrices=False)
        svd_h = Vh[0, :]
        candidates.append(svd_h)
    except:
        pass

    for _ in range(num_hand):
        h_rand = np.random.randn(L)
        candidates.append(h_rand)

    best_ck = -1e12
    best_h = np.zeros(L)
    best_h[L//2] = 1.0

    for h_candidate in candidates:
        h_norm = h_candidate / (np.linalg.norm(h_candidate) + 1e-12)
        y = X_short @ h_norm
        y_shift = X_shift @ h_norm
        y2_sum = np.sum(y ** 2)
        if y2_sum < 1e-12:
            continue
        ck = np.sum((y * y_shift) ** 2) / (y2_sum ** 2)
        if ck > best_ck:
            best_ck = ck
            best_h = h_norm

    return best_h

def _bounded_convolution(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    L = len(h)
    N = len(x)

    if L >= N:
        return np.convolve(x, h, mode='same')

    pad = L // 2
    x_pad = np.pad(x, (pad, pad), mode='reflect')
    y_pad = np.convolve(x_pad, h, mode='valid')

    start_idx = (len(y_pad) - N) // 2
    return y_pad[start_idx:start_idx+N]


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    np.random.seed(42)
    t = np.linspace(0, 1, 1000)
    signal = (2.0 * np.sin(2 * np.pi * 10 * t) +
              1.5 * np.sin(2 * np.pi * 25 * t) +
              0.8 * np.sin(2 * np.pi * 50 * t) +
              0.3 * np.random.randn(len(t)))

    modes, residual = fmd(signal, K=4)

    reconstructed = np.sum(modes, axis=0) + residual
    reconstruction_error = np.max(np.abs(signal - reconstructed))

    print("=" * 50)
    print(f"num of IMFs: {modes.shape[0]}")
    print(f"decomposition error: {reconstruction_error:.2e}")
    print(f"energy of the origin: {np.sum(signal ** 2):.4f}")
    print(f"energy of the constructed: {np.sum(reconstructed ** 2):.4f}")
    print(f"error: {abs(np.sum(signal ** 2) - np.sum(reconstructed ** 2)):.2e}")
    print("=" * 50)

    # 可视化
    fig, axes = plt.subplots(2 + len(modes), 1, figsize=(10, 2 * (2 + len(modes))))

    axes[0].plot(t, signal)
    axes[0].set_title("Origin")

    for i, mode in enumerate(modes):
        axes[i + 1].plot(t, mode)
        axes[i + 1].set_title(f"Modal {i + 1}")

    axes[-1].plot(t, residual)
    axes[-1].set_title("Res")

    plt.tight_layout()
    plt.show()