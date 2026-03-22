import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import firwin, lfilter, correlate, find_peaks
import pandas as pd

def initialize_filters(L, K):
    filters = []
    for k in range(1, K + 1):
       cutoff = 0.5 / k
       filter = firwin(L, cutoff, window='hann')
       filters.append(filter)
    return filters

def estimate_period(signal):
    correlation = correlate(signal, signal, mode='full')
    correlation = correlation[len(correlation) // 2:]
    peaks, _ = find_peaks(correlation)
    if len(peaks) > 1:
       period = peaks[1]
    else:
       period = len(signal)
    return period

def fmd(S, n, L=100, max_iters=10):
    """
    :param S: Signal (2-dim)
    :param n: store n IMFs
    :param L:
    :param max_iters:
    :return:
    """

    if not isinstance(S, np.ndarray):
        S = np.array(S)

    K = min(10, max(5, n))
    filters = initialize_filters(L, K)
    modes = []
    S = np.array(S) if isinstance(S, list) else S
    for i in range(max_iters):
       for filter in filters:
           filtered_signal = lfilter(filter, 1.0, S)
           period = estimate_period(filtered_signal)
           modes.append(filtered_signal)
       if len(modes) >= n:
           break
    return modes[:n]


if __name__ == '__main__':
    signal_df = np.random.rand(1, 100).squeeze()
    time = np.arange(0, signal_df.size, 1)
    n = 5
    modes = fmd(signal_df, n)
    for i, mode in enumerate(modes):
       print(f'Mode {i+1}: Max={np.max(mode)}, Min={np.min(mode)}')
    plt.figure(figsize=(10, 8))
    plt.subplot(len(modes) + 1, 1, 1)
    plt.plot(time, signal_df)
    plt.title('Original Signal')
    for i, mode in enumerate(modes, start=1):
       plt.subplot(len(modes) + 1, 1, i + 1)
       plt.plot(time, mode)
       plt.title(f'Mode {i}')
    plt.tight_layout()
    plt.show()