"""
Python version:  (must)
    3.10.11

Lib and Version:  (if None write None)
    scipy - 1.15.3
	matplotlib - 3.10.8
	numpy - 2.2.6

Only accessed by:  (must)
    Only __init__.py

Modify:  (must)
    2026.3.25

Description: (if None write None)
    Realize the FMD
"""

import numpy as np

def initialize_filters(L, K):
    from scipy.signal import firwin

    filters = []
    for k in range(1, K + 1):
       cutoff = 0.5 / k
       filter = firwin(L, cutoff, window='hann')
       filters.append(filter)
    return filters

def estimate_period(signal):
    from scipy.signal import correlate, find_peaks

    correlation = correlate(signal, signal, mode='full')
    correlation = correlation[len(correlation) // 2:]
    peaks, _ = find_peaks(correlation)
    if len(peaks) > 1:
       period = peaks[1]
    else:
       period = len(signal)
    return period

def fmd(S, n=10, L=100, max_iters=10):
    """
    :param S: Signal (2-dim)
    :param n: store n IMFs
    :param L:
    :param max_iters:
    :return:
    """
    from scipy.signal import lfilter

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
