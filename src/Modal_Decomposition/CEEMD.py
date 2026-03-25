"""
Python version:  (must)
    3.10.11

Lib and Version:  (if None write None)
    numpy - 2.2.6
	time - 1.39.2
	tqdm - 4.67.3

Only accessed by:  (must)
    Only __init__.py

Modify:  (must)
    2026.3.25

Description: (if None write None)
    Realize the CEEMD.
"""

from time import sleep
import numpy as np
from .EMD import emd
from .help_function import is_increasing
from tqdm import tqdm
import sys


def ceemd(S, T=None, fs=None, beta=0.3, max_imf=None, iterations=30, verbose: bool=False):
    """

    :param S:
    :param T:
    :param fs:
    :param beta:
    :param max_imf: -1 or other int | 100 is enough, if you choose -1, no one kown when it will finish(^_^)
    :param iterations:
    :param verbose:
    :return:
    """
    N = len(S)

    if max_imf is None:
        max_imf = int(np.log2(len(S))) + 2

    if T is None:
        if fs is not None:
            # smaple
            dt = 1.0 / fs
            T = np.arange(N) * dt
        else:
            # default smaple f = 1
            T = np.arange(N)
            print(f"Warn: T is None，default T = [0, 1, 2, ..., {N - 1}]")

    # Standard
    MEAN = np.mean(S)
    S = S - MEAN
    STD = np.std(S)
    S = S / STD

    Res = S
    IMFs = []

    std_dev = np.std(S)
    mean = np.mean(S)

    count = 0
    while True:
        IMF = np.zeros(N)

        if verbose:
            for i in tqdm(range(iterations // 2), position=0, file=sys.stdout, desc="iter: "):
                white_noise = np.random.normal(mean, std_dev * beta, N)

                S_plus = Res + white_noise
                S_minus = Res - white_noise

                IMFs_plus, _ = emd(S_plus, T)
                IMFs_minus, _ = emd(S_minus, T)

                if len(IMFs_plus) == 0 or len(IMFs_minus) == 0:
                    continue

                IMFs_avg = (IMFs_plus[0, :] + IMFs_minus[0, :]) / 2

                IMF += IMFs_avg

        else:
            for i in range(iterations // 2):
                white_noise = np.random.normal(mean, std_dev * beta, N)

                S_plus = Res + white_noise
                S_minus = Res - white_noise

                IMFs_plus, _ = emd(S_plus, T)
                IMFs_minus, _ = emd(S_minus, T)

                if len(IMFs_plus) == 0 or len(IMFs_minus) == 0:
                    continue

                IMFs_avg = (IMFs_plus[0, :] + IMFs_minus[0, :]) / 2

                IMF += IMFs_avg

        IMF = IMF / (iterations / 2)
        IMFs.append(IMF * STD + MEAN)
        count += 1

        if verbose:
            if count % 10 == 0:
                tqdm.write(f"has get {count} IMFs...")
                sleep(1)

        if max_imf != -1:
            if count >= max_imf:
                Res = Res - IMF
                Res = Res * STD + MEAN
                break

        Res = Res - IMF

        if np.std(Res) < 1e-10 * np.std(S):
            break

        if is_increasing(Res):
            Res = Res * STD + MEAN
            break

    return np.array(IMFs), np.array(Res)