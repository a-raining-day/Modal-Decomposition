"""
Python version:  (must)
    3.10.11

Lib and Version:  (if None write None)
    numpy - 2.2.6
	scipy - 1.15.3

Only accessed by:  (must)
    Only __init__.py

Description: (if None write None)
    Realize the SVMD. Optimize the use of scipy lib.

Modify:  (must)
    2026.3.25 - Create.
    2026.4.3  - Finish the Optimization of the SVMD. Give two kind of SVMD.
    2026.5.1  - The decomposition with JIT is Error.
"""

import numpy as np
from typing import Tuple, List, Optional, Union


class SVMD:
    """
    SVMD: Sequential Variational Mode Decomposition
    """

    def __init__(
            self,
            num_modes: int = 3,
            alpha: float = 2000,
            tau: float = 0.0,
            tol: float = 1e-7,
            max_iter: int = 500
    ):
        self.num_modes = max(1, int(num_modes))
        self.alpha = float(alpha)
        self.tau = float(tau)
        self.tol = float(tol)
        self.max_iter = int(max_iter)

        self._freqs: Optional[np.ndarray] = None
        self._signal_hat: Optional[np.ndarray] = None
        self._lambda_hat: Optional[np.ndarray] = None
        self._modes_hat: Optional[List[np.ndarray]] = None

    def __call__(self, signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return self.decompose(signal)

    def decompose(self, signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        from scipy.fft import fft, ifft, fftfreq

        signal = np.asarray(signal, dtype=np.float64).ravel()
        N = signal.size
        if N < 8:
            raise ValueError("Signal length must be ≥ 8")

        K = self.num_modes
        eps = 1e-12

        if self._freqs is None or len(self._freqs) != N:
            self._freqs = fftfreq(N, d=1.0)

        self._signal_hat = fft(signal)

        if self._lambda_hat is None or len(self._lambda_hat) != N:
            self._lambda_hat = np.zeros(N, dtype=np.complex128)
        else:
            self._lambda_hat.fill(0.0)  # Reuse memory

        modes = np.zeros((K, N), dtype=np.float64)

        if self._modes_hat is None or len(self._modes_hat) != K or len(self._modes_hat[0]) != N:
            self._modes_hat = [np.zeros(N, dtype=np.complex128) for _ in range(K)]
        else:
            for mh in self._modes_hat:
                mh.fill(0.0)

        freqs_abs = np.abs(self._freqs)
        freq_range = freqs_abs.max() - freqs_abs.min()
        omega = freqs_abs.min() + (np.arange(0.1, 0.4, 0.4 / K)[:K] + 0.1) * freq_range

        denominators = []
        for k in range(K):
            diff = self._freqs - omega[k]
            denom = 1.0 + self.alpha * diff * diff
            denominators.append(denom)

        sum_modes_hat = np.zeros(N, dtype=np.complex128)
        recon_hat = np.zeros(N, dtype=np.complex128)

        prev_recon = np.zeros(N, dtype=np.float64)

        for iteration in range(self.max_iter):
            sum_modes_hat.fill(0.0)

            for k in range(K):
                numerator = self._signal_hat - (sum_modes_hat - self._modes_hat[k]) + 0.5 * self._lambda_hat

                self._modes_hat[k] = numerator / denominators[k]

                sum_modes_hat += self._modes_hat[k]

                power = np.abs(self._modes_hat[k])
                power_sq = power ** 2

                omega_num = np.sum(self._freqs * power_sq)
                omega_den = np.sum(power_sq) + eps

                omega[k] = omega_num / omega_den

                diff = self._freqs - omega[k]
                denominators[k] = 1.0 + self.alpha * diff * diff

            np.copyto(recon_hat, sum_modes_hat)

            self._lambda_hat += self.tau * (self._signal_hat - recon_hat)

            recon_time = np.real(ifft(recon_hat))

            error = np.linalg.norm(recon_time - prev_recon) / (np.linalg.norm(signal) + eps)

            if error < self.tol and iteration > 1:
                break

            np.copyto(prev_recon, recon_time)

        for k in range(K):
            modes[k] = np.real(ifft(self._modes_hat[k]))

        residual = signal - np.sum(modes, axis=0)

        return modes, residual

def give_svmd_JIT():
    from .Error.RealizationError import RealizationError
    raise RealizationError("this method' realization is error.")

    from scipy.fft import fft, ifft, fftfreq
    from numba import jit, prange, complex128, float64
    import numba as nb

    @jit(nopython=True, parallel=False, cache=True)
    def _svmd_iteration_numba(
            signal_hat: np.ndarray,
            freqs: np.ndarray,
            alpha: float,
            tau: float,
            K: int,
            N: int,
            max_iter: int,
            tol: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Numba-accelerated SVMD core iteration
        """
        modes_hat = np.zeros((K, N), dtype=np.complex128)
        lambda_hat = np.zeros(N, dtype=np.complex128)

        freqs_abs = np.abs(freqs)
        freq_range = freqs_abs.max() - freqs_abs.min()
        omega = np.zeros(K, dtype=np.float64)
        for k in range(K):
            omega[k] = freqs_abs.min() + (0.1 + 0.3 * k / max(1, K - 1)) * freq_range

        denominators = np.zeros((K, N), dtype=np.float64)
        for k in range(K):
            for n in range(N):
                diff = freqs[n] - omega[k]
                denominators[k, n] = 1.0 + alpha * diff * diff

        prev_recon = np.zeros(N, dtype=np.float64)
        sum_modes_hat = np.zeros(N, dtype=np.complex128)
        eps = 1e-12

        for it in range(max_iter):
            sum_modes_hat.fill(0 + 0j)

            for k in range(K):
                numerator = signal_hat - (sum_modes_hat - modes_hat[k]) + 0.5 * lambda_hat

                for n in range(N):
                    modes_hat[k, n] = numerator[n] / denominators[k, n]

                for n in range(N):
                    sum_modes_hat[n] += modes_hat[k, n]

                num, den = 0.0, 0.0
                for n in range(N):
                    power = abs(modes_hat[k, n])
                    power_sq = power * power
                    num += freqs[n] * power_sq
                    den += power_sq

                omega[k] = num / (den + eps)

                for n in range(N):
                    diff = freqs[n] - omega[k]
                    denominators[k, n] = 1.0 + alpha * diff * diff

            for n in range(N):
                lambda_hat[n] += tau * (signal_hat[n] - sum_modes_hat[n])

            recon_error = 0.0
            sig_norm = 0.0
            for n in range(N):
                re = np.real(sum_modes_hat[n]) - prev_recon[n]
                recon_error += re * re
                sig_norm += np.sum(np.abs(signal_hat)**2) / N

            if it > 1 and np.sqrt(recon_error) / (np.sqrt(sig_norm) + eps) < tol:
                break

            for n in range(N):
                prev_recon[n] = np.real(sum_modes_hat[n])

        modes = np.zeros((K, N), dtype=np.float64)
        for k in range(K):
            for n in range(N):
                modes[k, n] = np.real(modes_hat[k, n])

        residual = np.zeros(N, dtype=np.float64)
        for n in range(N):
            recon_val = 0.0
            for k in range(K):
                recon_val += modes[k, n]
            residual[n] = np.real(signal_hat[n]) - recon_val

        return modes, residual

    class NumbaSVMD(SVMD):

        def decompose(self, signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            signal = np.asarray(signal, dtype=np.float64).ravel()
            N = signal.size
            if N < 8:
                raise ValueError("Signal length must be ≥ 8")

            signal_hat = fft(signal)
            freqs = fftfreq(N, d=1.0)

            modes, residual = _svmd_iteration_numba(
                signal_hat=signal_hat,
                freqs=freqs,
                alpha=self.alpha,
                tau=self.tau,
                K=self.num_modes,
                N=N,
                max_iter=self.max_iter,
                tol=self.tol
            )

            return modes, residual

    return NumbaSVMD

def svmd \
    (
        S: Union[list, np.ndarray],
        use_JIT: bool = False,
        num_modes: int = 3,
        alpha: float = 2000,
        tau: float = 0.0,
        tol: float = 1e-7,
        max_iter: int = 500
    ) -> Tuple[np.ndarray, np.ndarray]:
    """

    :param S: Signal (N,)
    :param use_JIT: if True, use Numba version.
    :param num_modes: the num of modes
    :param alpha:
    :param tau:
    :param tol: epsilon
    :param max_iter: the max num of the iterations.
    :return: IMFs(num_modes, N), Res(N,)
    """

    if not use_JIT:
        Cls = SVMD(num_modes=num_modes, alpha=alpha, tau=tau, tol=tol, max_iter=max_iter)
        IMFs, Res = Cls(S)

    else:
        Module = give_svmd_JIT()
        Cls = Module(num_modes=num_modes, alpha=alpha, tau=tau, tol=tol, max_iter=max_iter)
        IMFs, Res = Cls(S)

    return IMFs, Res

if __name__ == "__main__":
    import time
    import matplotlib.pyplot as plt

    np.random.seed(42)
    t = np.linspace(0, 1, 1000)
    signal = (
            2.0 * np.cos(2 * np.pi * 5 * t) +  # 5Hz
            1.5 * np.cos(2 * np.pi * 12 * t) +  # 12Hz
            1.0 * np.cos(2 * np.pi * 20 * t) +  # 20Hz
            0.1 * np.random.randn(len(t))
    )

    f = SVMD(num_modes=3, alpha=2000, tol=1e-7, max_iter=200)

    start = time.time()
    modes, residual = f(signal)
    end = time.time()

    print(f"the time of the optimized : {end - start:.4f}s")
    print(f"the shape of Modal: {modes.shape}")
    print(f"energy of Res: {np.sum(residual ** 2):.6f}")

    plt.figure(figsize=(12, 8))
    plt.subplot(4, 1, 1)
    plt.plot(t, signal, label="Origin")
    plt.legend()

    for i in range(modes.shape[0]):
        plt.subplot(4, 1, i + 2)
        plt.plot(t, modes[i], label=f"Mode {i + 1}")
        plt.legend()

    plt.subplot(4, 1, 4)
    plt.plot(t, residual, label="Res", color='red')
    plt.legend()
    plt.tight_layout()
    plt.show()