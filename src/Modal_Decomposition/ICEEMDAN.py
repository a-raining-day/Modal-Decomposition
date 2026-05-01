"""

Python version:  (must)
    3.10.11

Lib and Version:  (if None write None)
    numpy - 2.2.6

Only accessed by:  (must)
    Only __init__.py

Description: (if None write None)
    Realize the ICEEMDAN

Modify:  (must)
    2026.3.25 - Create
    2026.4.3  -
"""

import numpy as np
from typing import Union, Tuple, Optional
from .EMD import emd
from .Utils import is_monotonic, Check_Time_and_Signal


def iceemdan(
        signal: Union[np.ndarray, list],
        time_axis: Optional[Union[np.ndarray, list]] = None,
        ensemble_size: int = 300,
        epsilon_0: float = 0.2,
        max_imfs: Optional[int] = None,
        spline_kind: str = "cubic",
        nbsym: int = 2,
        rng_seed: Optional[int] = None,
        verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray, None]:
    """
    ICEEMDAN: Improved Complete Ensemble EMD with Adaptive Noise

    Reference: Colominas, M. A., Schlotthauer, G., & Torres, M. E. (2014).
               Improved complete ensemble EMD: A suitable tool for
               biomedical signal processing. Biomedical Signal Processing
               and Control, 14, 19-29.)
    """

    # ===================== 1. INPUT VALIDATION =====================
    # Signal validation
    signal, _, _ = Check_Time_and_Signal(signal)

    signal = np.asarray(signal, dtype=np.float64)
    if signal.ndim != 1:
        signal = signal.ravel()

    n_samples = len(signal)
    if n_samples < 4:
        raise ValueError("Signal length must be at least 4")

    # Time axis validation
    if time_axis is None:
        time_axis = np.arange(n_samples, dtype=np.float64)
    else:
        time_axis = np.asarray(time_axis, dtype=np.float64)
        if len(time_axis) != n_samples:
            raise ValueError(f"Time axis length mismatch: {len(time_axis)} != {n_samples}")

    # Check uniform sampling
    if n_samples > 1:
        dt = np.diff(time_axis)
        if not np.allclose(dt, dt[0], rtol=1e-10, atol=1e-14):
            raise ValueError("Time axis must be uniformly sampled")

    # Max IMFs
    if max_imfs is None:
        max_imfs = int(np.log2(n_samples)) + 5
    elif max_imfs <= 0:
        raise ValueError(f"max_imfs must be positive, got {max_imfs}")

    # Parameters validation
    if ensemble_size < 1:
        raise ValueError(f"ensemble_size must be >= 1, got {ensemble_size}")

    if epsilon_0 <= 0:
        raise ValueError(f"epsilon_0 must be > 0, got {epsilon_0}")

    # Random seed for reproducibility
    if rng_seed is not None:
        np.random.seed(rng_seed)

    if verbose:
        print(f"ICEEMDAN: N={n_samples}, M={ensemble_size}, ε₀={epsilon_0}, max_imfs={max_imfs}")

    # ===================== 2. PRE-COMPUTE NOISE IMFs =====================
    if verbose:
        print("1. Generating and decomposing white noise...")

    # Generate fixed pool of white noise
    white_noise = np.random.randn(ensemble_size, n_samples)

    # Pre-decompose all noise realizations
    noise_imfs_list = []  # List of 2D arrays: (K, n_samples) for each noise
    valid_indices = []

    for i in range(ensemble_size):
        try:
            imfs, _ = emd(
                white_noise[i],
                time_axis,
                max_imf=max_imfs + 5,  # Decompose 5 more IMFs than needed
                spline_kind=spline_kind,
                nbsym=nbsym)

            if imfs.shape[0] > 0:
                noise_imfs_list.append(imfs)
                valid_indices.append(i)
        except Exception as e:
            if verbose and i < 5:  # Show first few errors only
                print(f"  Noise {i} decomposition failed: {e}")
            continue

    # Check valid noise count
    M = len(valid_indices)
    if M == 0:
        raise RuntimeError("All noise decompositions failed")

    if verbose:
        print(f"  Valid noise realizations: {M}/{ensemble_size} ({M / ensemble_size * 100:.1f}%)")

    # Determine common number of IMFs across all noise realizations
    min_noise_imfs = min(imfs.shape[0] for imfs in noise_imfs_list)
    K = min(min_noise_imfs, max_imfs)  # Actual maximum number of IMFs to extract

    if K == 0:
        raise RuntimeError("No valid IMFs found in noise decompositions")

    if verbose:
        print(f"  Maximum IMFs to extract: {K}")

    # Compute noise IMF standard deviations (global std, as per paper)
    if verbose:
        print("2. Computing noise IMF statistics...")

    noise_std = np.zeros(K, dtype=np.float64)
    for k in range(K):
        # Stack all k-th IMFs from all noise realizations
        kth_imfs = np.stack([imfs[k] for imfs in noise_imfs_list])
        # Compute global standard deviation (across all realizations and samples)
        noise_std[k] = np.std(kth_imfs)

    # ===================== 3. ICEEMDAN DECOMPOSITION =====================
    if verbose:
        print("3. Performing ICEEMDAN decomposition...")

    residual = signal.copy()
    imfs_list = []
    signal_energy = np.sum(signal ** 2)
    if signal_energy < 1e-12:
        signal_energy = 1e-12  # Prevent division by zero

    for k in range(K):
        if verbose:
            print(f"  Extracting IMF {k + 1}/{K}...")

        # Noise scaling factor: β_k = ε₀ * std(r_{k-1}) / std(E_k(ω))
        beta = epsilon_0 * np.std(residual) / (noise_std[k] + 1e-12)
        beta = np.clip(beta, 1e-12, 1e12)  # Numerical stability

        # Pre-allocate array for local means
        local_means = np.zeros((M, n_samples), dtype=np.float64)
        valid_count = 0

        # Ensemble loop
        for i, noise_imfs in enumerate(noise_imfs_list):
            try:
                # Noisy signal: r_{k-1} + β_k * E_k(ω^(i))
                noisy_signal = residual + beta * noise_imfs[k]

                # Apply E₁ operator: decompose and get first IMF
                sig_imfs, _ = emd(
                    noisy_signal,
                    time_axis,
                    max_imf=1,  # Only need the first IMF
                    spline_kind=spline_kind,
                    nbsym=nbsym
                )

                if sig_imfs.shape[0] > 0:
                    # Local mean: M(·) = signal - E₁(·)
                    local_means[valid_count] = noisy_signal - sig_imfs[0, :]
                    valid_count += 1

            except Exception as e:
                if verbose and valid_count == 0:  # Only show first error
                    print(f"    Warning: IMF {k + 1}, noise {i} failed: {e}")
                continue

        if valid_count == 0:
            if verbose:
                print(f"  No valid realizations for IMF {k + 1}, stopping")
            break

        # Compute ensemble average of local means
        r_k = np.mean(local_means[:valid_count], axis=0)

        # Extract IMF: IMF_k = r_{k-1} - r_k
        imf_k = residual - r_k

        # Check IMF validity
        imf_energy = np.sum(imf_k ** 2)
        if imf_energy < 1e-12 * signal_energy:
            if verbose:
                print(f"  IMF {k + 1} has negligible energy, stopping")
            break

        imfs_list.append(imf_k)

        # Update residual: r_k
        residual = r_k

        # Stopping criteria (original paper conditions)
        if is_monotonic(residual):
            if verbose:
                print(f"  Residual is monotonic, stopping")
            break

        if len(imfs_list) >= max_imfs:
            if verbose:
                print(f"  Reached maximum IMF count, stopping")
            break

    # ===================== 4. OUTPUT AND VALIDATION =====================
    if not imfs_list:
        if verbose:
            print("Warning: No IMFs were extracted")
        return np.empty((0, n_samples), dtype=np.float64), residual

    imfs_array = np.vstack(imfs_list)

    # Validate reconstruction
    reconstruction = np.sum(imfs_array, axis=0) + residual
    reconstruction_error = np.max(np.abs(signal - reconstruction))
    relative_error = reconstruction_error / (np.max(np.abs(signal)) + 1e-12)

    if verbose:
        print("Decomposition complete!")
        print(f"   Extracted {len(imfs_list)} IMFs")
        print(f"   Maximum reconstruction error: {reconstruction_error:.2e}")
        print(f"   Relative reconstruction error: {relative_error:.2e}")

        if relative_error > 1e-8:
            print("   Warning: Reconstruction error is relatively high")

    return imfs_array, residual, None

def validate_iceemdan_decomposition(
        signal: np.ndarray,
        imfs: np.ndarray,
        residual: np.ndarray,
        tol: float = 1e-8
) -> dict:
    """
    Validate ICEEMDAN decomposition quality

    Parameters
    ----------
    signal : ndarray
        Original signal
    imfs : ndarray
        Extracted IMFs
    residual : ndarray
        Residual
    tol : float
        Tolerance for reconstruction error

    Returns
    -------
    dict
        Validation results
    """
    # Reconstruction error
    reconstruction = np.sum(imfs, axis=0) + residual
    abs_error = np.max(np.abs(signal - reconstruction))
    rel_error = abs_error / (np.max(np.abs(signal)) + 1e-12)

    # IMF statistics
    imf_stats = []
    for i, imf in enumerate(imfs):
        imf_stats.append({
            'index': i + 1,
            'mean': float(np.mean(imf)),
            'std': float(np.std(imf)),
            'energy': float(np.sum(imf ** 2)),
            'zero_crossings': int(np.sum(np.diff(np.sign(imf)) != 0)),
            'is_monotonic': bool(is_monotonic(imf))
        })

    return {
        'reconstruction_error': {
            'absolute': float(abs_error),
            'relative': float(rel_error)
        },
        'n_imfs': len(imfs),
        'imf_properties': imf_stats,
        'is_valid': rel_error < tol
    }