"""
Python version:  (must)
    3.10.11

Lib and Version:  (if None write None)
    EMD-signal - 1.9.0

Only accessed by:  (must)
    Only __init__.py

Modify:  (must)
    2026.3.25

Description: (if None write None)
    Realize the CEEFD and CEEMDAN.
"""

import numpy as np

class CEEFD:
    """
    CEEFD
    """

    def __init__ \
        (
            self,
            trials=10,
            noise_width=0.05,  # default: 0.05-0.3
            noise_seed=42,  # seed
            spline_kind='cubic',
            nbsym=2,  # Number of boundary symmetry points
            extrema_detection='parabol',
            parallel=False,
            processes=None,  # None = auto, int >= 1
            random_state=42,
            noise_scale=1.0,  # scale factor of noise
            noise_kind='normal',  # noise kind: 'normal', 'uniform'
            range_thr=0.01,  # Stop threshold
            total_power_thr=0.005
        ):

        self.trials = trials
        self.noise_width = noise_width
        self.noise_seed = noise_seed
        self.spline_kind = spline_kind
        self.nbsym = nbsym
        self.extrema_detection = extrema_detection
        self.parallel = parallel
        self.processes = processes
        self.random_state = random_state
        self.noise_scale = noise_scale
        self.noise_kind = noise_kind
        self.range_thr = range_thr
        self.total_power_thr = total_power_thr

    def give_ceemdan(self):
        from PyEMD import CEEMDAN

        return CEEMDAN \
            (
                trials=self.trials,
                noise_width=self.noise_width,
                noise_seed=self.noise_seed,
                spline_kind=self.spline_kind,
                nbsym=self.nbsym,
                extrema_detection=self.extrema_detection,
                parallel=self.parallel,
                processes=self.processes,
                random_state=self.random_state,
                noise_scale=self.noise_scale,
                noise_kind=self.noise_kind,
                range_thr=self.range_thr,
                total_power_thr=self.total_power_thr
            )

    def ceemdan(self, S, T=None, max_imf=-1):
        CEEMDAN = self.give_ceemdan()
        IMF_Residue = CEEMDAN.ceemdan(S, T, max_imf)

        IMFs = IMF_Residue[:-1, :]  # shape [n_imfs, len(S)]
        Res = IMF_Residue[-1, :]

        return IMFs, Res

    def ceefd(self, S, T=None, max_imf=-1):
        """
        :return: other_IMFs (ndarray-2D), IMF_ (np.ndarray-2D), Res (ndarray), Res_ (ndarray)
        """
        from .EFD import EFD
        import antropy as ant

        CEEMDAN = self.give_ceemdan()
        IMF_Residue = CEEMDAN.ceemdan(S, T, max_imf)

        IMFs = IMF_Residue[:-1, :]  # shape [n_imfs, len(S)]
        Res = IMF_Residue[-1, :]

        # cal sample entropy of IMF
        Entropy = [ant.sample_entropy(IMF) for IMF in IMFs]
        max_entropy_mask = np.argmax(Entropy)
        maxIMF = IMFs[max_entropy_mask]

        # EMD for max antropy IMF
        if T is None:
            T_efd = np.arange(len(maxIMF))
        else:
            T_efd = T
        IMF_, Res_ = EFD(maxIMF, T_efd)

        # get other_IMFs
        other_IMFs_list = [IMF for i, IMF in enumerate(IMFs) if i != max_entropy_mask]
        if other_IMFs_list:
            other_IMFs = np.stack(other_IMFs_list)
        else:
            other_IMFs = None

        return other_IMFs, np.array(IMF_), Res, Res_