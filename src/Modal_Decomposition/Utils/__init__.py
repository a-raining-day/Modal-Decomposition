from .Check import Check, Check_Time_and_Signal
from .Monotonicity import Monotony, monotonic, give_monotony, MONOTONY_MODS, MONOTONY_MODS_ARGS
from .Hilbert import hilbert, fht_forward, fht_inverse, hilbert_transform
from .Envelope import envelope, ENVELOPE_MODS, ENVELOPE_MODS_ARGS
from .Peaks import (
    find_peaks,
    PEAKS_MODS, PEAKS_MODS_ARGS, PEAKS_BACKEND, PEAKS_BACKEND_ARGS, PEAKS_SUPPORTED_FILTERS, PEAKS_SUPPORTED_FILTERS_ARGS
)
from .Mirror import mirror, MIRROR_MODS, MIRROR_MODS_ARGS
from .Slepian import slepian, resolve_backend, available_backends, cache_info, clear_cache
from .Poly import fast_poly