import pytest
import numpy as np
from src.Modal_Decomposition.Utils.Hilbert import fht_forward, fht_inverse

def test_fht_forward():
    # 1. 自逆性: FHT(FHT(x))/N ≈ x
    x = np.random.randn(1024)
    X = fht_forward(x)
    x_rec = fht_inverse(X)
    assert np.allclose(x_rec, x, atol=1e-10), "FHT 自逆性失败"

    # 2. 与 DHT 定义对照: H[k] = Σ_n x[n] * cas(2πkn/N)
    N = 256
    x = np.random.randn(N)
    X_ref = np.array([
        sum(x[n] * (np.cos(2*np.pi*k*n/N) + np.sin(2*np.pi*k*n/N))
            for n in range(N))
        for k in range(N)
    ])
    X_impl = fht_forward(x, normalize_order=True)
    assert np.allclose(X_impl, X_ref, atol=1e-8), "FHT 与 DHT 定义不符"

    print("FHT 实现正确")