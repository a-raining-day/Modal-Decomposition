"""
Native VMD regression tests.

Locks the invariants established while refactoring ``VMD.py`` (memmap through
``Utils.Chunk``, mirror through ``Utils.Mirror``, helpers inside the class,
``Literal`` string modes, ``DC: bool``):

* the three storage engines agree to round-off;
* ``engine="auto"`` picks ``ram`` in budget and ``chunked`` (+ out-of-core) when
  the projected working set trips ``Utils.Memory``;
* out-of-core temp files are released when the run ends;
* history / DC / parameter validation behave as documented;
* the vmdpy branch reproduces the native solver.

Note on vmdpy parity: ``vmdpy.VMD`` has no iteration argument (it hard-codes
``Niter = 500``), so comparisons must use ``n >= 500`` — a smaller ``n`` only
truncates the *native* side and the difference is an iteration-budget artefact,
not an implementation difference.
"""

import glob
import os
import tempfile

import numpy as np
import pytest

from Modal_Decomposition import Class
from Modal_Decomposition.Utils import Check_Time_and_Signal

BASE = dict(num_imf=3, n=500, alpha=2000.0, tau=0.0, epsilon=1e-7,
            DC=False, init_mod="uniform", seed=11)


def cfg(**over):
    """基准参数 + 覆盖项 (避免同一关键字重复传入)。"""
    return {**BASE, **over}


def tone(N, seed=0):
    """三音 + 噪声: 有明确的 3 个频带, 适合 K=3。"""
    rng = np.random.default_rng(seed)
    t = np.arange(N) / N
    x = (np.sin(2 * np.pi * 5 * t) + 0.6 * np.cos(2 * np.pi * 23 * t)
         + 0.3 * np.sin(2 * np.pi * 61 * t))
    return x + 0.05 * rng.standard_normal(N)


@pytest.mark.parametrize("N", [256, 1024, 3001])
@pytest.mark.parametrize("chunk", [64, 257, None])
def test_engines_agree(N, chunk):
    """ram / chunked / chunked+out_of_core 的差异只能来自求和顺序 (~1e-14)。"""
    x = tone(N)
    base = Class.VMD(engine="ram", **BASE).decompose(x)
    kw = {} if chunk is None else {"chunk_size": chunk}
    ch = Class.VMD(engine="chunked", **kw, **BASE).decompose(x)
    ooc = Class.VMD(engine="chunked", out_of_core=True, **kw, **BASE).decompose(x)
    assert np.max(np.abs(base.IMFs - ch.IMFs)) < 1e-12
    assert np.max(np.abs(base.IMFs - ooc.IMFs)) < 1e-12
    assert np.max(np.abs(base.info["omega"] - ch.info["omega"])) < 1e-12


def test_auto_engine_in_budget_uses_ram():
    res = Class.VMD(**BASE).decompose(tone(1024))
    assert res.info["engine"] == "ram"
    assert res.info["out_of_core"] is False


def test_auto_engine_under_pressure_uses_chunked_out_of_core(monkeypatch):
    """预算判定为"该落盘"时, auto 必须走 chunked + out_of_core, 数值不变。"""
    import Modal_Decomposition.VMD as vmd_mod

    monkeypatch.setattr(vmd_mod, "should_use_memmap", lambda *a, **k: True)
    x = tone(1024)
    res = Class.VMD(**BASE).decompose(x)
    assert res.info["engine"] == "chunked"
    assert res.info["out_of_core"] is True
    base = Class.VMD(engine="ram", **BASE).decompose(x)
    assert np.max(np.abs(base.IMFs - res.IMFs)) < 1e-12


def test_out_of_core_releases_temp_files():
    """外存工作区跑完即删 (不依赖 atexit 兜底)。"""
    pattern = os.path.join(tempfile.gettempdir(), "md_store_*.dat")
    before = set(glob.glob(pattern))
    Class.VMD(engine="chunked", out_of_core=True, chunk_size=64, **BASE).decompose(tone(1024))
    assert set(glob.glob(pattern)) - before == set()


def test_failed_temp_memmap_leaves_no_file():
    """``Chunk.temp_memmap`` 建图失败时不把空文件遗弃在临时目录。"""
    from Modal_Decomposition.Utils.Chunk import temp_memmap

    pattern = os.path.join(tempfile.gettempdir(), "md_store_*.dat")
    before = set(glob.glob(pattern))
    with pytest.raises(TypeError):
        temp_memmap(np.float64, (16,))      # 参数顺序写错: shape 位给了 dtype
    assert set(glob.glob(pattern)) - before == set()


@pytest.mark.parametrize("N", [64, 512, 2048])
def test_ram_matches_plain_numpy_reference(N):
    """engine='ram' 与"镜像 + 单边谱 + 顺序 ADMM"的朴素参考实现一致。"""
    x = tone(N).astype(np.float64)
    K, n, alpha, tau, tol = 3, 60, 2000.0, 0.0, 1e-7
    res = Class.VMD(engine="ram", num_imf=K, n=n, alpha=alpha, tau=tau,
                    epsilon=tol, DC=False, init_mod="uniform").decompose(x)

    left = N // 2
    f = np.concatenate([x[:left][::-1], x, x[N - left:][::-1]])
    T = 2 * N
    f_hat = np.fft.fft(f)
    f_hat_plus = f_hat[:N]
    freqs = np.arange(N, dtype=np.float64) / T
    omega = (0.5 / K) * np.arange(K, dtype=np.float64)
    u_hat = np.zeros((K, N), dtype=np.complex128)
    lam = np.zeros(N, dtype=np.complex128)
    for _ in range(n):
        udiff = 0.0
        for k in range(K):
            resid = f_hat_plus - (u_hat.sum(axis=0) - u_hat[k]) - lam / 2.0
            denom = alpha * (freqs - omega[k]) ** 2 + 1.0
            new = resid / denom
            udiff += float(np.sum(np.abs(new - u_hat[k]) ** 2))
            u_hat[k] = new
            p = np.abs(new) ** 2
            if p.sum() > 0:
                omega[k] = float(np.dot(freqs, p) / p.sum())
        lam = lam + tau * (u_hat.sum(axis=0) - f_hat_plus)
        if udiff / T <= tol:
            break
    half = np.zeros((K, N + 1), dtype=np.complex128)
    half[:, :N] = u_hat
    modes = np.array([np.fft.irfft(half[k], n=T)[left:left + N] for k in range(K)])
    assert np.max(np.abs(modes - res.IMFs)) < 1e-12
    assert np.max(np.abs(omega - res.info["omega"])) < 1e-12


def test_store_history():
    res = Class.VMD(store_history=True, **BASE).decompose(tone(512))
    uh = np.asarray(res.info["u_hat_history"])
    ud = np.asarray(res.info["udiff_history"])
    assert uh.ndim == 3 and uh.shape[1] == BASE["num_imf"] and uh.shape[2] == 512
    assert uh.shape[0] == ud.shape[0] == res.info["n_iter"]

    plain = Class.VMD(store_history=False, **BASE).decompose(tone(512))
    assert plain.info.get("u_hat_history") is None
    assert plain.info.get("udiff_history") is None


def test_dc_keeps_first_mode_at_zero():
    res = Class.VMD(**cfg(DC=True)).decompose(tone(512))
    assert float(res.info["omega"][0]) == 0.0
    res = Class.VMD(**cfg(DC=False)).decompose(tone(512))
    assert float(res.info["omega"][0]) != 0.0


def test_vmdpy_branch_matches_native():
    pytest.importorskip("vmdpy")
    x = tone(2048)
    for init in ("zero", "uniform"):
        nat = Class.VMD(**cfg(engine="ram", init_mod=init)).decompose(x)
        ref = Class.VMD(**cfg(vmdpy=True, init_mod=init)).decompose(x)
        assert nat.info["n_iter"] == ref.info["n_iter"]
        assert np.max(np.abs(nat.IMFs - ref.IMFs)) < 1e-4
        assert np.max(np.abs(nat.info["omega"] - ref.info["omega"])) < 1e-4


def test_vmdpy_rejects_native_only_init():
    pytest.importorskip("vmdpy")
    with pytest.raises(ValueError, match="vmdpy"):
        Class.VMD(vmdpy=True, init_mod="peak").decompose(tone(512))


@pytest.mark.parametrize("bad", [
    {"init_mod": "bogus"}, {"engine": "bogus"}, {"DC": 1}, {"num_imf": 0},
    {"n": 0}, {"alpha": 0.0}, {"chunk_size": 0},
])
def test_invalid_parameters(bad):
    with pytest.raises((ValueError, TypeError)):
        Class.VMD(**{**BASE, **bad})


def test_short_signal_rejected():
    with pytest.raises(ValueError, match="length"):
        Class.VMD(**BASE).decompose(np.arange(4.0))


def test_accepts_check_time_and_signal_output():
    S, T, N = Check_Time_and_Signal(tone(600))
    res = Class.VMD(**BASE).decompose(S, T)
    assert res.IMFs.shape == (BASE["num_imf"], N)
    assert res.IMFs.shape[1] == S.shape[0]
