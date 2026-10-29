"""EFD / FMD 本轮改动的回归测试（分段口径、求解器选择、默认成本旋钮）。

* EFD：分段规则 = 保留 `max_IMFs` 个**最大**局部极大（Segm_tec），频带互不重叠；
  `max_IMFs=-1` 返回全部频带；重构恒等式成立；双音信号能被分开。
  频带数 **≤** `max_IMFs`（相邻候选之间的幅值最小处可能与端点重合而被去重），
  因此只断言上界与可分离性，不硬编码"恰好等于"。
* FMD：`solver` ∈ {"ck","mckd"}、非法值报错、两种求解器结果不同但契约一致；
  `L_factor` 默认 2；`solver` 进入 `FMDConfig` 快照。
"""

from __future__ import annotations

import numpy as np
import pytest

import Modal_Decomposition as MD
from Modal_Decomposition import Class, Function
from Modal_Decomposition.Define import DecompositionResult
from Modal_Decomposition.FMD import FMDConfig

FS = 1000.0


def two_tone(n=2048, noise=0.1):
    """37 Hz + 113 Hz 双音（带弱噪声，与基准 signals.py 的 case A 同构）。"""
    t = np.arange(n) / FS
    a = np.sin(2 * np.pi * 37.0 * t)
    b = 0.5 * np.sin(2 * np.pi * 113.0 * t)
    rng = np.random.default_rng(0)
    return a + b + noise * rng.standard_normal(n), a, b


def _corr(x, y):
    x = x - x.mean()
    y = y - y.mean()
    den = np.linalg.norm(x) * np.linalg.norm(y)
    return abs(float(np.dot(x, y))) / den if den else 0.0


# --------------------------------------------------------------------------- #
# EFD：分段口径
# --------------------------------------------------------------------------- #
def test_efd_separates_two_tones():
    S, a, b = two_tone()
    res = Class.EFD(max_IMFs=3).decompose(S)
    assert 1 <= res.IMFs.shape[0] <= 3
    assert max(_corr(a, m) for m in res.IMFs) > 0.95
    assert max(_corr(b, m) for m in res.IMFs) > 0.90


def test_efd_reconstruction_identity():
    S, _, _ = two_tone(1024)
    res = Class.EFD(max_IMFs=3).decompose(S)
    assert np.max(np.abs(res.IMFs.sum(axis=0) + res.Res - S)) < 1e-10


def test_efd_bands_do_not_overlap():
    """频带互不重叠 ⇒ 各阶能量之和不超过信号能量（高频尾巴由残差吸收）。"""
    S, _, _ = two_tone(1024)
    res = Class.EFD(max_IMFs=3).decompose(S)
    assert float(np.sum(res.IMFs ** 2)) <= float(np.sum(S ** 2)) * (1 + 1e-9)


@pytest.mark.parametrize("max_imfs", [1, 2, 3, 4])
def test_efd_mode_count_within_request(max_imfs):
    S, _, _ = two_tone(1024)
    res = Class.EFD(max_IMFs=max_imfs).decompose(S)
    assert 1 <= res.IMFs.shape[0] <= max_imfs


def test_efd_all_bands_when_minus_one():
    S, _, _ = two_tone(1024)
    res = Class.EFD(max_IMFs=-1).decompose(S)
    assert res.IMFs.shape[0] >= 3
    assert np.max(np.abs(res.IMFs.sum(axis=0) + res.Res - S)) < 1e-10


# --------------------------------------------------------------------------- #
# FMD：solver 选择
# --------------------------------------------------------------------------- #
def test_fmd_solver_default_is_ck():
    assert Class.FMD().solver == "ck"
    assert Class.FMD().config_generate().solver == "ck"


@pytest.mark.parametrize("solver", ["ck", "mckd"])
def test_fmd_solver_runs_and_contract(solver):
    S, _, _ = two_tone(1024)
    MD.set_seed(0, verbose=False)
    res = Class.FMD(K=2, max_iter=5, solver=solver).decompose(S)
    assert isinstance(res, DecompositionResult)
    assert res.IMFs.shape[0] == 2 and res.IMFs.shape[1] == S.size
    assert np.max(np.abs(res.IMFs.sum(axis=0) + res.Res - S)) < 1e-10
    assert res.config.solver == solver


def test_fmd_solver_invalid_value():
    with pytest.raises(ValueError, match="solver"):
        Class.FMD(solver="nope")


def test_fmd_solvers_differ():
    """两种求解器走不同的更新，结果不应逐位相同。"""
    S, _, _ = two_tone(1024)
    MD.set_seed(0, verbose=False)
    a = Class.FMD(K=2, max_iter=5, seed=0, solver="ck").decompose(S).IMFs
    MD.set_seed(0, verbose=False)
    b = Class.FMD(K=2, max_iter=5, seed=0, solver="mckd").decompose(S).IMFs
    assert not np.array_equal(np.asarray(a), np.asarray(b))


def test_fmd_solver_facade_forwarding():
    S, _, _ = two_tone(512)
    MD.set_seed(0, verbose=False)
    a = Class.FMD(K=1, max_iter=3, seed=0, solver="mckd").decompose(S).IMFs
    MD.set_seed(0, verbose=False)
    b = Function.FMD(S, K=1, max_iter=3, seed=0, solver="mckd").IMFs
    assert np.array_equal(np.asarray(a), np.asarray(b))


def test_fmd_config_snapshot_fields():
    cfg = Class.FMD(K=2, solver="mckd").config_generate()
    assert isinstance(cfg, FMDConfig)
    assert (cfg.solver, cfg.K, cfg.L_factor) == ("mckd", 2, 2)


def test_fmd_default_l_factor_is_two():
    """成本旋钮默认值：L = 2*M（见 FMD 报告的时间-质量扫描）。"""
    assert Class.FMD().L_factor == 2


def test_fmd_l_factor_validation():
    with pytest.raises(ValueError):
        Class.FMD(L_factor=0)
    with pytest.raises(TypeError):
        Class.FMD(L_factor=2.0)
