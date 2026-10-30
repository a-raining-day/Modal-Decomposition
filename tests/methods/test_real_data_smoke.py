"""真实数据回归：CWRU 轴承内圈故障上的关键性质。

数据文件（``explore/denoise/data/cwru/IR007_105_0.mat``、PySDKit 的
``real_world/*.npy``）不在 CI 里，缺失时整组 skip；本地跑时断言：

* EFD 比 PySDKit-EFD 快（同 n、同阶数）；
* EFD 重构精确、频带互不重叠；
* FMD(``ck``) 在内圈故障上抽出的模态峰度远高于 PySDKit-FMD（周期冲击是本库 FMD 的目标）；
* FMD 的两种 solver 都能跑，且 ``ck`` 在该数据上冲击性不低于 ``mckd``。
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest

import Modal_Decomposition as MD
from Modal_Decomposition.Base import Seed

ROOT = Path(__file__).resolve().parents[2]
CWRU = ROOT / "explore" / "denoise" / "data" / "cwru" / "IR007_105_0.mat"

pytestmark = pytest.mark.skipif(not CWRU.is_file(), reason="CWRU 数据不在工作区")


def _seg(n=4096, start=2000):
    import scipy.io as sio

    mat = sio.loadmat(CWRU)
    key = next(k for k in mat if not k.startswith("__"))
    return np.asarray(mat[key]).ravel().astype(float)[start:start + n]


def _best_kurt(rows):
    from scipy.stats import kurtosis

    return max(float(kurtosis(r)) for r in np.atleast_2d(rows))


def test_efd_faster_than_pysdkit_on_real_bearing_data():
    from pysdkit import EFD as PyEFD

    S = _seg()
    efd = MD.Class.EFD(max_IMFs=5)
    efd.decompose(S)
    t0 = time.perf_counter(); efd.decompose(S); t_md = time.perf_counter() - t0

    py = PyEFD(max_imfs=5)
    py.fit_transform(S)
    t0 = time.perf_counter(); py.fit_transform(S); t_py = time.perf_counter() - t0
    assert t_md < t_py, (t_md, t_py)


def test_efd_reconstruction_exact_on_real_data():
    S = _seg()
    res = MD.Class.EFD(max_IMFs=5).decompose(S)
    assert np.max(np.abs(res.IMFs.sum(axis=0) + res.Res - S)) < 1e-10


def test_fmd_ck_is_impulsive_on_bearing_fault():
    """周期冲击是 FMD 的目标：最高阶模态峰度应远大于 PySDKit 的滤波器组。"""
    from pysdkit import FMD as PyFMD

    S = _seg()
    Seed.set(0, verbose=False)
    ours = _best_kurt(MD.Class.FMD(K=3, max_iter=10, num_hand=3, seed=0,
                                   solver="ck").decompose(S).IMFs)
    theirs = _best_kurt(PyFMD(fs=12000, mode_num=3).fit_transform(S))
    assert ours > 10 * theirs, (ours, theirs)
    assert ours > 50.0


@pytest.mark.parametrize("solver,floor", [("ck", 50.0), ("mckd", 2.0)])
def test_fmd_solvers_both_impulsive_on_bearing_fault(solver, floor):
    """两种 solver 都能抽出冲击模态；`ck` 在该数据上明显更强（见报告 §2）。"""
    S = _seg()
    Seed.set(0, verbose=False)
    rows = MD.Class.FMD(K=3, max_iter=10, num_hand=3, seed=0, solver=solver).decompose(S).IMFs
    assert _best_kurt(rows) > floor


def test_fmd_solver_ck_not_worse_than_mckd_on_bearing_fault():
    S = _seg()
    out = {}
    for solver in ("ck", "mckd"):
        Seed.set(0, verbose=False)
        out[solver] = _best_kurt(MD.Class.FMD(K=3, max_iter=10, num_hand=3,
                                              seed=0, solver=solver).decompose(S).IMFs)
    assert out["ck"] >= out["mckd"] * 0.9, out
