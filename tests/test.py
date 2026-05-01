import time
from typing import Callable, Any
import numpy as np


def time_show(f: Callable) -> Callable:
    def F(*args, **kwargs):
        this = time.time()
        f(*args, **kwargs)
        now = time.time()
        delta = now - this
        print(f"this function use {delta}s")

    return F

def check_result(**kwargs):
    IMFs: np.ndarray = kwargs.get("IMFs", None)  # IMFs
    Res: np.ndarray = kwargs.get("Res", None)  # Res
    S: np.ndarray = kwargs.get("S", None)  # origin S

    RMSE = None
    OI = None
    R = None
    MSE = None

    if IMFs is None or IMFs.size == 0:
        print("this function doesn't decompose the useful IMFs. Exit...")
        return

    if IMFs.ndim == 1:
        IMFs = IMFs.reshape(1, -1)

    Res = np.asarray(Res).flatten()

    reconstructed = np.sum(IMFs, axis=0) + Res
    # 计算 RMSE 和 MSE
    mse = np.mean((S - reconstructed) ** 2)
    rmse = np.sqrt(mse)
    RMSE = rmse
    MSE = mse

    K, N = IMFs.shape
    if K <= 1:
        OI = 0.0

    else:
        corr_matrix = np.corrcoef(IMFs)

        upper_tri = np.triu_indices_from(corr_matrix, k=1)
        OI = np.mean(np.abs(corr_matrix[upper_tri]))

        R = corr_matrix

    print(f"RMSE={RMSE} | MES={MSE} | OI={OI} | R={R}")

@time_show
def test_SSA(S):
    from src.Modal_Decomposition import Function as F
    SSA = F.SSA

    RCs = SSA(S)

    IMFs = RCs[:-1, :]
    Res = RCs[-1, :]

    check_result(IMFs=IMFs, Res=Res, S=S)

    return RCs


if __name__ == '__main__':
    from src.Modal_Decomposition import Function

    Function.AAA