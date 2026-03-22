import numpy as np
from vmdpy import VMD

def vmd(S, alpha = 2000, tau = 0.0, K = 2, DC = 0, init = 1, tol = 1e-7):
    """
    :param S: Signal (1-dim)
    :param alpha: broadband constraints
    :param tau: noise tolerance
    :param K: num of IMFs
    :param DC: is included directional component
    :param init: way of initial
    :param tol: convergence threshold
    :return:
    """

    u, u_hat, omega = VMD(S, alpha, tau, K, DC, init, tol)

    return u, u_hat, omega