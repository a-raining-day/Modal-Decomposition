"""
Python version:  (must)
    3.10.11

Lib and Version:  (if None write None)
    vmdpy - 0.2

Only accessed by:  (must)
    Only __init__.py

Modify:  (must)
    2026.3.25

Description: (if None write None)
    Realize VMD by using vmdpy lib.
"""

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
    from vmdpy import VMD

    u, u_hat, omega = VMD(S, alpha, tau, K, DC, init, tol)

    return u, u_hat, omega