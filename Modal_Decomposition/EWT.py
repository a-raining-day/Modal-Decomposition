from ewtpy import EWT1D

def ewt \
(
    S,
    N: int = 5,
    log: int = 0,
    detect: str = "locmax",
    completion: int = 0,
    reg: str = 'average',
    lengthFilter: int = 10,
    sigmaFilter: int = 5):
    """

    :param S: Signal
    :param N:
    :param log:
    :param detect:
    :param completion:
    :param reg:
    :param lengthFilter:
    :param sigmaFilter:
    :return: (N, len(S)) -> np.ndarray (2-dim)
    """

    ewt, mfb, boundaries = EWT1D(S, N, log, detect, completion, reg, lengthFilter, sigmaFilter)

    return ewt, mfb, boundaries


if __name__ == '__main__':
    import numpy as np

    S = np.random.rand(1, 500).squeeze()

    EWT, mfb, boundarire = ewt(S)

    print(EWT)

    print(type(EWT))