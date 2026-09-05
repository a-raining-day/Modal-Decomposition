import numpy as np
from typing import Tuple, Any, Optional

def uniform_type(S: list | np.ndarray | Any) -> np.ndarray:
    """
    CheckDim the type of S, and transform other type to ndarray.
    :param S: Signal(array-like)
    :param RAISE: True -> Raise Error | False -> return ERROR
    :return: np.ndarray
    """

    if isinstance(S ,np.ndarray):
        return S.squeeze()

    elif isinstance(S, list):
        S = np.asarray(S)  # transform list to ndarray.
        return S.squeeze()

    else:
        try:
            S = np.asarray(S)  # try to transform other array-like to ndarray.

        except (TypeError, ValueError):
            raise TypeError("The S can not transform to np.ndarray!")

        return S.squeeze()  # clear the superfluous dimension.

class CheckDim:
    def __init__(self, dim: int, RAISE: bool = True):
        self.dim = dim  # confirm the dim.
        self.RAISE = RAISE

    def __call__(self, S: list | np.ndarray, *args, **kwargs) -> Tuple[bool, np.ndarray]:
        """

        :param S:
        :param T: Time axis. If None, seem as uniform. (kwargs)
        :param args:
        :param kwargs:
        :return:
        """

        # check signal
        S = uniform_type(S)
        dim_sure, dim = self._check_dim(S)

        if not dim_sure:
            if self.RAISE:
                raise ValueError(f"The dim of S: {dim} != the required dim: {self.dim}!")
            else:
                return False, S

        else:
            return True, S

    def _check_dim(self, S: list | np.ndarray, goal_dim: Optional[int]=None) -> Tuple[bool, int]:
        """
        CheckDim the dimension of S, it must be 1-dim.

        Example: (1,2,3), [1,2,3], [[1,2,3]], [[...[1,2,3]...]]... -> 1-dim like.
        :param S:
        :return:
        """

        # uniform the array-like input to np.ndarray
        S = uniform_type(S)

        dim = S.ndim
        goal_dim = self.dim if goal_dim is None else goal_dim

        if dim != goal_dim:
            return False, dim

        else:
            return True, dim

def Check_Time(S: list | np.ndarray, T: list | np.ndarray) -> Tuple[bool, np.ndarray, np.ndarray]:
    """
    Check the length of T and dose the T is uniform.
    :param S:
    :param T:
    :return:
    """

    S = uniform_type(S)
    T = uniform_type(T)

    if S.size != T.size:
        raise ValueError("The length of T is not equal to the length of S!")

    if len(np.unique(np.diff(T))) == 1:
        return True, S, T
    else:
        return False, S, T

if __name__ == "__main__":
    a = [1, 2, 3]
    b = [[1, 2, 3]]
    c = (1, 2, 3)
    d = [[[[[[[1, 2, 3], [4, 5, 6]]]]]]]

    one_dim = CheckDim(1)
    two_dim = CheckDim(2)

    one_dim(a)