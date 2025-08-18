import numpy as np


def bool_to_float(x: bool) -> float:
    if x == True:
        return 1.0
    elif x == False:
        return -1.0
    else:
        raise ValueError(f"{x=}, expected bool")


def vectorize_bool_to_float(y: np.ndarray) -> np.ndarray:
    f = np.vectorize(bool_to_float)
    return f(y)


def get_probabilities_from_mapped_bools(h: np.ndarray) -> np.ndarray:
    p = 1 / (1 + np.exp(-2.0 * h))
    p = np.array([1 - p, p]).T
    return p
