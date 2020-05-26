import numpy as np


def normal(current: np.ndarray, mean: float = 0, std: float = 1) -> np.ndarray:
    return current + np.random.normal(loc=mean, scale=std, size=current.shape)


def uniform(current: np.ndarray, low: float = -1, high: float = 1) -> np.ndarray:
    return current + np.random.uniform(low=low, high=high, size=current.shape)
