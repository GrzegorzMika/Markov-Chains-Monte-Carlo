from abc import ABCMeta, abstractmethod

import numpy as np
from scipy.stats import norm


class ProposalDistribution(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self):
        ...

    @abstractmethod
    def sample(self, x: np.ndarray) -> np.ndarray:
        ...

    @abstractmethod
    def pdf(self, x: np.ndarray, cond: np.ndarray) -> np.ndarray:
        ...


class Normal(ProposalDistribution):
    def __init__(self, mean: float, std: float):
        super().__init__()
        self.mean = mean
        self.std = std
        assert self.std > 0, "Wrong specification of distribution!"

    def sample(self, x):
        return x + np.random.normal(self.mean, self.std, x.shape)

    def pdf(self, x, cond):
        return 1/(np.sqrt(2*np.pi) * self.std)*np.exp(-(x-self.mean-cond)**2/(2*self.std**2))
        # return norm(loc=self.mean + cond, scale=self.std).pdf(x)
