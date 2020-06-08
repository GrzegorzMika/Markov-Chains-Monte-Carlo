from abc import ABCMeta, abstractmethod

import numpy as np


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
    __slots__ = ['mean', 'std']

    def __init__(self, mean: float, spread: float):
        super().__init__()
        self.mean = mean
        self.std = spread
        assert self.std > 0, "Wrong specification of distribution!"

    def sample(self, x):
        return x + np.random.normal(self.mean, self.std, x.shape)

    def pdf(self, x, cond):
        return 1 / (np.sqrt(2 * np.pi) * self.std) * np.exp(-(x - self.mean - cond) ** 2 / (2 * self.std ** 2))


class Uniform(ProposalDistribution):
    __slots__ = ['spread']

    def __init__(self, spread: float):
        super().__init__()
        self.spread = spread
        assert self.spread > 0, "Wrong specification of distribution!"

    def sample(self, x):
        return x + np.random.uniform(low=-self.spread / 2, high=self.spread / 2, size=x.shape)

    def pdf(self, x, cond):
        return np.array(1 / self.spread)
