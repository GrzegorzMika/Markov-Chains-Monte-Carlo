from typing import Optional

from scipy.optimize import minimize_scalar
import numpy as np

from utils import timer


def rejection(pdf, c: Optional[float] = None) -> np.ndarray:
    if not c:
        c = -minimize_scalar(lambda x: -pdf(x), bounds=(0, 1), method='bounded', tol=1e-10).fun
    sample = []
    while len(sample) < 1:
        u1 = np.random.uniform(0, 1, 1)
        u2 = np.random.uniform(0, 1, 1)
        if u2 <= pdf(u1) / c:
            sample.append(u1)
    return np.array(sample).flatten()


class MetropolisHastingsSymmetric:
    def __init__(self, target, proposal, initial: Optional[float] = None):
        self.target = target
        self.proposal = proposal
        if initial:
            self.initial = initial
        else:
            self.initial = np.random.uniform(0, 1, 1)

    def run(self, size: int, burnin: Optional[int] = 1000, verbose: int = 0):
        sample = np.empty(size + burnin)
        sample[0] = self.initial
        u = np.random.uniform(0, 1, size + burnin)
        counter = 1
        for i in range(1, size + burnin):
            # propose
            current_x = sample[i - 1]
            proposed = self.proposal.sample(current_x)
            # acceptance proability
            a = np.min([1, self.target(proposed) / self.target(current_x)])
            # reject or accept
            if u[i] < a:
                counter += 1
                sample[i] = proposed
            else:
                sample[i] = current_x

        if verbose > 0:
            print("Proportion of samples accepted: {}%".format(round(counter / (size + burnin) * 100, 2)))

        return sample[burnin:]


class MetropolisHastings:
    def __init__(self, target, proposal, initial: Optional[float] = None):
        self.target = target
        self.proposal = proposal
        if initial:
            self.initial = initial
        else:
            self.initial = np.random.uniform(0, 1, 1)

    def run(self, size: int, burnin: Optional[int] = 1000, verbose: int = 0):
        sample = np.empty(size + burnin)
        sample[0] = self.initial
        u = np.random.uniform(0, 1, size + burnin)
        counter = 1
        for i in range(1, size + burnin):
            # propose
            current_x = sample[i - 1]
            proposed = self.proposal.sample(current_x)
            # acceptance proability
            a = np.min([1, (self.target(proposed) * self.proposal.pdf(current_x, proposed)) /
                        (self.target(current_x) * self.proposal.pdf(proposed, current_x))])
            # reject or accept
            if u[i] < a:
                counter += 1
                sample[i] = proposed
            else:
                sample[i] = current_x

        if verbose > 0:
            print("Proportion of samples accepted: {}%".format(round(counter / (size + burnin) * 100, 2)))

        return sample[burnin:]
