from typing import Optional, List, Tuple, Union

import numpy as np


class MetropolisHastingsSymmetric:
    def __init__(self, target, proposal, initial: Optional[float] = None,
                 shape: Optional[Union[Tuple[int], List[int]]] = None):
        self.target = target
        self.proposal = proposal
        assert initial or shape, 'At least one of the initial or shape arguments must be specified!'
        if initial:
            self.initial = np.array(initial)
        else:
            self.initial = np.random.uniform(low=-1, high=1, size=shape)

    def run(self, size: int, burnin: Optional[int] = 1000, verbose: int = 0):
        sample = np.empty((size + burnin, *self.initial.shape))
        sample[0] = self.initial
        u = np.random.uniform(0, 1, size + burnin)
        counter = 1
        for i in range(1, size + burnin):
            # propose
            current_x = sample[i - 1]
            proposed = self.proposal.sample(current_x)
            # acceptance probability
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
    def __init__(self, target, proposal, initial: Optional[float] = None,
                 shape: Optional[Union[Tuple[int], List[int]]] = None):
        self.target = target
        self.proposal = proposal
        assert initial or shape, 'At least one of the initial or shape arguments must be specified!'
        if initial:
            self.initial = np.array(initial)
        else:
            self.initial = np.random.uniform(low=-1, high=1, size=shape)

    def run(self, size: int, burnin: Optional[int] = 1000, verbose: int = 0):
        sample = np.empty((size + burnin, *self.initial.shape))
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
