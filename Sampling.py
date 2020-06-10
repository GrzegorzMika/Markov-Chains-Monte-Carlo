from typing import Optional, List, Tuple, Union, Callable

import numpy as np

from ProposalDistribution import ProposalDistribution


# TODO: deal with overflow

class MetropolisHastingsSymmetric:
    def __init__(self, target: Callable, proposal: ProposalDistribution,
                 initial: Optional[Union[float, np.ndarray]] = None,
                 shape: Optional[Union[Tuple[int], List[int]]] = None):
        self.target: Callable = target
        self.proposal: ProposalDistribution = proposal
        self.accepted: float = 0
        self.sample: Optional[np.ndarray] = None
        self.used: bool = False
        assert initial is not None or shape, 'At least one of the initial or shape arguments must be specified!'
        if initial is not None:
            self.initial: np.ndarray = np.array(initial)
        else:
            self.initial: np.ndarray = np.random.uniform(low=-1, high=1, size=shape)

    def run(self, size: int, burnin: Optional[int] = 1000, thinning: Optional[int] = None, verbose: int = 0):
        if self.used:
            self.initial = self.sample[-1]
            burnin = 0
        self.sample = np.empty((size + burnin, *self.initial.shape))
        self.sample[0] = self.initial
        u = np.random.uniform(0, 1, size + burnin)
        counter = 0

        for i in range(1, burnin):
            current_x = self.sample[i - 1]
            proposed = self.proposal.sample(current_x)
            a = np.min([1, self.target(proposed) / self.target(current_x)])
            if u[i] < a:
                self.sample[i] = proposed
            else:
                self.sample[i] = current_x
        for i in range(burnin + 1, size + burnin):
            current_x = self.sample[i - 1]
            proposed = self.proposal.sample(current_x)
            a = np.min([1, self.target(proposed) / self.target(current_x)])
            if u[i] < a:
                counter += 1
                self.sample[i] = proposed
            else:
                self.sample[i] = current_x

        self.accepted = counter / size * 100
        if verbose > 0:
            print("Proportion of samples accepted: {}%".format(round(counter / size * 100, 2)))

        self.used = True
        return self.sample[burnin:][::thinning]


class MetropolisHastings:
    def __init__(self, target: Callable, proposal: ProposalDistribution,
                 initial: Optional[Union[float, np.ndarray]] = None,
                 shape: Optional[Union[Tuple[int], List[int]]] = None):
        self.target: Callable = target
        self.proposal: ProposalDistribution = proposal
        self.accepted: float = 0
        self.sample: Optional[np.ndarray] = None
        self.used: bool = False
        assert initial is not None or shape, 'At least one of the initial or shape arguments must be specified!'
        if initial is not None:
            self.initial: np.ndarray = np.array(initial)
        else:
            self.initial: np.ndarray = np.random.uniform(low=-1, high=1, size=shape)

    def run(self, size: int, burnin: Optional[int] = 1000, thinning: Optional[int] = None, verbose: int = 0):
        if self.used:
            self.initial = self.sample[-1]
            burnin = 0
        self.sample = np.empty((size + burnin, *self.initial.shape))
        self.sample[0] = self.initial
        u = np.random.uniform(0, 1, size + burnin)
        counter = 0

        for i in range(1, burnin):
            current_x = self.sample[i - 1]
            proposed = self.proposal.sample(current_x)
            a = np.min([1, (self.target(proposed) * self.proposal.pdf(current_x, proposed)) /
                        (self.target(current_x) * self.proposal.pdf(proposed, current_x))])
            if u[i] < a:
                self.sample[i] = proposed
            else:
                self.sample[i] = current_x
        for i in range(burnin + 1, size + burnin):
            current_x = self.sample[i - 1]
            proposed = self.proposal.sample(current_x)
            a = np.min([1, (self.target(proposed) * self.proposal.pdf(current_x, proposed)) /
                        (self.target(current_x) * self.proposal.pdf(proposed, current_x))])
            if u[i] < a:
                counter += 1
                self.sample[i] = proposed
            else:
                self.sample[i] = current_x

        self.accepted = counter / size * 100
        if verbose > 0:
            print("Proportion of samples accepted: {}%".format(round(counter / size * 100, 2)))

        self.used = True
        return self.sample[burnin:][::thinning]
