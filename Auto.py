from multiprocessing import cpu_count
from typing import Union, List, Tuple, Callable

import numpy as np
from dask.distributed import Client

from ProposalDistribution import Uniform
from Sampling import MetropolisHastings, MetropolisHastingsSymmetric
from utils import timer


# TODO: remove shape parameter

class AutoSample:
    def __init__(self, target: Callable, shape: Union[Tuple[int], List[int]], symmetric: bool = False, njobs: int = -1):
        self.target: Callable = target
        self.symmetric: float = symmetric
        self.shape: Union[Tuple[int], List[int]] = shape
        self.fitted: bool = False
        self.sampler = None
        if njobs is None or njobs < 0 or not isinstance(njobs, int):
            njobs = cpu_count()
        self.client = Client(threads_per_worker=1, n_workers=njobs)
        print('Dashboard available under: {}'.format(self.client.dashboard_link))

    def __built_samplers(self, n_samplers: int) -> list:
        spread = np.unique(np.random.gamma(shape=1, scale=2, size=n_samplers))
        proposals = [Uniform(s) for s in spread]
        if self.symmetric:
            samplers = [MetropolisHastingsSymmetric(target=self.target, proposal=p, shape=self.shape) for p in
                        proposals]
        else:
            samplers = [MetropolisHastings(target=self.target, proposal=p, shape=self.shape) for p in proposals]

        return samplers

    @staticmethod
    def __select_best(results: list, samplers) -> Union[MetropolisHastings, MetropolisHastingsSymmetric]:
        accepted = [abs(50 - r) for r in results]
        return samplers[accepted.index(min(accepted))]

    @timer
    def test_samplers(self) -> None:
        n_samplers = 100
        burnin = 1000
        size = 10000  # TODO: just hard code them for now

        samplers = self.__built_samplers(n_samplers)
        client = self.client

        def _run_sampler(sampler):
            sampler.run(burnin=burnin, size=size)
            return sampler.accepted

        futures = []
        for s in samplers:
            futures.append(client.submit(_run_sampler, s))
        results = client.gather(futures)

        self.sampler = self.__select_best(results, samplers)
        self.fitted = True

    def sample(self, size: int, chains: int = 1) -> np.ndarray:
        if not self.fitted:
            print('Looking for a best sampler...')
            self.test_samplers()

        if chains > 1:
            samplers = [self.sampler for _ in range(chains)]
            client = self.client

            def _run_sampler(sampler):
                chain = sampler.run(burnin=0, size=size)
                return chain

            print('Sampling...')
            futures = []
            for s in samplers:
                futures.append(client.submit(_run_sampler, s))
            results = client.gather(futures)

            return np.array(results)

        print('Sampling...')
        return self.sampler.run(burnin=0, size=size)
