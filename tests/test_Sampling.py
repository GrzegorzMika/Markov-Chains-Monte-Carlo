import pytest
from numpy.testing import assert_array_almost_equal

from ProposalDistribution import Uniform
from Sampling import *


class Test_MetropolisHastingsSymmetric:
    @classmethod
    def setup_method(cls):
        np.random.seed(123)
        cls.sampler = Uniform(spread=1)
        cls.target = lambda x: np.where(x < 0, 0, np.exp(-x))
        cls.shape = (1,)
        cls.algo = MetropolisHastingsSymmetric(target=cls.target, proposal=cls.sampler, shape=cls.shape)

    def test_shape(self):
        assert_array_almost_equal(self.algo.initial, np.array([0.39293837]))

    @pytest.mark.filterwarnings('ignore::RuntimeWarning')
    def test_sample(self):
        assert_array_almost_equal(self.algo.run(burnin=100, size=1), np.array([[6.926473e-310]]))


class Test_MetropolisHastings:
    @classmethod
    def setup_method(cls):
        np.random.seed(123)
        cls.sampler = Uniform(spread=1)
        cls.target = lambda x: np.where(x < 0, 0, np.exp(-x))
        cls.shape = (1,)
        cls.algo = MetropolisHastings(target=cls.target, proposal=cls.sampler, shape=cls.shape)

    def test_shape(self):
        assert_array_almost_equal(self.algo.initial, np.array([0.39293837]))

    @pytest.mark.filterwarnings('ignore::RuntimeWarning')
    def test_sample(self):
        assert_array_almost_equal(self.algo.run(burnin=100, size=1), np.array([[0.666625]]))
