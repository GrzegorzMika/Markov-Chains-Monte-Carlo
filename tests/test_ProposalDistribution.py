from numpy.testing import assert_array_almost_equal

from ProposalDistribution import *


class Test_Normal:
    @classmethod
    def setup_method(cls):
        np.random.seed(123)
        cls.sampler = Normal(mean=0, spread=1)

    def test_pdf(self):
        assert_array_almost_equal(self.sampler.pdf(np.array([0]), np.array([0])), np.array([0.39894228]))

    def test_sample(self):
        assert_array_almost_equal(self.sampler.sample(np.array([0])), np.array([-1.0856306]))


class Test_Uniform:
    @classmethod
    def setup_method(cls):
        np.random.seed(123)
        cls.sampler = Uniform(spread=1)

    def test_pdf(self):
        assert_array_almost_equal(self.sampler.pdf(np.array([0]), np.array([0])), np.array([1.]))

    def test_sample(self):
        assert_array_almost_equal(self.sampler.sample(np.array([0])), np.array([0.19646919]))
