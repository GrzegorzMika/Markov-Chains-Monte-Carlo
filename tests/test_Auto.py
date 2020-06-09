import pytest
from numpy.testing import assert_array_almost_equal

from Auto import *


class Test_AutoSample:
    @classmethod
    def setup_method(cls):
        np.random.seed(123)
        cls.target = lambda x: np.where(x < 0, 0, np.exp(-x))
        cls.shape = (1,)
        cls.njobs = 1
        cls.algo = AutoSample(target=cls.target, shape=cls.shape, njobs=cls.njobs)

    @pytest.mark.filterwarnings("ignore::UserWarning")
    @pytest.mark.filterwarnings('ignore::RuntimeWarning')
    def test_sample(self):
        sample = self.algo.sample(size=1, chains=1)
        assert sample.shape == (1, 1)
