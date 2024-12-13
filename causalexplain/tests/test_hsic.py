import pytest
import numpy as np

from ..independence.hsic import HSIC
from hyppo.independence import Hsic


@pytest.fixture
def vectors():
    X = np.array([
        0.136486655, 0.108931511, 0.022105488, 0.349472863, 0.148249433,
        -0.321564702, 0.320629400, -0.282624440, 0.263522936, -0.316252276])
    Y = np.array([
        -0.1669332713, 0.4886635816, 0.5315530519, 0.1330376544, -0.0632027887,
        0.1640341743, -0.1833757726, 0.3754725901, -0.0722728821, -0.0395241960])
    return X, Y


# This is the test from causalexplain
def test_HSIC(vectors):
    X, Y = vectors

    np.random.seed(1234)
    h = HSIC()
    result = h.fit(X, Y)
    assert result.hsic == pytest.approx(0.0)
    assert result.stat == pytest.approx(0.0)
    assert result.p_value == pytest.approx(0.33058547413699446)
    assert result.independence


# This is the test from hyppo. p-value matches the one from causalexplain
def test_HSIC_own(vectors):
    X, Y = vectors

    np.random.seed(1234)
    stat, pval = Hsic().test(X, Y)
    assert stat == pytest.approx(0.18, abs=0.01)
    assert pval == pytest.approx(0.30, abs=0.01)
