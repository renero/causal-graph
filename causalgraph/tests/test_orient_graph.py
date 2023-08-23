import numpy as np
import pytest

from causalgraph.independence.hsic import HSIC
from causalgraph.independence.regressors import fit_and_get_residuals


def test_orient_graph():
    np.random.seed(1234)
    X = np.random.normal(0, 1, 200).reshape(-1, 1)
    Y = X ** 3
    # Noise is added here.
    err_fwd = fit_and_get_residuals(X, Y, method="gpr")
    err_bwd = fit_and_get_residuals(Y, X, method="gpr")
    hsic_fwd = HSIC().fit(err_fwd, X)
    hsic_bwd = HSIC().fit(err_bwd, Y)
    dir_stat = "->" if hsic_bwd.hsic - hsic_fwd.hsic > 0 else "<-"

    if hsic_fwd.independence and not hsic_bwd.independence:
        dir_pv = "->"
    elif not hsic_fwd.independence and hsic_bwd.independence:
        dir_pv = "<-"
    else:
        dir_pv = "--"
    likelihood_fwd = -np.log(np.var(X)) - np.log(np.var(err_fwd))
    likelihood_bwd = -np.log(np.var(err_bwd)) - np.log(np.var(Y))
    dir_log = "->" if likelihood_fwd > likelihood_bwd else "<-"

    assert hsic_fwd.p_value == pytest.approx(0.62829005030, abs=0.0001)
    assert hsic_fwd.hsic == pytest.approx(0.00131107788741, abs=0.0001)
    assert hsic_fwd.independence == True

    assert hsic_bwd.p_value == pytest.approx(2.825e-10)
    assert hsic_bwd.hsic == pytest.approx(0.0069988146279, abs=0.0001)
    assert hsic_bwd.independence == False

    assert likelihood_fwd == pytest.approx(4.861426871058, abs=0.0001)
    assert likelihood_bwd == pytest.approx(1.126397359198, abs=0.0001)

    assert dir_stat == "->"
    assert dir_stat == "->"
    assert dir_stat == "->"
