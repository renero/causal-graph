"""
Methods in this module perform HSIC independence test, and compute the HSIC
value and statistic.

You can cross check results with R dHSIC library.

    ```{r}
    > library(dHSIC)
    > dhsic.test(X, Y, method="gamma")
    $statistic
    [1] 0.3097533

    $crit.value
    [1] 0.5069064

    $p.value
    [1] 0.2518158

    $time
    GramMat   dHSIC CritVal
      0.000   0.000   0.001

    $bandwidth
    [1] 0.2110883 0.2121115
    ```

    Example:
        >>> X = np.array([\
        0.136486655, 0.108931511, 0.022105488, 0.349472863, 0.148249433,\
        -0.321564702, 0.320629400, -0.282624440, 0.263522936, -0.316252276])
        >>> Y = np.array([\
        -0.1669332713, 0.4886635816, 0.5315530519, 0.1330376544, -0.0632027887,\
        0.1640341743, -0.1833757726, 0.3754725901, -0.0722728821, -0.0395241960])
        
        >>> print(f"{hsic_value(X, Y):.6f}")
        0.030975
        >>> print(f"{hsic_stat(X, Y):.6f}")
        0.309753
        >>> print(f"{hsic_test(X, Y):.6f}")
        0.251816
        >>> hsic_independence(X, Y, alpha=0.01)
        True

    (c) J. Renero 2021
"""
import sys
from dataclasses import dataclass

import numpy as np
from scipy.stats import gamma

from .kernels import rbf_dot

MAX_INT = sys.maxsize * 2 + 1
MAX_SAMPLES = 200


def _prepare_input(X, Y, normalize: bool = True, max_samples: int = MAX_SAMPLES):
    """
    Prepare the input vectors (dimensions, max_samples and normalization).

    Params:
        - X (np.array): A 1-D or 2-D numpy array
        - Y (np.array): A 1-D or 2-D numpy array
        - normalize (bool): Default True, performs minMax normalization.

    Returns:
        Xm, Ym (np.array tuple) A normalized 2D tuple with a max nr of samples.
    """
    m = X.shape[0]
    if np.ndim(X) == 1:
        X = X.reshape(-1, 1)
    if np.ndim(Y) == 1:
        Y = Y.reshape(-1, 1)

    if m > max_samples:
        indx = np.random.choice(m, max_samples, replace=False)
        Xm = X[indx].astype(float)
        Ym = Y[indx].astype(float)
    else:
        Xm = X.astype(float)
        Ym = Y.astype(float)

    if normalize:
        Xm = (Xm - np.mean(Xm)) / np.std(Xm)
        Ym = (Ym - np.mean(Ym)) / np.std(Ym)

    return Xm, Ym


def _fast_hsic_stat(
        X, Y, sig=None, normalize=True, max_samples=MAX_SAMPLES, return_kernels=False
):
    """
    This function implements the HSIC independence test using a Gamma approximation
    to the test threshold. Use at most maxpnt points to save time.
    Copyright (c) 2018 Diviyan Kalainathan. Modified by J. Renero (2021)

    :param X: contains dx columns, m rows. Each row is an i.i.d sample
    :param Y: contains dy columns, m rows. Each row is an i.i.d sample
    :param sig: [0] (resp [1]) is kernel size for x(resp y)
        (set to median distance if -1)
    :param normalize: Whether perform MinMax normalization (default=True)
    :param max_samples: Max nr of samples to use. If data contains more rows than this
        a random sample of this size is used.
    :param return_kernels: Default is False, but if True this method also return the
        value of the kernels to perform the statistic test.

    :return: test statistic
    """

    if sig is None:
        sig = [-1, -1]
    m = X.shape[0]
    H = np.eye(m) - 1.0 / m * np.ones((m, m))

    K = rbf_dot(X, sig[0])
    L = rbf_dot(Y, sig[1])

    Kc = np.dot(H, np.dot(K, H))
    Lc = np.dot(H, np.dot(L, H))

    testStat = (1.0 / m) * (Kc.T * Lc).sum()
    if ~np.isfinite(testStat):
        testStat = 0

    if return_kernels:
        return testStat, K, L
    return testStat


def hsic_stat(X, Y, sig=None, normalize=True, max_samples=MAX_SAMPLES, return_kernels=False):
    r"""
    This method returns the value of the HSIC statistic, not the HSIC value itself.
    The HSIC statistic value is used to perform the independence test and get a
    p-value. The HSIC value (not the statististic) is simply

    :math: HSIC = \frac{HSIC_stat}{m}

    where _m_ is the nr of samples or the 1st dimension of _X_.
    """
    if sig is None:
        sig = [-1, -1]
    Xm, Ym = _prepare_input(X, Y, normalize, max_samples)
    return _fast_hsic_stat(Xm, Ym, sig, normalize, max_samples, return_kernels)


def hsic_test(X, Y, normalize=True, max_samples=MAX_SAMPLES):
    """
    This method performs the independence test using HSIC.

    Arguments:
        X (np.array): samples
        Y (np.array): samples
        normalize (bool): whether perform MinMax normalization (default=True)
        max_samples (int): Max nr. of samples for HSIC (default=MAX_SAMPLES)

    Returns:
        p_value: The p-value of the HSIC indep test.
    """
    Xm, Ym = _prepare_input(X, Y, normalize, max_samples)
    m = Xm.shape[0]
    stat, k1, k2 = hsic_stat(
        Xm, Ym, None, False, max_samples, return_kernels=True)
    a = [np.sum(K) / (m ** 2) for K in [k1, k2]]
    b = [np.sum(K ** 2) / (m ** 2) for K in [k1, k2]]
    c = [np.sum(np.sum(K, axis=0) ** 2) / (m ** 3) for K in [k1, k2]]
    d = [i ** 2 for i in a]

    prod_a = np.prod(a)
    prod_b = np.prod(b)
    prod_c = np.prod(c)
    prod_d = np.array(prod_a ** 2)

    outprod_a = prod_a / a
    outprod_c = prod_c / c
    outprod_d = outprod_a ** 2

    # Expectation
    E = (1 + prod_a - np.sum(outprod_a)) / m

    term1 = prod_b
    term2 = prod_d
    term3 = 2 * prod_c
    term4, term5, term6, term7 = 0, 0, 0, 0

    term4 = term4 + b[0] * outprod_d[0]
    term5 = term5 + b[0] * outprod_c[0]
    term6 = term6 + c[0] * outprod_d[0]
    term7 = term7 + 2 * c[0] * c[1] * outprod_d[0] / d[1]

    term4 = term4 + b[1] * outprod_d[1]
    term5 = -2 * (term5 + b[1] * outprod_c[1])
    term6 = -2 * (term6 + c[1] * outprod_d[1])

    m4 = m * (m - 1) * (m - 2) * (m - 3)

    # Variance
    V = (2 * (m - 4) * (m - 5) / m4) * (
        term1 + term2 + term3 + term4 + term5 + term6 + term7
    )

    # calculation of alpha and beta of the gamma approximation
    a = (E ** 2) / V
    b = (m * V) / E

    # Cumulative distribution function
    p_value = 1 - gamma.cdf(stat, a, scale=b)

    return p_value


def hsic_value(X, Y, sig=None, normalize=True, max_samples=MAX_SAMPLES):
    r"""
    This method returns the value of HSIC, not the statistic.
    The HSIC value (not the statististic) is simply

    :math: HSIC = \frac{stat}{m}

    where _m_ is the nr of samples or the 1st dimension of _X_.
    """
    Xm, Ym = _prepare_input(X, Y, normalize, max_samples)
    stat = hsic_stat(Xm, Ym, sig, False, max_samples)
    return stat / Xm.shape[0]


def hsic_independence(X, Y, alpha=0.02, sig=None, normalize=True, max_samples=MAX_SAMPLES):
    p_value = hsic_stat(X, Y, sig, normalize, max_samples)
    if p_value > alpha:
        return True
    else:
        return False


@dataclass
class HSIC_Values:
    hsic: float = 0.
    p_value: float = 0.
    stat: float = 0.
    independence: bool = False

    def __init__(self, hsic, p_value, stat, independence):
        self.hsic = hsic
        self.p_value = p_value
        self.stat = stat
        self.independence = independence

    def __str__(self):
        s = f"HSIC...: {self.hsic:.6g}\n"
        s += f"p_value: {self.p_value:.6g}\n"
        s += f"stat...: {self.stat:.6g}\n"
        s += f"indep..: {self.independence}"
        return s


class HSIC:
    """
    Provides a class sklearn-type interface to the hsic methods in this module.

    Params:
        normalize (bool) default is True for MinMax
        max_samples (int): default is MAX_INT (2^64 - 1)
        sig (Tuple[int, int]):
        alpha (float): Significance level for independence test (def. 0.05)

    Example:
        >>> from causalgraph.independence.hsic import HSIC

        >>> X = np.array([\
        0.136486655, 0.108931511, 0.022105488, 0.349472863, 0.148249433,\
        -0.321564702, 0.320629400, -0.282624440, 0.263522936, -0.316252276])
        >>> Y = np.array([\
        -0.1669332713, 0.4886635816, 0.5315530519, 0.1330376544, -0.0632027887,\
        0.1640341743, -0.1833757726, 0.3754725901, -0.0722728821, -0.0395241960])

        >>> h = HSIC()
        >>> result = h.fit(X, Y)
        >>> print(result)
        HSIC...: 0.0309753
        p_value: 0.251816
        stat...: 0.309753
        indep..: True

    """

    def __init__(self, normalize=True, max_samples=MAX_SAMPLES, sig=None, alpha=0.05):
        self.normalize = normalize
        self.max_samples = max_samples
        if sig is None:
            self.sig = [-1, -1]
        self.alpha = alpha

        self.p_val: float = 0.0
        self.h_val: float = 0.0
        self.h_stat: float = 0.0
        self.h_indep: bool = False

    def fit(self, X, Y):
        """
        Computes the HSIC value, the p_value and the statistic value.

        Args:
            X (np.ndarray):
            Y (np.ndarray):

        Returns:
            HSIC_Values
        """
        self.Xm, self.Ym = self._prepare_input(X, Y)
        self.h_stat = self._fast_hsic_stat()
        self.p_val = self._test()
        self.h_val = self.h_stat / self.m
        self.h_indep = self.p_value > self.alpha

        return HSIC_Values(self.h_val, self.p_value, self.h_stat, self.h_indep)

    def _prepare_input(self, X, Y):
        """
        Prepare the input vectors (dimensions, max_samples and normalization).

        Params:
            - X (np.array): A 1-D or 2-D numpy array
            - Y (np.array): A 1-D or 2-D numpy array
            - normalize (bool): Default True, performs minMax normalization.

        Returns:
            Xm, Ym (np.array tuple) A normalized 2D tuple with a max nr of samples.
        """
        m = X.shape[0]
        if np.ndim(X) == 1:
            X = X.reshape(-1, 1)
        if np.ndim(Y) == 1:
            Y = Y.reshape(-1, 1)

        if m > self.max_samples:
            indx = np.random.choice(m, self.max_samples, replace=False)
            Xm = X[indx].astype(float)
            Ym = Y[indx].astype(float)
        else:
            Xm = X.astype(float)
            Ym = Y.astype(float)

        self.m = Xm.shape[0]

        if self.normalize:
            Xm = (Xm - np.mean(Xm)) / np.std(Xm)
            Ym = (Ym - np.mean(Ym)) / np.std(Ym)

        self.Xm = Xm
        self.Ym = Ym

        return self.Xm, self.Ym

    def _fast_hsic_stat(self):
        """
        This function implements the HSIC independence test using a Gamma approximation
        to the test threshold. Use at most maxpnt points to save time.
        Copyright (c) 2018 Diviyan Kalainathan. Modified by J. Renero (2021)

        :return: test statistic
        """
        m = self.m
        H = np.eye(m) - 1.0 / m * np.ones((m, m))

        self.K = rbf_dot(self.Xm, self.sig[0])
        self.L = rbf_dot(self.Ym, self.sig[1])

        Kc = np.dot(H, np.dot(self.K, H))
        Lc = np.dot(H, np.dot(self.L, H))

        testStat = (1.0 / m) * (Kc.T * Lc).sum()
        if ~np.isfinite(testStat):
            testStat = 0

        return testStat

    def _test(self):
        """
        This method performs the independence test using HSIC.

        Returns:
            p_value: The p-value of the HSIC indep test.
        """
        m = self.m
        stat, k1, k2 = self.stat, self.K, self.L

        a = [np.sum(K) / (m ** 2) for K in [k1, k2]]
        b = [np.sum(K ** 2) / (m ** 2) for K in [k1, k2]]
        c = [np.sum(np.sum(K, axis=0) ** 2) / (m ** 3) for K in [k1, k2]]
        d = [i ** 2 for i in a]

        prod_a = np.prod(a)
        prod_b = np.prod(b)
        prod_c = np.prod(c)
        prod_d = np.array(prod_a ** 2)

        outprod_a = prod_a / a
        outprod_c = prod_c / c
        outprod_d = outprod_a ** 2

        # Expectation
        E = (1 + prod_a - np.sum(outprod_a)) / m

        term1 = prod_b
        term2 = prod_d
        term3 = 2 * prod_c
        term4, term5, term6, term7 = 0, 0, 0, 0

        term4 = term4 + b[0] * outprod_d[0]
        term5 = term5 + b[0] * outprod_c[0]
        term6 = term6 + c[0] * outprod_d[0]
        term7 = term7 + 2 * c[0] * c[1] * outprod_d[0] / d[1]

        term4 = term4 + b[1] * outprod_d[1]
        term5 = -2 * (term5 + b[1] * outprod_c[1])
        term6 = -2 * (term6 + c[1] * outprod_d[1])

        m4 = m * (m - 1) * (m - 2) * (m - 3)

        # Variance
        V = (2 * (m - 4) * (m - 5) / m4) * (
            term1 + term2 + term3 + term4 + term5 + term6 + term7
        )

        # calculation of alpha and beta of the gamma approximation
        a = (E ** 2) / V
        b = (m * V) / E

        # Cumulative distribution function
        p_value = 1 - gamma.cdf(stat, a, scale=b)

        return p_value

    @property
    def p_value(self):
        return self.p_val

    @property
    def hsic(self):
        return self.h_val

    @property
    def stat(self):
        return self.h_stat

    @property
    def independence(self):
        return self.h_indep
