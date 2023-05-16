"""
Defining a set of classes that represent causal functions/ mechanisms.
"""

import random
import numpy as np
from scipy.stats import bernoulli
from sklearn.mixture import GaussianMixture as GMM
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.gaussian_process import GaussianProcessRegressor


class LinearMechanism(object):
    """Linear mechanism, where Effect = alpha*Cause + Noise."""

    def __init__(self, ncauses, points, d=4, noise_coeff=.7):
        """Init the mechanism."""
        super(LinearMechanism, self).__init__()
        self.n_causes = ncauses
        self.points = points

        self.coefflist = []

        for i in range(ncauses):
            self.coefflist.append(random.random())

        self.noise = np.random.randn(points, 1)
        self.d = d

    def __call__(self, causes):
        """Run the mechanism."""
        # Additive only, for now
        effect = np.zeros((self.points, 1))
        # Compute each cause's contribution
        for par in range(causes.shape[1]):
            effect[:, 0] = effect[:, 0] + self.coefflist[par]*causes[:, par]
        effect[:, 0] = effect[:, 0] + self.noise[:, 0]

        return effect


class SigmoidAM_Mechanism(object):

    def __init__(self, ncauses, points, d=4, noise_coeff=.7):
        """Init the mechanism."""
        super(SigmoidAM_Mechanism, self).__init__()
        self.n_causes = ncauses
        self.points = points

        self.a = np.random.exponential(1/4) + 1
        ber = bernoulli.rvs(0.5)
        self.b = ber * np.random.uniform(-2, -0.5) + (1-ber)*np.random.uniform(0.5, 2)
        self.c = np.random.uniform(-2, 2)
        self.noise = 0.1*np.random.randn(points, 1)

    def mechanism(self, x):
        result = np.\
            zeros((self.points, 1))
        for i in range(self.points):

            result[i, 0] = self.a * self.b * (x[i] + self.c) / (1 + abs(self.b * (x[i] + self.c)))

        return result

    def __call__(self, causes):
        """Run the mechanism."""
        # Additive only
        effect = np.zeros((self.points, 1))
        # Compute each cause's contribution
        for par in range(causes.shape[1]):
            effect[:, 0] = effect[:, 0] + self.mechanism(causes[:, par])[:, 0]

        effect[:, 0] = effect[:, 0] + self.noise[:, 0]

        return effect


class SigmoidMix_Mechanism(object):

    def __init__(self, ncauses, points, d=4, noise_coeff=.7):
        """Init the mechanism."""
        super(SigmoidMix_Mechanism, self).__init__()
        self.n_causes = ncauses
        self.points = points

        self.a = np.random.exponential(1/4) + 1
        ber = bernoulli.rvs(0.5)
        self.b = ber * np.random.uniform(-2, -0.5) + (1-ber)*np.random.uniform(0.5, 2)
        self.c = np.random.uniform(-2, 2)

        self.noise = 0.1*np.random.randn(points, 1)

    def mechanism(self, causes):
        result = np.zeros((self.points, 1))
        for i in range(self.points):
            pre_add_effect = 0
            for c in range(causes.shape[1]):
                pre_add_effect += causes[i, c]
            pre_add_effect += self.noise[i]

            result[i, 0] = self.a * self.b * \
                (pre_add_effect + self.c)/(1 + abs(self.b*(pre_add_effect + self.c)))

        return result

    def __call__(self, causes):
        """Run the mechanism."""
        effect = np.zeros((self.points, 1))
        # Compute each cause's contribution

        effect[:, 0] = self.mechanism(causes)[:, 0]
        return effect


class Polynomial_Mechanism(object):

    def __init__(self, ncauses, points, d=2, noise_coeff=.7):
        """Init the mechanism."""
        super(Polynomial_Mechanism, self).__init__()
        self.n_causes = ncauses
        self.points = points
        self.d = d
        self.polycause = []

        for c in range(ncauses):
            self.coefflist = []
            for j in range(self.d + 1):
                self.coefflist.append(random.random())
            self.polycause.append(self.coefflist)

        self.ber = bernoulli.rvs(0.5)
        self.noise = 0.1*np.random.randn(points, 1)

    def mechanism(self, x, par):

        list_coeff = self.polycause[par]
        result = np.zeros((self.points, 1))
        for i in range(self.points):
            for j in range(self.d+1):
                result[i, 0] += list_coeff[j]*np.power(x[i], j)
            result[i, 0] = min(result[i, 0], 1)
            result[i, 0] = max(result[i, 0], -1)

        return result

    def __call__(self, causes):
        """Run the mechanism."""
        effect = np.zeros((self.points, 1))
        # Compute each cause's contribution
        for par in range(causes.shape[1]):
            effect[:, 0] = effect[:, 0] + self.mechanism(causes[:, par], par)[:, 0]

        if(self.ber > 0 and causes.shape[1] > 0):
            effect[:, 0] = effect[:, 0] * self.noise[:, 0]
        else:
            effect[:, 0] = effect[:, 0] + self.noise[:, 0]

        return effect


def computeGaussKernel(x):
    xnorm = np.power(euclidean_distances(x, x), 2)
    return np.exp(-xnorm / (2.0))


class GaussianProcessAdd_Mechanism(object):

    def __init__(self, ncauses, points):
        """Init the mechanism."""
        super(GaussianProcessAdd_Mechanism, self).__init__()
        self.n_causes = ncauses
        self.points = points

        self.noise = 0.1*np.random.randn(points, 1)
        self.nb_step = 0

    def mechanism(self, x):
        """Run the mechanism."""
        self.nb_step += 1
        x = np.reshape(x, (x.shape[0], 1))

        if(self.nb_step < 5):
            cov = computeGaussKernel(x)
            mean = np.zeros((1, self.points))[0, :]
            y = np.random.multivariate_normal(mean, cov)
        elif(self.nb_step == 5):
            cov = computeGaussKernel(x)
            mean = np.zeros((1, self.points))[0, :]
            y = np.random.multivariate_normal(mean, cov)
            self.gpr = GaussianProcessRegressor()
            self.gpr.fit(x, y)
            y = self.gpr.predict(x)
        else:
            y = self.gpr.predict(x)

        return y

    def __call__(self, causes):
        """Run the mechanism."""
        # Additive only
        effect = np.zeros((self.points, 1))
        # Compute each cause's contribution
        for par in range(causes.shape[1]):
            effect[:, 0] = effect[:, 0] + self.mechanism(causes[:, par])

        effect[:, 0] = effect[:, 0] + self.noise[:, 0]

        return effect


class GaussianProcessMix_Mechanism(object):

    def __init__(self, ncauses, points):
        """Init the mechanism."""
        super(GaussianProcessMix_Mechanism, self).__init__()
        self.n_causes = ncauses
        self.points = points
        self.noise = 0.1*np.random.randn(points, 1)
        self.nb_step = 0

    def mechanism(self, x):
        self.nb_step += 1
        x = np.reshape(x, (x.shape[0], x.shape[1]))

        if(self.nb_step < 2):
            cov = computeGaussKernel(x)
            mean = np.zeros((1, self.points))[0, :]
            y = np.random.multivariate_normal(mean, cov)
        elif(self.nb_step == 2):
            cov = computeGaussKernel(x)
            mean = np.zeros((1, self.points))[0, :]
            y = np.random.multivariate_normal(mean, cov)
            self.gpr = GaussianProcessRegressor()
            self.gpr.fit(x, y)
            y = self.gpr.predict(x)
        else:
            y = self.gpr.predict(x)

        return y

    def __call__(self, causes):
        """Run the mechanism."""
        effect = np.zeros((self.points, 1))
        # Compute each cause's contribution
        if(causes.shape[1] > 0):
            mix = np.hstack((causes, self.noise))
            effect[:, 0] = self.mechanism(mix)
        else:
            effect[:, 0] = self.mechanism(self.noise)

        return effect


def gmm_cause(n, k=4, p1=2, p2=2):
    g = GMM(k)
    g.means_ = p1 * np.random.randn(k, 1)
    g.covars_ = np.power(abs(p2 * np.random.randn(k, 1) + 1), 2)
    g.weights_ = abs(np.random.rand(k, 1))
    g.weights_ = g.weights_ / sum(g.weights_)
    return np.random.uniform(-1, 1, n)


def gaussian_cause(n):
    return np.random.randn(n, 1)[:, 0]


def noise(n, v):
    return v * np.random.rand(1) * np.random.randn(n, 1) + random.sample([2, -2], 1)


