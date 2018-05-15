from math import pi
from scipy.misc import logsumexp
from scipy.stats import chi2
import numpy as np

from collections import namedtuple

from pyspark.mllib.clustering import GaussianMixture
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.linalg.distributed import RowMatrix


# Calculate epsilon like https://github.com/apache/spark/blob/master/mllib-local/src/main/scala/org/apache/spark/ml/impl/Utils.scala
EPSILON = 1.0
while (1.0 + (EPSILON / 2.0)) != 1.0:
    EPSILON = EPSILON / 2.0


def remove_dim(matrix, index):
    return np.delete(np.delete(matrix, index, axis=0), index, axis=1)


def zero_off_diagonal(matrix, indices):
    result = np.copy(matrix)

    for x in indices:
        for j in range(result.shape[1]):
            if x != j:
                result[x][j] = 0.0
        for i in range(result.shape[0]):
            if x != i:
                result[i][x] = 0.0

    return result


def svd_pinv(m):
  U,s,Vh = np.linalg.svd(m)
  tol = EPSILON * max(s) * s.shape[0]
  s_inv = np.array([x**-1 if x > tol else 0.0 for x in s])
  m_inv = np.matmul(np.matmul(np.transpose(Vh), np.diag(s_inv)), np.transpose(U))
  return m_inv


def svd_pdet(m):
    U,s,Vh = np.linalg.svd(m)
    tol = EPSILON * max(s) * s.shape[0]
    return np.prod([x for x in s if x > tol])


class Gaussian(object):
    def __init__(self, mu, sigma):
        self.mu = mu  # type: np.ndarray
        self.sigma = sigma  # type: np.ndarray

    @classmethod
    def fromDict(cls, d):
        mu = np.array(d['mu'])
        sigma = np.array(d['sigma']).reshape(d['sigma_shape'])
        return cls(mu, sigma)

    def toDict(self):
        return {
            'mu': self.mu.tolist(),
            'sigma': self.sigma.flatten().tolist(),
            'sigma_shape': self.sigma.shape
        }

    @property
    def num_dimensions(self):
        return self.mu.shape[0]

    def marginal(self, dimensions):
        removed_dimensions = [i for i in range(self.num_dimensions) if i not in dimensions]
        return self.__class__(np.delete(self.mu, removed_dimensions), remove_dim(self.sigma, removed_dimensions))


class SingularCovarianceException(Exception):
    pass


class ModelingService(object):
    def __init__(self, training_data, max_iterations=200):
        self.training_data = training_data
        self.max_iterations = max_iterations
        self.gmms = {}
        self.subspaces = {}

    @property
    def num_dimensions(self):
        return self.training_data.take(1)[0].toArray().shape[0]

    def subspace(self, dimensions):
        key = tuple(dimensions)
        if key not in self.subspaces:
            removed_dimensions = [i for i in range(self.num_dimensions) if i not in dimensions]
            new_training_data = self.training_data \
                .map(lambda v: Vectors.dense(np.delete(v.toArray(), removed_dimensions)))
            self.subspaces[key] = self.__class__(new_training_data, self.max_iterations)
        return self.subspaces[key]

    def get_gmm(self, k):
        k = int(k)
        key = k

        if key not in self.gmms:
            if k == 1:
                training_data = self.training_data.rdd.map(lambda r: Vectors.fromML(r.features))
                row_matrix = RowMatrix(training_data)
                mean = row_matrix.computeColumnSummaryStatistics().mean()
                cov = row_matrix.computeCovariance().toArray()
                weights = [1.0]
                gaussians = [Gaussian(mean, cov)]
            else:
                m = GaussianMixture.train(self.training_data, k, maxIterations=self.max_iterations)
                weights = m.weights
                gaussians = [Gaussian(g.mu, g.sigma.toArray()) for g in m.gaussians]

            self.gmms[key] = GaussianMixtureModel(weights, gaussians)

        return self.gmms[key]

    def get_log_likelihood(self, k):
        gmm = self.get_gmm(k)
        return gmm.log_likelihood(self.training_data)

    def get_stats(self, k):
        gmm = self.get_gmm(k)
        return gmm.stats(self.training_data)


GMMStats = namedtuple('GMMStats', ['log_likelihood', 'null_log_likelihoods', 'p_values'])


class GaussianMixtureModel(object):

    def __init__(self, weights, gaussians):
        self.weights = weights
        self.gaussians = gaussians

    @property
    def num_dimensions(self):
        return self.gaussians[0].num_dimensions

    @property
    def corrected_model(self):
        singular_sigmas = 0
        weights = []
        gaussians = []
        inverses = []

        for w, g in zip(self.weights, self.gaussians):
            try:
                s_inv = np.linalg.inv(g.sigma)
                weights.append(w)
                gaussians.append(g)
                inverses.append(s_inv)
            except np.linalg.LinAlgError:
                singular_sigmas += 1
        if singular_sigmas > 0:
            print("Eliminated {} gaussians with singular covariance matrix.".format(singular_sigmas))

        total = sum(weights)
        weights = [w / total for w in weights]

        return weights, gaussians, inverses

    @property
    def k(self):
        return len(self.weights)

    def marginal(self, dimensions):
        new_weights = list(self.weights)
        new_gaussians = [g.marginal(dimensions) for g in self.gaussians]
        return self.__class__(new_weights, new_gaussians)

    def log_likelihood_summand_func(self):
        ll_gaussians = [(g.mu, svd_pinv(g.sigma)) for w, g in zip(self.weights, self.gaussians)]

        d = self.num_dimensions
        coeffs = np.array([w * np.power(2 * pi, -d / 2.0) * np.power(svd_pdet(g.sigma), -0.5)
                           for w, g in zip(self.weights, self.gaussians)])

        def f(x):
            exponents = []
            for mu, s_inv in ll_gaussians:
                diff = x - mu
                exponents.append(-0.5 * np.matmul(diff, np.matmul(s_inv, diff)))
            return logsumexp(np.array(exponents), b=coeffs)

        return f

    def log_likelihood(self, data):
        val = data.map(self.log_likelihood_summand_func()).sum()
        return -1 * np.abs(val) # because the pseudo-inverse sometimes flips the sign.

    def stats_func(self):
        k = self.k
        if k < 2:
            raise SingularCovarianceException("Only {} gaussians have non-singular covariance.".format(k))

        ll_func = self.log_likelihood_summand_func()

        d = self.num_dimensions

        if d < 2:
            marginal_ll_funcs = []
            null_ll_funcs = []
        else:
            dimensions = range(d)
            marginal_ll_funcs = [self.marginal(np.delete(dimensions, i)).log_likelihood_summand_func()
                                 for i in dimensions]
            null_ll_funcs = [self.marginal([dim]).log_likelihood_summand_func() for dim in dimensions]

        other_ll_funcs = list(enumerate(zip(marginal_ll_funcs, null_ll_funcs)))

        def f(v):
            x = v.toArray()
            results = [ll_func(x)]
            results.extend([
                marg_ll(np.delete(x, i)) + null_ll(x[i])
                for i, (marg_ll, null_ll) in other_ll_funcs
            ])
            return np.array(results)

        return f

    def stats(self, data):
        d = self.num_dimensions
        k = self.k

        sums = data.map(self.stats_func()).sum()
        sums = -1 * np.abs(sums) # because the pseudo-inverse sometimes flips the sign.

        ll = sums[0]
        null_log_likelihoods = sums[1:]
        p_values = [chi2.sf(-2 * (x - ll), df=(d - 1) * k) for x in sums[1:]]

        return GMMStats(ll, null_log_likelihoods, p_values)
