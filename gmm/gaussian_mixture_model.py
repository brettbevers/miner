from math import pi
from scipy.misc import logsumexp
from scipy.stats import chi2
import numpy as np

from collections import namedtuple

from pyspark.mllib.clustering import GaussianMixture
from pyspark.mllib.linalg import Vectors

Gaussian = namedtuple('Gaussian', ['mu', 'sigma'])

GMMStats = namedtuple('GMMStats', ['log_likelihood', 'null_log_likelihoods', 'p_values'])


def remove_dim(matrix, index):
    return np.delete(np.delete(matrix, index, axis=0), index, axis=1)


class memoized:
    def __init__(self, f):
        self.f = f

    def __call__(self, *args, **kwargs):
        if not hasattr(self, 'val'):
            self.val = self.f(*args, **kwargs)

        return self.val


class SingularCovarianceException(Exception):
    pass


class ModelingService(object):
    def __init__(self, training_data):
        self.training_data = training_data
        self.gmms = {}
        self.subspaces = {}

    def get_subspace(self, dimensions):
        dimensions = tuple(dimensions)
        if dimensions not in self.subspaces:
            self.subspaces[dimensions] = self.training_data.map(lambda v: Vectors.dense([v[i] for i in dimensions]))
        return self.subspaces[dimensions]

    def get_gmm(self, k, dimensions):
        key = (k, tuple(dimensions))
        if key not in self.gmms:
            train = self.get_subspace(dimensions)
            self.gmms[key] = GaussianMixture.train(train, k)
        gmm = self.gmms[key]
        return GaussianMixtureModel(dimensions, gmm.weights, gmm.gaussians)

    def get_log_likelihood(self, k, subspace, dimensions):
        gmm = self.get_gmm(k, subspace)
        return gmm.log_likelihood(self, dimensions)

    def get_stats(self, k, dimensions):
        gmm = self.get_gmm(k, dimensions)
        return gmm.stats(self)


class GaussianMixtureModel(object):

    def __init__(self, dimensions, weights, gaussians):
        self.dimensions = dimensions
        self.weights = weights
        self.gaussians = gaussians

    @property
    @memoized
    def corrected_model(self):
        singular_sigmas = 0
        weights = []
        gaussians = []
        inverses = []

        for w, g in zip(self.weights, self.gaussians):
            try:
                s_inv = np.linalg.inv(g.sigma.toArray())
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
    @memoized
    def k(self):
        weights, gaussians, inverses = self.corrected_model
        return len(weights)

    def log_likelihood_summand_func(self, dimensions=None):
        if dimensions is None:
            dimensions = self.dimensions

        indices = [i for i, dim in enumerate(self.dimensions) if dim not in dimensions]

        gmm = [
            (w, np.delete(g.mu.toArray(), indices), remove_dim(g.sigma.toArray(), indices))
            for w, g, s_inv in zip(*self.corrected_model)
        ]

        d = len(dimensions)
        gaussians = [(mu, np.linalg.inv(sigma)) for w, mu, sigma in gmm]
        coeffs = np.array([w * np.power(2 * pi, -d / 2.0) * np.power(np.linalg.det(sigma), -0.5)
                           for w, mu, sigma in gmm])

        def f(x):
            exponents = []
            for mu, s_inv in gaussians:
                diff = x - mu
                exponents.append(-0.5 * np.matmul(np.matmul(diff, s_inv), diff))
            return logsumexp(np.array(exponents), b=coeffs)

        return f

    def log_likelihood(self, modeling_service, dimensions=None):
        k = self.k
        if k < 2:
            raise SingularCovarianceException("Only {} gaussians have non-singular covariance.".format(k))

        if dimensions is None:
            dimensions = self.dimensions

        data = modeling_service.get_subspace(dimensions)
        ll_func = self.log_likelihood_summand_func(dimensions)

        return data.map(ll_func).sum()

    def stats(self, modeling_service, dimensions=None):
        k = self.k
        if k < 2:
            raise SingularCovarianceException("Only {} gaussians have non-singular covariance.".format(k))

        if dimensions is None:
            dimensions = self.dimensions
        dimensions = np.array(dimensions)
        d = dimensions.shape[0]

        data = modeling_service.get_subspace(dimensions)

        ll_func = self.log_likelihood_summand_func(dimensions)
        if d < 2:
            marginal_ll_funcs = []
            null_ll_funcs = []
        else:
            marginal_ll_funcs = [self.log_likelihood_summand_func(np.delete(dimensions, i)) for i in range(d)]
            null_ll_funcs = [modeling_service.get_gmm(k, [dim]).log_likelihood_summand_func() for dim in dimensions]

        def f(v):
            x = v.toArray()
            results = [ll_func(x)]
            results.extend([
                marg_ll(np.delete(x, i)) + null_ll(x[i])
                for i, (marg_ll, null_ll) in enumerate(zip(marginal_ll_funcs, null_ll_funcs))
            ])
            return np.array(results)

        sums = data.map(f).sum()
        ll = sums[0]
        null_log_likelihoods = sums[1:]
        p_values = [chi2.sf(-2 * (x - ll), df=(d - 1) * k) for x in sums[1:]]

        return GMMStats(ll, null_log_likelihoods, p_values)
