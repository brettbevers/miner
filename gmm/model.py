from math import pi
from scipy.misc import logsumexp
from scipy.stats import norm
import numpy as np
from threading import Thread, Lock

from pyspark import RDD
from pyspark.ml.clustering import GaussianMixture
from pyspark.mllib.linalg import Vectors
from pyspark.ml.linalg import Vectors as MlVectors
from pyspark.mllib.linalg.distributed import RowMatrix
from pyspark.sql import DataFrame


# Calculate epsilon like
# https://github.com/apache/spark/blob/master/mllib-local/src/main/scala/org/apache/spark/ml/impl/Utils.scala
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


class ModelingService(object):
    def __init__(self, training_data, max_iterations=200, spark_session=None,
                 ll_sample_size=5, ll_sample_fraction=0.99, fit_model_retries=10):
        if hasattr(training_data, 'rdd'):
            self.ml_training_data = training_data  # type: DataFrame
            self.mllib_training_data = training_data.rdd\
                .map(lambda r: Vectors.fromML(r.features)).persist()  # type: RDD
        else:
            if spark_session is None:
                raise Exception("Spark session must be provided if training data is not a dataframe.")
            self.mllib_training_data = training_data  # type: RDD
            self.ml_training_data = spark_session.createDataFrame(
                training_data.map(lambda v: (MlVectors.dense(v),)),
                ['features']
            )  # type: DataFrame

        self.max_iterations = max_iterations
        self.ll_sample_size = ll_sample_size
        self.ll_sample_fraction = ll_sample_fraction
        self.ll_samples = {}
        self.fit_model_retries = fit_model_retries

    @property
    def num_dimensions(self):
        return self.mllib_training_data.take(1)[0].toArray().shape[0]

    def fit_ml_model(self, k, sample_fraction=None, retry=True):
        if sample_fraction:
            training_data = self.ml_training_data.sample(False, fraction=sample_fraction)
        else:
            training_data = self.ml_training_data

        result = GaussianMixture(k=k, maxIter=self.max_iterations).fit(training_data)
        ll = result.summary.logLikelihood

        # Retry to get a valid model if the calculated log likelihood is > 0.
        retries = 0
        while retry and ll > 0 and retries < self.fit_model_retries:
            retry_sample_fraction = sample_fraction or self.ll_sample_fraction
            retry_data = self.ml_training_data.sample(False, fraction=retry_sample_fraction)
            result = GaussianMixture(k=k, maxIter=self.max_iterations).fit(retry_data)
            ll = result.summary.logLikelihood
            retries += 1

        return result

    def get_gmm(self, k, sample_fraction=None, retry=True):
        if k == 1:
            if sample_fraction:
                data = self.mllib_training_data.sample(False, sample_fraction)
            else:
                data = self.mllib_training_data
            row_matrix = RowMatrix(data)
            mean = row_matrix.computeColumnSummaryStatistics().mean()
            cov = row_matrix.computeCovariance().toArray()
            weights = [1.0]
            gaussians = [Gaussian(mean, cov)]
            log_likelihood = None
        else:
            m = self.fit_ml_model(k, sample_fraction=sample_fraction, retry=retry)
            weights = m.weights
            gaussians = [Gaussian(g.mean, g.cov.toArray()) for g in m.gaussiansDF.collect()]
            log_likelihood = m.summary.logLikelihood

        return GaussianMixtureModel(weights, gaussians, log_likelihood)

    def get_log_likelihood(self, k, retry=True):
        gmm = self.get_gmm(k, retry=retry)
        return gmm.log_likelihood(self.mllib_training_data)

    def get_stats_sample(self, k):
        k = int(k)
        key = k

        if key not in self.ll_samples:
            sample = []
            sample_lock = Lock()

            def f():
                gmm = self.get_gmm(k, sample_fraction=self.ll_sample_fraction)  # type: GaussianMixtureModel
                data = self.mllib_training_data.sample(False, self.ll_sample_fraction)
                lls = gmm.calc_stats(data)
                sample_lock.acquire()
                sample.append(lls)
                sample_lock.release()

            threads = []
            for _ in range(self.ll_sample_size):
                t = Thread(target=f)
                t.start()
                threads.append(t)

            for t in threads:
                t.join()

            self.ll_samples[key] = np.array(sample)

        return self.ll_samples[key]

    def get_stats(self, k, calc_p_values=True):
        if not calc_p_values:
            gmm = self.get_gmm(k)
            return gmm.stats(self.mllib_training_data)

        stats_sample = self.get_stats_sample(k)
        sample_size = stats_sample.shape[0]
        mean_lls = np.mean(stats_sample, axis=0)
        diffs = np.array([v[1:] - v[0] for v in stats_sample])

        p_values = [norm.cdf(x) for x in (np.mean(diffs, axis=0) * np.sqrt(sample_size)) / np.std(diffs, axis=0)]

        return {
            'num_dimensions': diffs.shape[1],
            'k': k,
            'stats_sample': stats_sample.tolist(),
            'mean_log_likelihood': mean_lls[0],
            'mean_null_log_likelihoods': mean_lls[1:].tolist(),
            'p-values': p_values
        }


class GaussianMixtureModel(object):

    def __init__(self, weights, gaussians, log_likelihood=None):
        self.weights = weights  # type: np.ndarray
        self.gaussians = gaussians  # type: list
        self._log_likelihood = log_likelihood  # type: float

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

    def calc_log_likelihood(self, data):
        return data.map(self.log_likelihood_summand_func()).sum()

    def log_likelihood(self, data=None):
        if self._log_likelihood is None:
            if data is None:
                raise Exception("Must provide data to calculate log-likelihood.")
            self._log_likelihood = self.calc_log_likelihood(data)
        return self._log_likelihood

    def stats_func(self):
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

    def calc_stats(self, data):
        return data.map(self.stats_func()).sum()

    def stats(self, data):
        lls = self.calc_stats(data)

        return {
            'num_dimensions': self.num_dimensions,
            'k': self.k,
            'log_likelihood': lls[0],
            'null_log_likelihoods': lls[1:]
        }
