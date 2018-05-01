import copy, random, re

import numpy as np
from pyspark import StorageLevel
import pyspark.sql.functions as F
import pyspark.sql.types as T
from scipy import stats

from gmm.model import ModelingService, Gaussian
from lib.sparse_vector_dataset_v1 import SparseVectorDimension
from lib.sparse_standard_scaler import SparseStandardScaler


pi = F.udf(lambda probability: float(probability[0]), T.FloatType())
one_minus_pi = F.udf(lambda probability: float(probability[1]), T.FloatType())


def loadBaseFeature(data):
    return SparseVectorDimension.fromDict(data)


def loadForwardSelectionIteration(data):
    result = copy.copy(data)
    result['features'] = [SparseVectorDimension.fromDict(f) for f in data['features']]
    result['gaussians'] = [Gaussian.fromDict(g) for g in data['gaussians']]
    return result


def baseFeatureToDict(data):
    return data.toDict()


def forwardSelectionIterationToDict(data):
    result = copy.copy(data)
    result['features'] = [f.toDict() for f in data['features']]
    result['gaussians'] = [g.toDict() for g in data['gaussians']]
    return result


def readJson(path):
    pass


def writeJsonWithRetry(rdd, savepath, mode='error'):
    pass


class GaussianMixtureModelMinerState(object):
    PRESELECTION_ALPHA = 0.2
    FORWARD_SELECTION_ALPHA = 0.01
    FINAL_SELECTION_ALPHA = 0.01

    @classmethod
    def load_or_initialize(cls, savepath, baseFeatures):
        # try:
        #     dbutils.fs.ls(savepath)    # TODO: fix dbutils
        # except Exception as err:
        #     if "java.io.FileNotFoundException" in err.message:
        #         return cls(baseFeatures)
        #     else:
        #         raise
        # else:
        #     return cls.load(savepath)
        return cls(baseFeatures)

    @classmethod
    def load(cls, savepath):
        attrs = readJson(savepath).collect()[0]  # TODO: fix readJson
        baseFeatures = [loadBaseFeature(d) for d in attrs['baseFeatures']]
        forwardSelectionIterations = [loadForwardSelectionIteration(d) for d in attrs['forwardSelectionIterations']]
        return cls(baseFeatures, forwardSelectionIterations)

    def __init__(self, baseFeatures, forwardSelectionIterations=[]):
        self.baseFeatures = baseFeatures
        self.forwardSelectionIterations = forwardSelectionIterations

    def save(self, spark_context, savepath, mode='overwrite'):
        result = {
            'baseFeatures': [baseFeatureToDict(x) for x in self.baseFeatures],
            'forwardSelectionIterations': [forwardSelectionIterationToDict(x) for x in self.forwardSelectionIterations],
        }
        rdd = spark_context.parallelize([result], numSlices=1)
        writeJsonWithRetry(rdd, savepath, mode=mode)  # TODO: fix writeJsonWithRetry

    def registerForwardSelection(self, data, savepath=None, mode='overwrite'):
        self.forwardSelectionIterations.append(data)
        if savepath:
            self.save(savepath, mode=mode)

    def baseModelFeatures(self):
        if len(self.forwardSelectionIterations) == 0:
            raise Exception("Must complete at least one forward selection iteration")
        lastIteration = self.forwardSelectionIterations[-1]
        return [
            f for p, f in zip(lastIteration['p-values'], lastIteration['features'])
            if p < self.FORWARD_SELECTION_ALPHA
        ]


class GaussianMixtureModelMiner(SparseStandardScaler):

    def __init__(self, sparseDataSet, columnFilter=None, state=None,
                 min_iterations=20, max_features=12,
                 step_size=5, alpha=0.01, max_training_iterations=100, memory=2):
        self._state = state or GaussianMixtureModelMinerState(self.fields)
        self.sparseDataSet = sparseDataSet
        self.columnFilter = columnFilter
        self.min_iterations = min_iterations
        self.max_features =  max_features
        self.step_size = step_size
        self.alpha = alpha
        self.max_training_iterations = max_training_iterations
        self.memory = memory
        self._standardized_columns = {}
        self._standardized_data_set = None
        self._has_retained_features = False

    @classmethod
    def _fields(cls, sparseDataSet, columnFilter=None):
        if columnFilter is None:
            return sparseDataSet.columns
        else:
            return filter(columnFilter, sparseDataSet.columns)

    @classmethod
    def load_or_initialize(cls, savepath, sparseDataSet, successFunct, columnFilter=None, baseFeatures=None):
        if baseFeatures is None:
            filteredBaseFeatures = cls._fields(sparseDataSet, columnFilter)
        elif columnFilter is None:
            filteredBaseFeatures = baseFeatures
        else:
            filteredBaseFeatures = filter(columnFilter, baseFeatures)

        state = GaussianMixtureModelMinerState.load_or_initialize(savepath, filteredBaseFeatures)
        return cls(sparseDataSet, successFunct, columnFilter, state)

    def fields(self):
        return self._state.baseFeatures

    def perform(self, savepath=None):
        self.performForwardSelections(savepath=savepath)

    def continueForwardSelection(self):
        last_p_values = self._state.forwardSelectionIterations[-1]['p-values']
        num_retained_features = len([p for p in last_p_values if p < self.alpha])
        return len(self._state.forwardSelectionIterations) < self.min_iterations \
            or num_retained_features < self.max_features

    def performForwardSelections(self, savepath=None):
        while self.continueForwardSelection():
            data = self.forwardSelection()
            self._state.registerForwardSelection(data, savepath=savepath)

    def selectModel(self, features, k_range, sampleFraction=None):
        data_set = self.generateMlDataSet(columns=features)
        if sampleFraction:
            data_set = data_set.sample(False, sampleFraction)
        training_set = data_set
        service = ModelingService(training_set, max_iterations=self.max_training_iterations)
        k, log_likelihoods = service.fit_k(k_range)
        gmm = service.get_gmm(self, k)
        stats = gmm.stats(service)
        result = {'features': features,
                  'k': k,
                  'weights': gmm.weights,
                  'gaussians': gmm.gaussians,
                  'p-values': stats.p_values,
                  'log_likelihoods': stats.log_likelihood,
                  'null_log_likelihoods': stats.null_log_likelihoods,
                  'k_range': k_range,
                  'k_range_log_likelihoods': log_likelihoods}
        training_set.unpersist()
        return result

    def forwardSelection(self):
        new_features = self.get_forward_select_features()
        k_range = self.get_forward_select_k_range()
        return self.selectModel(new_features, k_range)

    def get_retained_features(self):
        prev_iterations = self._state.forwardSelectionIterations
        previous_features = []
        for i in range(self.memory):
            if len(prev_iterations) > i:
                iteration = prev_iterations[-1 - i]
                previous_features = previous_features + [
                    iteration['features'][i]
                    for (i, p) in enumerate(iteration['p-values'])
                    if p < self.alpha
                ]
        if previous_features:
            self._has_retained_features = True
        return list(set(previous_features))

    def get_forward_select_features(self):
        previous_features = self.get_retained_features()
        sample_of_features = random.sample(self.fields(), self.step_size)
        return list(set(previous_features + sample_of_features))

    def get_forward_select_k_range(self):
        if self._has_retained_features:
            last_k = self._state.forwardSelectionIterations[-1]['k']
            return range(max([last_k - 2, 2]), last_k + 3)
        else:
            return range(2, 11)


