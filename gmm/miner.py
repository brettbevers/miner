import copy, random

from gmm.model import ModelingService, Gaussian
from ml_bme_scoring.transformer.legacy.sparse_vector_dataset_v1 import SparseVectorDimension
from ml_bme_scoring.mining.sparse_standard_scaler import SparseStandardScaler


def loadBaseFeature(data):
    return SparseVectorDimension.fromDict(data)


def loadForwardSelectionIteration(data):
    result = copy.deepcopy(data)
    result['features'] = [SparseVectorDimension.fromDict(f) for f in data['features']]
    for k, gmm_stats in data['k_range_stats'].items():
        result['k_range_stats'][k]['gaussians'] = [Gaussian.fromDict(g) for g in gmm_stats['gaussians']]
    return result


def baseFeatureToDict(data):
    return data.toDict()


def forwardSelectionIterationToDict(data):
    result = copy.deepcopy(data)
    result['features'] = [f.toDict() for f in data['features']]
    for k, gmm_stats in data['k_range_stats'].items():
        result['k_range_stats'][k]['gaussians'] = [g.toDict() for g in gmm_stats['gaussians']]
    return result


class GaussianMixtureModelMinerState(object):
    PRESELECTION_ALPHA = 0.2
    FORWARD_SELECTION_ALPHA = 0.01
    FINAL_SELECTION_ALPHA = 0.01

    @classmethod
    def load_or_initialize(cls, fs_adapter, savepath, baseFeatures):
        try:
            fs_adapter.ls(savepath)
        except Exception as err:
            if "java.io.FileNotFoundException" in err.message:
                return cls(baseFeatures)
            else:
                raise
        else:
            return cls.load(fs_adapter, savepath, fs_adapter=fs_adapter)
        return cls(baseFeatures, fs_adapter=fs_adapter)

    @classmethod
    def load(cls, fs_adapter, savepath):
        attrs = fs_adapter.read_json(savepath).collect()[0]
        baseFeatures = [loadBaseFeature(d) for d in attrs['baseFeatures']]
        forwardSelectionIterations = [loadForwardSelectionIteration(d) for d in attrs['forwardSelectionIterations']]
        return cls(baseFeatures, forwardSelectionIterations, fs_adapter=fs_adapter)

    def __init__(self, baseFeatures, forwardSelectionIterations=[], fs_adapter=None):
        self.fs_adapter = fs_adapter
        self.baseFeatures = baseFeatures
        self.forwardSelectionIterations = forwardSelectionIterations

    def save(self, savepath, mode='overwrite'):
        result = {
            'baseFeatures': [baseFeatureToDict(x) for x in self.baseFeatures],
            'forwardSelectionIterations': [forwardSelectionIterationToDict(x) for x in self.forwardSelectionIterations],
        }
        rdd = self.fs_adapter.spark_context.parallelize([result], numSlices=1)
        self.fs_adapter.write_json_with_retry(rdd, savepath, mode=mode)

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

    def __init__(self, sparseDataSet, columnFilter=None, state=None, fs_adapter=None,
                 min_iterations=20, max_features=12,
                 step_size=5, alpha=0.01, max_training_iterations=200,
                 memory=2, k_search_radius=2, initial_k_range=range(1, 5)):
        self._state = state or GaussianMixtureModelMinerState(
            self.__class__._fields(sparseDataSet, columnFilter),
            fs_adapter=fs_adapter
        )
        self.sparseDataSet = sparseDataSet
        self.columnFilter = columnFilter
        self.min_iterations = min_iterations
        self.max_features = max_features
        self.step_size = step_size
        self.alpha = alpha
        self.max_training_iterations = max_training_iterations
        self.memory = memory
        self.k_search_radius = k_search_radius
        self.initial_k_range = initial_k_range
        self._standardized_columns = {}
        self._standardized_data_set = None

    @classmethod
    def _fields(cls, sparseDataSet, columnFilter=None):
        if columnFilter is None:
            return sparseDataSet.columns
        else:
            return filter(columnFilter, sparseDataSet.columns)

    @classmethod
    def load_or_initialize(cls, fs_adapter, savepath, sparseDataSet, columnFilter=None, baseFeatures=None, **kwargs):
        if baseFeatures is None:
            filteredBaseFeatures = cls._fields(sparseDataSet, columnFilter)
        elif columnFilter is None:
            filteredBaseFeatures = baseFeatures
        else:
            filteredBaseFeatures = filter(columnFilter, baseFeatures)

        state = GaussianMixtureModelMinerState.load_or_initialize(fs_adapter, savepath, filteredBaseFeatures)
        return cls(sparseDataSet, columnFilter, state, **kwargs)

    def fields(self):
        return self._state.baseFeatures

    def perform(self, savepath=None):
        self.performForwardSelections(savepath=savepath)

    def continueForwardSelection(self):
        return len(self._state.forwardSelectionIterations) < self.min_iterations \
            or len(self.get_retained_features()) < self.max_features

    def performForwardSelections(self, savepath=None):
        while self.continueForwardSelection():
            data = self.forwardSelection()
            if data:
                self._state.registerForwardSelection(data, savepath=savepath)

    def selectModel(self, features, k_range, sampleFraction=None):
        k_range = sorted(k_range)
        data_set = self.generateMlDataSet(columns=features)
        if sampleFraction:
            data_set = data_set.sample(False, sampleFraction)
        training_set = data_set.persist()
        service = ModelingService(training_set, max_iterations=self.max_training_iterations)

        result_stats = {}
        for k in k_range:
            ll = service.get_log_likelihood(k, retry=False)
            gmm = service.get_gmm(k)
            result_stats[k] = {
                'weights': gmm.weights,
                'gaussians': gmm.gaussians,
                'log_likelihood': ll
            }

        # Sometimes using the pseudo-inverse leads to positive values,
        # like when the covariance matrix is not even nearly invertible.
        # We exclude these pathological cases.
        valid_k = [
            (k, result_stats[k]['log_likelihood'])
            for k in k_range
            if result_stats[k]['log_likelihood'] < 0
        ]

        if len(valid_k) < 2:
            return None

        selected_k = max([
            (k, ll - valid_k[i - 1][1])
            for i, (k, ll) in enumerate(valid_k)
            if i > 0
        ], key=lambda x: x[1])[0]

        selected_stats = service.get_stats(selected_k, calc_p_values=True)

        result = {'features': features,
                  'k_range': k_range,
                  'k_range_stats': result_stats,
                  'selected_k': selected_k,
                  'selected_k_stats': selected_stats,
                  }
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
                iteration_features = iteration['features']
                p_values = iteration['selected_k_stats']['p-values']
                for f, p in zip(iteration_features, p_values):
                    if p < self.alpha:
                        previous_features.append(f)
        return list(set(previous_features))

    def get_forward_select_features(self):
        previous_features = self.get_retained_features()
        fields = self.fields()
        step_size = min([len(fields), self.step_size])
        sample_of_features = random.sample(fields, step_size)
        return list(set(previous_features + sample_of_features))

    def get_forward_select_k_range(self):
        if self._state.forwardSelectionIterations:
            last_k = self._state.forwardSelectionIterations[-1]['selected_k']
            start_k = max([last_k - self.k_search_radius, 1])
            end_k = last_k + self.k_search_radius
            return range(start_k, end_k + 1)
        else:
            return self.initial_k_range


