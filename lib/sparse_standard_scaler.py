import numpy as np
import pyspark.ml as ml
from pyspark.sql import Row

from lib.sparse_vector_dataset_v1 import (SparseVectorDimension, SparseVectorDatasetV1)


class SparseStandardScaler(object):
    """
    Standardizes the dimensions of a SparseVectorDataset so their values
    have zero mean and unit variance.
    """

    def __init__(self, sparseDataSet=None, standardized_columms=None):
        self.sparseDataSet = sparseDataSet
        self._standardized_columns = standardized_columms or {}
        self._standardized_data_set = None

    def generateStandardizationFunction(self, col, mean, nf):
        """
        Returns a function which standardizes a value to zero mean and
        unit variance.
        """

        def f(value):
            if value is None:
                value = 0.
            return float((value - mean) / nf)

        return f

    def generateStandardizedColumns(self, data_set, columns):
        """
        Returns a set of SparseVectorDimensions which are each assigned
        a transformation function appropriate for standardizing the
        respective dimension.
        """
        cstat = data_set.colStats()
        stdDevs = np.sqrt(cstat.variance())
        means = cstat.mean()
        normalizingFactors = map(lambda x: x if x != 0.0 else 1.0, stdDevs)
        result = []
        for col, mean, nf in zip(columns, means, normalizingFactors):
            standardized = self.generateStandardizationFunction(col, mean, nf)
            result.append(SparseVectorDimension(
                name=col.name,
                ind_vars=[col],
                function=standardized
            ))
        return result

    def standardizedDataSet(self):
        if self._standardized_data_set is None:
            columns = self.sparseDataSet.columns
            stdzd_cols = self.generateStandardizedColumns(self.sparseDataSet, columns)
            self._standardized_data_set = self.sparseDataSet.transform(stdzd_cols)
        return self._standardized_data_set

    def standardize(self, columns):
        uncached_columns = [col for col in columns if col not in self._standardized_columns]
        if len(uncached_columns) > 0:
            data_set = self.sparseDataSet.transform(uncached_columns)
            stdzd_cols = self.generateStandardizedColumns(data_set, uncached_columns)
            for old_col, new_col in zip(uncached_columns, stdzd_cols):
                self._standardized_columns[old_col] = new_col
        return [self._standardized_columns[col] for col in columns]

    def generateMlRow(self, successFunct=None, columns=None, labelColName="label",
                      featureColName="features", standardizeCols=True):
        """
        Returns a function which standardizes a row's data and returns a
        potentially labeled `Row`.
        """
        if columns is None:
            columns = self.sparseDataSet.columns

        if standardizeCols:
            standardized_columns = self.standardize(columns)
        else:
            standardized_columns = columns

        transformation = SparseVectorDatasetV1.generateTransformation(standardized_columns)
        if successFunct:
            def f(vec):
                trans_vec = transformation(vec)
                attrs = {
                    featureColName: ml.linalg.SparseVector(trans_vec.size, trans_vec.indices, trans_vec.values),
                    labelColName: successFunct(vec)
                }
                return Row(**attrs)
        else:
            def f(vec):
                trans_vec = transformation(vec)
                attrs = {
                    featureColName: ml.linalg.SparseVector(trans_vec.size, trans_vec.indices, trans_vec.values)
                }
                return Row(**attrs)
        return f

    def generateMlDataSet(self, successFunct=None, columns=None, labelColName="label",
                          featureColName="features", standardizeCols=True):
        """
        Returns a DataFrame where the rows have been transformed according
        to the dataset transformation.
        """
        generate_ml_row = self.generateMlRow(successFunct, columns, labelColName=labelColName,
                                             featureColName=featureColName, standardizeCols=standardizeCols)
        return self.sparseDataSet.data \
            .map(generate_ml_row) \
            .toDF()
