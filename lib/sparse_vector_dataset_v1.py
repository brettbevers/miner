import binascii
import copy, json, marshal, types

import pyspark.mllib as mllib
from pyspark.mllib.stat import Statistics
from pyspark.mllib.linalg import SparseVector as MllibSparseVector


class SparseVectorDimension(object):

    def __init__(self, name, index=None, ind_vars=None, function=None):
        if index is None and ind_vars is None and function is None:
            raise Exception("Must specify either index, ind_vars or a function: {}".format(name))
        self.index = index
        self.name = name
        self.ind_vars = ind_vars
        self.function = function
        self._size = None

    def __hash__(self):
        """Override the default hash behavior (that returns the id or the object)"""
        # canonicalize ind_vars
        if self.ind_vars is None:
            ind_vars_part = None
        else:
            sorted_ind_vars = sorted(self.ind_vars, key=lambda var: var.__hash__())
            ind_vars_part = tuple(sorted_ind_vars)
        # extract bytecode from function
        if self.function is None:
            function_part = None
        else:
            function_part = self.function.func_code.co_code
        return hash((self.index, ind_vars_part, function_part))

    def __eq__(self, other):
        """Override the default Equals behavior"""
        if isinstance(other, self.__class__):
            return self.__hash__() == other.__hash__()
        return NotImplemented

    def __ne__(self, other):
        """Define a non-equality test"""
        return not self.__eq__(other)

    @classmethod
    def loads(cls, line):
        d = json.loads(line)
        return cls.fromDict(d)

    @classmethod
    def make_cell(cls, val=None):
        x = val
        def closure():
            return x
        return closure.__closure__[0]

    @classmethod
    def fromDict(cls, d):
        attrs = {'name': d['name']}
        if 'index' in d:
            attrs['index'] = d['index']
        if 'ind_vars' in d:
            attrs['ind_vars'] = [ cls.fromDict(v) for v in d['ind_vars'] ]
        if 'function' in d:
            f = types.FunctionType(
                marshal.loads(binascii.a2b_qp(d['function'][0])),
                globals(),
                name=None,
                argdefs=None,
                closure=tuple([ cls.make_cell(val) for val in d['function'][1] ])
            )
            attrs['function'] = f
        return cls(**attrs)

    def toDict(self):
        attrs = {'name': self.name}
        if self.index is not None:
            attrs['index'] = self.index
        if self.ind_vars is not None:
            attrs['ind_vars'] = [v.toDict() for v in self.ind_vars]
        if self.function is not None:
            attrs['function'] = [
                binascii.b2a_qp(marshal.dumps(self.function.func_code)),
                [cell.cell_contents for cell in (self.function.func_closure or [])]
            ]
        return attrs

    def dumps(self):
        attrs = self.toDict()
        return json.dumps(attrs)

    def evaluate(self):
        if self.index is not None:
            i = self.index
            return lambda vec: vec[i]
        elif self.ind_vars is None:
            return self.function
        elif self.function is None:
            ind_vars = [v.evaluate() for v in self.ind_vars]
            return lambda vec: reduce(lambda x, y: x * y, [v(vec) for v in ind_vars])
        else:
            ind_vars = [v.evaluate() for v in self.ind_vars]
            f = self.function
            return lambda vec: f(*[v(vec) for v in ind_vars])


class SparseVectorDatasetV1(object):
    COLUMNS_DIR = 'columns/'
    DATA_DIR = 'data/'
    NULL_VALUES = [0, None]
    PRIMARY_KEY = 'customer_id'

    def __init__(self, columns, rddOfSparseVectors, primaryKey=None):
        self.columns = self.normalizeColumns(columns)
        self.data = rddOfSparseVectors
        self.primaryKey = primaryKey or self.PRIMARY_KEY
        self._nameToIndexMap = None
        self._nameToColumnMap = None

    def normalizeColumns(self, columns):
        """Convert all incoming columns to SparseVectorDimension instances."""
        cols = sorted([
            (c if isinstance(c, SparseVectorDimension) else SparseVectorDimension(name=c, index=i))
            for i, c in enumerate(columns)
        ], key=lambda col: col.index)
        return cols

    @classmethod
    def loadColumns(cls, path):
        """Load a set of SparseVectorDimensions from file."""
        path_contents = dbutils.fs.ls(path)  # TODO: fix dbutils usage
        columns_path = next(x.path for x in path_contents if x.name == cls.COLUMNS_DIR)
        cols = [ SparseVectorDimension.loads(s) for s in sc.textFile(columns_path).collect() ]
        return sorted(cols, key=lambda col: col.index)

    @classmethod
    def loadData(cls, path):
        """Load a set of sparse vectors from a file and convert to mllib SparseVectors."""
        path_contents = dbutils.fs.ls(path) # TODO: fix dbutils usage
        data_path = next(x.path for x in path_contents if x.name == cls.DATA_DIR)
        #         return sc.textFile(data_path).map(mllib.linalg.SparseVector.parse)
        return sc.textFile(data_path).map(MllibSparseVector.parse)

    @classmethod
    def load(cls, path):
        """Load and return a SparseVectorDataset from the given path."""
        columns = cls.loadColumns(path)
        data = cls.loadData(path)
        return cls(columns, data)

    @classmethod
    def fromDataFrame(cls, sourceDF):
        """
        Convert a dataframe to a SparseVectorDataset with columns that are
        keyed on the dataframe primary key.
        """
        col_names = sorted(sourceDF.columns)
        size = len(col_names)
        columns = [SparseVectorDimension(index=index, name=col) for index, col in enumerate(col_names)]
        null_values = cls.NULL_VALUES
        primaryKey = cls.PRIMARY_KEY
        def convertRow(row):
            d = {index: row[col] for index, col in enumerate(col_names) if not row[col] in null_values}
            return [row[primaryKey], (size, d.keys(), d.values())]
        data = sourceDF.rdd.map(convertRow)
        return cls(columns, data)

    @classmethod
    def sparseVectorsFromDataFrame(cls, sourceDF):
        """Convert a dataframe to a SparseVectorDataset without adding a key."""
        col_names = sorted(sourceDF.columns)
        size = len(col_names)
        columns = [SparseVectorDimension(index=index, name=col) for index, col in enumerate(col_names)]
        null_values = cls.NULL_VALUES
        def convertRow(row):
            d = {index: row[col] for index, col in enumerate(col_names) if not row[col] in null_values}
            return MllibSparseVector(size, d)
        data = sourceDF.rdd.map(convertRow)
        return cls(columns, data)

    @classmethod
    def generateTransformation(cls, columns):
        """
        Returns a function which transforms a given vector per column
        based on the transform from the corresponding SparseVectorDimension.
        """
        new_size = len(columns)
        functs = [c.evaluate() for c in columns]
        null_values = cls.NULL_VALUES
        def f(vec):
            result = {}
            for index, funct in enumerate(functs):
                val = funct(vec)
                if not val in null_values:
                    result[index] = val
            return MllibSparseVector(new_size, result)
        return f

    def getIndex(self, col_name):
        """
        Find the index into the SparseVectorDimensions for a given column name.
        """
        if self._nameToIndexMap is None:
            self._nameToIndexMap = {c.name: c.index for c in self.columns}
        return self._nameToIndexMap[col_name]

    def take(self, num):
        return self.data.take(num)

    def collect(self):
        return self.data.collect()

    def count(self):
        return self.data.count()

    def cache(self):
        result = self.copy()
        result.data = result.data.cache()
        return result

    def filter(self, predicate):
        new_data = self.data.filter(predicate)
        result = self.copy()
        result.data = new_data
        return result

    def copy(self):
        return SparseVectorDatasetV1(
            copy.deepcopy(self.columns),
            self.data,
            copy.copy(self.primaryKey)
        )

    def saveColumns(self, path):
        lines = map(lambda col: col.dumps(), self.columns)
        sc.parallelize(lines, numSlices=1).saveAsTextFile(path)  # TODO: update SparkContext references

    def saveData(self, path):
        self.data.map(lambda vec: str(vec)).saveAsTextFile(path)

    def save(self, path):
        col_path = path + "columns" if path.endswith('/') else path + "/columns"
        data_path = path + "data" if path.endswith('/') else path + "/data"
        self.saveColumns(col_path)
        self.saveData(data_path)

    def getColumnByName(self, name):
        if self.hasColumnByName(name):
            return self._nameToColumnMap[name]
        else:
            raise ValueError("No column with name '{}'".format(name))

    def hasColumnByName(self, name):
        if self._nameToColumnMap is None:
            self._nameToColumnMap = {c.name: c for c in self.columns}
        return name in self._nameToColumnMap

    def size(self):
        """Returns the number of SparseVectorDimensions."""
        if self._size is None:
            self._size = len(self.columns)
        return self._size

    def transform(self, new_columns):
        """
        Transforms the `data` of this SparseVectorDataset by applying
        the transformation functions associated with the `new_columns`
        and returns a new SparseVectorDataset.
        """
        transformation = self.generateTransformation(new_columns)
        result_columns = [SparseVectorDimension(c.name, index=i) for i, c in enumerate(new_columns)]
        result_data = self.data.map(transformation)
        return SparseVectorDatasetV1(result_columns, result_data)

    def colStats(self):
        """Return descriptive stats for all of our columns."""
        return Statistics.colStats(self.data)

    def columns(self):
        return self.getShard(0).columns
