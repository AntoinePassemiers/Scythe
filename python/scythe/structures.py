# -*- coding: utf-8 -*-
# structures.py : Python wrappers of the C data structures
# author : Antoine Passemiers

import numpy as np
import ctypes

# Load the C interfaces
# scythe = ctypes.cdll.LoadLibrary("scythe.lib")
# TODO
scythe = ctypes.cdll.LoadLibrary("C://Users/Xanto183/git/Scythe/src/scythe.dll")

# C types
c_int_p        = ctypes.POINTER(ctypes.c_int)
c_uint_p       = ctypes.POINTER(ctypes.c_uint)
c_int16_p      = ctypes.POINTER(ctypes.c_int16)
c_uint16_p     = ctypes.POINTER(ctypes.c_uint16)
c_int32_p      = ctypes.POINTER(ctypes.c_int32)
c_uint32_p     = ctypes.POINTER(ctypes.c_uint32)
c_int64_p      = ctypes.POINTER(ctypes.c_int64)
c_uint64_p     = ctypes.POINTER(ctypes.c_uint64)
c_size_t_p     = ctypes.POINTER(ctypes.c_size_t)
c_float_p      = ctypes.POINTER(ctypes.c_float)
c_double_p     = ctypes.POINTER(ctypes.c_double)
c_longdouble_p = ctypes.POINTER(ctypes.c_longdouble)

# Data types
data_np = np.double
data_t  = ctypes.c_double
data_p  = ctypes.POINTER(data_t)
target_np = np.double
target_t  = ctypes.c_double
target_p  = ctypes.POINTER(target_t)

# Magic numbers
CLASSIFICATION_TASK = 0xF55A90
REGRESSION_TASK     = 0xF55A91

MLOG_LOSS = 0x7711A0
MSE       = 0xC97B00

QUARTILE_PARTITIONING   = 0xB23A40
DECILE_PARTITIONING     = 0xB23A41
PERCENTILE_PARTITIONING = 0xB23A42

RANDOM_FOREST          = 0
COMPLETE_RANDOM_FOREST = 1
GRADIENT_BOOSTING      = 2

REG_L1 = 0x778C10
REG_L2 = 0x778C11

ADABOOST          = 0x28FE90
GRADIENT_BOOSTING = 0x28FE91

DTYPE_PROBA  = 0
DTYPE_DATA   = 1
DTYPE_UINT_8 = 2


class Dataset(ctypes.Structure):
    """
    Structure of a dataset

    Fields
    ------
    data : np.ndarray[ndim = 2]
        Array containing the data samples
    n_rows : int
        Number of rows in the data array
    n_cols : int
        Number of columns in the data array
    """
    _fields_ = [
        ("data", ctypes.c_void_p),
        ("n_rows", ctypes.c_size_t),
        ("n_cols", ctypes.c_size_t)]

    def __init__(self, data):
        """
        Ensures the data array is C-contiguous and stores the pointer
        to the first element.

        Parameters
        ----------
        data : np.ndarray[ndim = 2]
            Array containing the data samples
        """
        # Ensuring the data is C-contiguous
        self.np_data = np.ascontiguousarray(data, dtype = data_np)
        # Retrieving the number of rows and the number of columns
        self.n_rows = self.np_data.shape[0]
        self.n_cols = self.np_data.shape[1]
        # Retrieving the pointer to the first element
        self.data = self.np_data.ctypes.data_as(ctypes.c_void_p)
    def __len__(self):
        """ Return the size / number of rows of the dataset """
        return self.n_rows


class Labels(ctypes.Structure):
    """
    Structure of the labels

    Fields
    ------
    data : np.ndarray[ndim = 1]
        One-dimensional array containing the target values,
        where each element data[i] corresponds to a row of data
        samples dataset[i, :].
    n_rows : int
        Number of rows in the labels array
    """
    _fields_ = [
        ("data", target_p),
        ("n_rows", ctypes.c_size_t)]

    def __init__(self, data):
        """
        Ensures the labels array is C-contiguous and stores the pointer
        to the first element.

        Parameters
        ----------
        data : np.ndarray[ndim = 1]
            Array containing the labels
        """
        m, M = np.min(data), np.max(data)
        # TODO: different treatment in case of classification and regression
        # assert(m == 0)
        self.n_classes = M + 1
        # Ensuring the data is C-contiguous
        self.np_data = np.ascontiguousarray(data, dtype = target_np)
        # Retrieving the number of rows
        self.n_rows = self.np_data.shape[0]
        # Retrieving the pointer to the first element
        self.data = self.np_data.ctypes.data_as(target_p)
    def __len__(self):
        """ Return the number of targets """
        return self.n_rows


class MDDataset(ctypes.Structure):
    """
    Multi-dimensional dataset, with a maximum of 7 dimensions

    Fields
    ------
    data : np.ndarray
        Multi-dimensional array containing the data samples
    """
    _fields_ = [
        ("data", data_p),
        ("n_dims", ctypes.c_size_t),
        ("dims", ctypes.c_size_t * 7),
        ("dtype", ctypes.c_int)]

    def __init__(self, data):
        """
        Ensures the data array is C-contiguous and stores the pointer
        to the first element.

        Parameters
        ----------
        data : np.ndarray
            Array containing the data samples
        """
        # Retrieving the raw data type
        self.dtype = DTYPE_UINT_8 if data.dtype == np.uint8 else DTYPE_DATA
        # Ensuring the data is C-contiguous
        self.np_data = np.ascontiguousarray(data, dtype = data_np)
        # Retrieving the pointer to the first element
        self.data = self.np_data.ctypes.data_as(data_p)
        # Retrieving the number of dimensions
        self.n_dims = len(self.np_data.shape)
        # Retrieving the size of each dimension
        for i in range(7):
            self.dims[i] = self.np_data.shape[i] if i < self.n_dims else 0


class TreeConfig(ctypes.Structure):
    """
    Configuration of a single tree

    Fields
    ------
    task : int
        Type of problem to solve (Regression or classification)
    is_incremental : int
        Indicates whether the tree can learn incrementally or not
    is_complete_random : int
        Indicates whether the tree is complete-random or not.
        A complete-random tree is grown by picking a random split value
        for each feature considered.
    min_threshold : double
        Minimum gain in the objective function while adding a new node.
        The algorithms stops if this value is not reached.
    max_height : int
        Maximum height of the tree
    n_classes : int
        Number of classes (in case of a classification task)
    max_nodes : int
        Maximum number of nodes
    partitioning : int
        Number of split values to consider in a partition.
        By default, 100 split values are considered.
    nan_value : double
        Value corresponding to NaNs in the dataset.
    """
    _fields_ = [
        ("task", ctypes.c_int),
        ("is_incremental", ctypes.c_int),
        ("min_threshold", ctypes.c_double),
        ("max_height", ctypes.c_size_t),
        ("max_n_features", ctypes.c_size_t),
        ("n_classes", ctypes.c_size_t),
        ("max_nodes", ctypes.c_size_t),
        ("partitioning", ctypes.c_int),
        ("nan_value", ctypes.c_double),
        ("is_complete_random", ctypes.c_int)]


class ForestConfig(ctypes.Structure):
    """
    Configuration of a gradient boosted forest

    Fields
    ------
    task : int
        Type of problem to solve (Regression or classification)
    n_classes : int
        Number of classes (in case of a classification task)
    score_metric : int
        Type of score metric for evaluating each tree fit
    n_iter : int
        Number of iterations
    max_n_trees : int
        Maximum number of trees in the forest, subject to the following
        constraint : n_iter * n_trees_per_iter <= max_n_trees
    max_n_nodes : int
        Maximum number of nodes per tree
    learning_rate : float
        Learning rate of the boosting algorithm
    n_leaves : int
        Maximum number of leaves per tree
    n_jobs : int
        Number of threads running in parallel
    n_samples_per_leaf : int
        Minimum number of samples per leaf
    regularization : int
        Regularization method
    """
    _fields_ = [
        ("type", ctypes.c_int),
        ("task", ctypes.c_int),
        ("n_classes", ctypes.c_size_t),
        ("score_metric", ctypes.c_int),
        ("n_iter", ctypes.c_size_t),
        ("max_n_trees", ctypes.c_size_t),
        ("max_n_nodes", ctypes.c_size_t),
        ("max_n_features", ctypes.c_size_t),
        ("learning_rate", ctypes.c_float),
        ("n_leaves", ctypes.c_size_t),
        ("n_jobs", ctypes.c_size_t),
        ("n_samples_per_leaf", ctypes.c_size_t),
        ("regularization", ctypes.c_int),
        ("bag_size", ctypes.c_size_t),
        ("early_stopping_round", ctypes.c_size_t),
        ("boosting_method", ctypes.c_int),
        ("max_depth", ctypes.c_int),
        ("l1_lambda", ctypes.c_float),
        ("l2_lambda", ctypes.c_float),
        ("seed", ctypes.c_float),
        ("verbose", ctypes.c_int),
        ("nan_value", ctypes.c_double),
        ("min_threshold", ctypes.c_double)]

    def __init__(self, ftype = RANDOM_FOREST, task = CLASSIFICATION_TASK,
                 n_classes = 2, score_metric = MLOG_LOSS, n_iter = 100,
                 max_n_trees = 150, max_n_nodes = 9999999, max_n_features = 9999999,
                 learning_rate = 0.001, n_leaves = 9999999, n_jobs = 1, n_samples_per_leaf = 50,
                 regularization = REG_L1, bag_size = 100, early_stopping_round = 300,
                 boosting_method = GRADIENT_BOOSTING, max_depth = 100, l1_lambda = 0.1,
                 l2_lambda = 0.1, seed = 4.0, verbose = True, nan_value = np.nan):
        self.type = ftype
        self.task = task
        self.n_classes = n_classes
        self.score_metric = score_metric
        self.n_iter = n_iter
        self.max_n_trees = max_n_trees
        self.max_n_nodes = max_n_nodes
        self.max_n_features = max_n_features
        self.learning_rate = learning_rate
        self.n_leaves = n_leaves
        self.n_jobs = n_jobs
        self.n_samples_per_leaf = n_samples_per_leaf
        self.regularization = regularization
        self.bag_size = bag_size
        self.early_stopping_round = early_stopping_round
        self.boosting_method = boosting_method
        self.max_depth = max_depth
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda
        self.seed = seed
        self.verbose = verbose
        self.nan_value = nan_value




class LayerConfig(ctypes.Structure):
    """
    Configuration of a deep forest's layer

    Fields
    ------
    fconfig : ForestConfig
        Configuration of each forest contained in the layer
    n_forests : size_t
        Number of forests in the layer
    forest_type : int
        Type of forests to be grown
    """
    _fields_ = [
        ("fconfig", ForestConfig),
        ("n_forests", ctypes.c_size_t),
        ("forest_type", ctypes.c_int)]

    def __init__(self, fconfig, n_forests, forest_type):
        self.fconfig = fconfig
        self.n_forests = n_forests
        self.forest_type = forest_type