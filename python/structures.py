# -*- coding: utf-8 -*-
# structures.py : Python wrappers of the C data structures
# author : Antoine Passemiers

import numpy as np
import ctypes

# Load the C interfaces
scythe = ctypes.cdll.LoadLibrary("../scythe/scythe.lib")

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

# Magic numbers
CLASSIFICATION_TASK = 0xF55A90
REGRESSION_TASK     = 0xF55A91

QUARTILE_PARTITIONING   = 0xB23A40
DECILE_PARTITIONING     = 0xB23A41
PERCENTILE_PARTITIONING = 0xB23A42

RANDOM_FOREST          = 0
COMPLETE_RANDOM_FOREST = 1
GRADIENT_BOOSTING      = 2


class Dataset(ctypes.Structure):
    """
    Structure of a dataset

    Fields
    ------
    data : np.ndarray[ndim = 2, dtype = np.double]
        Array containing the data samples
    n_rows : int
        Number of rows in the data array
    n_cols : int
        Number of columns in the data array
    """
    _fields_ = [
        ("data", ctypes.POINTER(ctypes.c_double)),
        ("n_rows", ctypes.c_size_t),
        ("n_cols", ctypes.c_size_t)]

    def __init__(self, data):
        """
        Ensures the data array is C-contiguous and stores the pointer
        to the first element.

        Parameters
        ----------
        data : np.ndarray[ndim = 2, dtype = np.double]
            Array containing the data samples
        """
        # Ensuring the data is C-contiguous
        self.np_data = np.ascontiguousarray(data, dtype = np.double)
        # Retrieving the number of rows and the number of columns
        self.n_rows = self.np_data.shape[0]
        self.n_cols = self.np_data.shape[1]
        # Retrieving the pointer to the first element
        self.data = self.np_data.ctypes.data_as(c_double_p)
    def __len__(self):
        """ Return the size / number of rows of the dataset """
        return self.n_rows


class Labels(ctypes.Structure):
    """
    Structure of the labels

    Fields
    ------
    data : np.ndarray[ndim = 1, dtype = np.double]
        One-dimensional array containing the target values,
        where each element data[i] corresponds to a row of data
        samples dataset[i, :].
    n_rows : int
        Number of rows in the labels array
    """
    _fields_ = [
        ("data", ctypes.POINTER(ctypes.c_double)),
        ("n_rows", ctypes.c_size_t)]

    def __init__(self, data):
        """
        Ensures the labels array is C-contiguous and stores the pointer
        to the first element.

        Parameters
        ----------
        data : np.ndarray[ndim = 1, dtype = np.double]
            Array containing the labels
        """
        # Ensuring the data is C-contiguous
        self.np_data = np.ascontiguousarray(data, dtype = np.double)
        # Retrieving the number of rows
        self.n_rows = self.np_data.shape[0]
        # Retrieving the pointer to the first element
        self.data = self.np_data.ctypes.data_as(c_double_p)
    def __len__(self):
        """ Return the number of targets """
        return self.n_rows


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
        ("nan_value", ctypes.c_double)]