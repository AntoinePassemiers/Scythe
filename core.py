# -*- coding: utf-8 -*-

import numpy as np
import ctypes

scythe = ctypes.cdll.LoadLibrary("scythe/scythe.lib")

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

CLASSIFICATION_TASK = 0xF55A90
REGRESSION_TASK     = 0xF55A91

QUARTILE_PARTITIONING   = 0xB23A40
DECILE_PARTITIONING     = 0xB23A41
PERCENTILE_PARTITIONING = 0xB23A42

class Dataset(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.POINTER(ctypes.c_double)),
        ("n_rows", ctypes.c_size_t),
        ("n_cols", ctypes.c_size_t)]

    def __init__(self, data):
        np_data = np.ascontiguousarray(data, dtype = np.double)
        self.n_rows = np_data.shape[0]
        self.n_cols = np_data.shape[1]

        self.data = np_data.ctypes.data_as(c_double_p)

class Labels(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.POINTER(ctypes.c_double)),
        ("n_rows", ctypes.c_size_t)]

    def __init__(self, data):
        data = np.ascontiguousarray(data, dtype = np.double)
        self.np_data = data.astype(np.int)
        self.n_rows = self.np_data.shape[0]

        self.data = data.ctypes.data_as(c_double_p)

class TreeConfig(ctypes.Structure):
    _fields_ = [
        ("task", ctypes.c_int),
        ("is_incremental", ctypes.c_int),
        ("min_threshold", ctypes.c_double),
        ("max_height", ctypes.c_size_t),
        ("n_classes", ctypes.c_size_t),
        ("max_nodes", ctypes.c_size_t),
        ("partitioning", ctypes.c_int),
        ("nan_value", ctypes.c_double)]

class ForestConfig(ctypes.Structure):
    _fields_ = [
        ("task", ctypes.c_int),
        ("n_classes", ctypes.c_size_t),
        ("score_metric", ctypes.c_int),
        ("n_iter", ctypes.c_size_t),
        ("max_n_trees", ctypes.c_size_t),
        ("max_n_nodes", ctypes.c_size_t),
        ("learning_rate", ctypes.c_float),
        ("n_leaves", ctypes.c_size_t),
        ("n_jobs", ctypes.c_size_t),
        ("n_samples_per_leaf", ctypes.c_size_t),
        ("regularization", ctypes.c_int),
        ("bagging_fraction", ctypes.c_float),
        ("early_stopping_round", ctypes.c_size_t),
        ("boosting_method", ctypes.c_int),
        ("max_depth", ctypes.c_int),
        ("l1_lambda", ctypes.c_float),
        ("l2_lambda", ctypes.c_float),
        ("seed", ctypes.c_float),
        ("verbose", ctypes.c_int),
        ("nan_value", ctypes.c_double)]


if __name__ == "__main__":
    config = TreeConfig()
    config.is_incremental = False
    config.min_threshold = 1e-06
    config.max_height = 50
    config.n_classes = 3
    config.max_nodes = 30
    config.partitioning = PERCENTILE_PARTITIONING
    config.nan_value = -1.0

    
    X_train = np.asarray(np.array([
        [0, 0, 0], # 0    1    5.6   6.65
        [0, 0, 1], # 0    0    7.8   7.8
        [1, 0, 0], # 1    1    4.2   4.2
        [2, 0, 0], # 1    1    3.5   3.5
        [2, 1, 0], # 2    1.5  9.8   7.9
        [2, 1, 1], # 0    0    5.4   5.4
        [1, 1, 1], # 1    1    2.1   2.1
        [0, 0, 0], # 2    1    7.7   6.65
        [0, 1, 0], # 2    2    8.8   8.8
        [2, 1, 0], # 1    1.5  6.0   7.9
        [0, 1, 1], # 1    1    5.7   5.7
        [1, 0, 1], # 1    1    7.0   7.0
        [1, 1, 0], # 1    1    6.9   6.9
        [2, 0, 1]  # 0    0    6.3   6.3
    ]), dtype = np.double)
    
    #X_train = np.random.rand(14, 3)

    y_train = np.array([0, 0, 1, 1, 2, 0, 1, 2, 2, 1, 1, 1, 1, 0])
    X_test = X_train

    dataset = Dataset(X_train)
    labels  = Labels(y_train)
    testset = Dataset(X_test)

    # CLASSIFICATION TREE
    config.task = CLASSIFICATION_TASK
    tree_addr = scythe.fit_classification_tree(ctypes.byref(dataset), ctypes.byref(labels), ctypes.byref(config))
    preds_addr = scythe.tree_classify(ctypes.byref(testset), ctypes.c_void_p(tree_addr), ctypes.byref(config))
    preds_p = ctypes.cast(preds_addr, c_float_p)
    preds = np.ctypeslib.as_array(preds_p, shape = (14, 3))
    print("\n")
    print(preds)

    # REGRESSION TREE
    targets = np.array([5.6, 7.8, 4.2, 3.5, 9.8, 5.4, 2.1, 7.7, 8.8, 6.0, 5.7, 7.0, 6.9, 6.3])
    targets  = Labels(targets)
    config.task = REGRESSION_TASK
    tree_addr = scythe.fit_regression_tree(
        ctypes.byref(dataset), 
        ctypes.byref(targets), 
        ctypes.byref(config))
    preds_addr = scythe.tree_predict(
        ctypes.byref(testset), 
        ctypes.c_void_p(tree_addr), 
        ctypes.byref(config))
    preds_p = ctypes.cast(preds_addr, c_double_p)
    preds = np.ctypeslib.as_array(preds_p, shape = (14,))
    print("\n")
    print(preds)

    # CLASSIFICATION FOREST
    n_instances = 1000
    dataset = Dataset(np.random.rand(n_instances, 3))
    labels  = Labels(np.random.randint(3, size = n_instances))

    fconfig = ForestConfig()
    fconfig.task = CLASSIFICATION_TASK
    fconfig.n_classes = 3
    fconfig.max_depth = 4
    fconfig.max_n_nodes = 500
    fconfig.nan_value = -1.0
    fconfig.n_iter    = 50
    fconfig.learning_rate = 0.05
    forest_addr = scythe.fit_classification_forest(
        ctypes.byref(dataset), 
        ctypes.byref(labels), 
        ctypes.byref(fconfig))


    print("Finished")