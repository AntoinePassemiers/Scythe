# -*- coding: utf-8 -*-

import numpy as np
import ctypes

scythe = ctypes.cdll.LoadLibrary("scythe/scythe.lib")

c_double_p = ctypes.POINTER(ctypes.c_double)
c_float_p  = ctypes.POINTER(ctypes.c_float)
c_int_p    = ctypes.POINTER(ctypes.c_int)

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
        data = np.asarray(data, dtype = np.double)
        self.np_data = data.astype(np.double)
        self.n_rows = self.np_data.shape[0]
        self.n_cols = self.np_data.shape[1]

        p = ctypes.POINTER(ctypes.c_double)
        self.data = data.ctypes.data_as(p)

class Labels(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.POINTER(ctypes.c_double)),
        ("n_rows", ctypes.c_size_t)]

    def __init__(self, data):
        data = np.asarray(data, dtype = np.double)
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
        ("n_iter", ctypes.c_size_t),
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
        ("verbose", ctypes.c_int)]


if __name__ == "__main__":
    config = TreeConfig()
    config.is_incremental = False
    config.threshold = 1e-06
    config.max_height = 50
    config.n_classes = 3
    config.max_nodes = 30
    config.partitioning = PERCENTILE_PARTITIONING
    config.nan_value = -1.0

    
    X_train = np.asarray(np.array([
        [0, 0, 0], # 0    1
        [0, 0, 1], # 0    0
        [1, 0, 0], # 1    1
        [2, 0, 0], # 1    1
        [2, 1, 0], # 2    1.5
        [2, 1, 1], # 0    0
        [1, 1, 1], # 1    1
        [0, 0, 0], # 2    1
        [0, 1, 0], # 2    2
        [2, 1, 0], # 1    1.5
        [0, 1, 1], # 1    1
        [1, 0, 1], # 1    1
        [1, 1, 0], # 1    1
        [2, 0, 1]  # 0    0
    ]), dtype = np.double)
    
    #X_train = np.random.rand(14, 3)


    y_train = np.array([0, 0, 1, 1, 2, 0, 1, 2, 2, 1, 1, 1, 1, 0])
    X_test = X_train

    dataset = Dataset(X_train)
    labels  = Labels(y_train)
    testset = Dataset(X_test)

    # CLASSIFICATION
    config.task = CLASSIFICATION_TASK
    tree_addr = scythe.fit_classification_tree(ctypes.byref(dataset), ctypes.byref(labels), ctypes.byref(config))
    preds_addr = scythe.tree_classify(ctypes.byref(testset), ctypes.c_void_p(tree_addr), ctypes.byref(config))
    preds_p = ctypes.cast(preds_addr, c_float_p)
    preds = np.ctypeslib.as_array(preds_p, shape = (14, 3))
    print("\n")
    print(preds)

    # REGRESSION
    config.task = REGRESSION_TASK
    tree_addr = scythe.fit_regression_tree(ctypes.byref(dataset), ctypes.byref(labels), ctypes.byref(config))
    preds_addr = scythe.tree_predict(ctypes.byref(testset), ctypes.c_void_p(tree_addr), ctypes.byref(config))
    preds_p = ctypes.cast(preds_addr, c_double_p)
    preds = np.ctypeslib.as_array(preds_p, shape = (14,))
    print("\n")
    print(preds)

    print("Finished")