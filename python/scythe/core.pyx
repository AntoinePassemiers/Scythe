# -*- coding: utf-8 -*-
# distutils: language=c++
# distutils: sources = ../../src/scythe.cpp
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=True

import numpy as np
cimport numpy as cnp
cnp.import_array()

from libc.stdlib cimport *
from libc.stdio cimport *
from libc.string cimport *
from cython cimport view


target_np = np.double
data_np = np.double
ctypedef cnp.double_t cy_target_np
ctypedef cnp.double_t cy_data_np

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

CLASSIFICATION = "classification"
REGRESSION     = "regression"


cdef Dataset to_dataset(object X):
    cdef Dataset dataset
    cdef cnp.ndarray[cy_data_np, ndim = 2] np_data = np.ascontiguousarray(X, dtype = data_np)
    dataset.n_rows = np_data.shape[0]
    dataset.n_cols = np_data.shape[1]
    dataset.data = <data_t*>np_data.data
    return dataset

cdef Labels to_labels(object y):
    cdef Labels labels
    m, M = np.min(y), np.max(y)
    n_classes = M + 1
    cdef cnp.ndarray[cy_target_np, ndim = 1] np_data = np.ascontiguousarray(y, dtype = target_np)
    labels.n_rows = np_data.shape[0]
    labels.data = <target_t*>np_data.data
    return labels

cdef cnp.ndarray ptr_to_cls_predictions(float* predictions, size_t n_rows, size_t n_classes):
    cdef float[:, ::1] mview = <float[:n_rows, :n_classes:1]>predictions
    return np.asarray(mview)

cdef cnp.ndarray ptr_to_reg_predictions(cy_data_np* predictions, size_t n_rows):
    cdef cy_data_np[::1] mview = <cy_data_np[:n_rows:1]>predictions
    return np.asarray(mview)


cdef class TreeConfiguration:
    cdef TreeConfig config

    def __init__(self):
        self.config.task = CLASSIFICATION_TASK
        self.config.is_incremental = False
        self.config.min_threshold = 1e-06
        self.config.max_height = 200
        self.config.max_n_features = 9999
        self.config.n_classes = 2
        self.config.max_nodes = <size_t>1e+15
        self.config.partitioning = PERCENTILE_PARTITIONING
        self.config.nan_value = <data_t>np.nan
        self.config.is_complete_random = False

    cpdef TreeConfig get_c_config(self):
        return self.config

    property task:
        def __get__(self): return self.config.task
        def __set__(self, value): self.config.task = value
    property is_incremental:
        def __get__(self): return self.config.is_incremental
        def __set__(self, value): self.config.is_incremental = value
    property min_threshold:
        def __get__(self): return self.config.min_threshold
        def __set__(self, value): self.config.min_threshold = value
    property max_height:
        def __get__(self): return self.config.max_height
        def __set__(self, value): self.config.max_height = value
    property max_n_features:
        def __get__(self): return self.config.max_n_features
        def __set__(self, value): self.config.max_n_features = value
    property n_classes:
        def __get__(self): return self.config.n_classes
        def __set__(self, value): self.config.n_classes = value
    property max_nodes:
        def __get__(self): return self.config.max_nodes
        def __set__(self, value): self.config.max_nodes = value
    property partitioning:
        def __get__(self): return self.config.partitioning
        def __set__(self, value): self.config.partitioning = value
    property nan_value:
        def __get__(self): return self.config.nan_value
        def __set__(self, value): self.config.nan_value = value
    property is_complete_random:
        def __get__(self): return self.config.is_complete_random
        def __set__(self, value): self.config.is_complete_random = value


cdef class Tree:
    cdef char* task
    cdef TreeConfig config
    cdef void* predictor_p

    def __init__(self, cy_config, task):
        self.task = task
        if task == CLASSIFICATION:
            cy_config.task = CLASSIFICATION_TASK
        else:
            cy_config.task = REGRESSION_TASK
        self.config = cy_config.get_c_config()
    def fit(self, X, y):
        cdef Dataset dataset = to_dataset(X)
        cdef Labels labels = to_labels(y)
        if self.task == REGRESSION:
            self.predictor_p = fit_regression_tree(
                &dataset, &labels, &self.config)
        else:
            self.predictor_p = fit_classification_tree(
                &dataset, &labels, &self.config)
    def predict(self, X):
        cdef Dataset dataset = to_dataset(X)
        n_rows = len(X)
        if self.task == REGRESSION:
            preds = ptr_to_reg_predictions(
                tree_predict(&dataset, self.predictor_p, &self.config), n_rows)
        else:
            n_classes = self.config.n_classes
            preds = ptr_to_cls_predictions(
                tree_classify(&dataset, self.predictor_p, &self.config),
                n_rows, n_classes)
        return preds


def py_api_test():
    cdef Dataset* dataset_p = <Dataset*>malloc(sizeof(Dataset))
    dataset_p.n_rows = 42
    # cdef Labels* labels_p = <Labels*>malloc(sizeof(Labels))
    # cdef TreeConfig* config_p = <TreeConfig*>malloc(sizeof(TreeConfig))
    # fit_classification_tree(dataset_p, labels_p, config_p)
    api_test(dataset_p)
