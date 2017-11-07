# -*- coding: utf-8 -*-
# distutils: language=c++
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

from libcpp.limits cimport numeric_limits

from scythe.utils import *


cdef Dataset to_dataset(cnp.ndarray X):
    cdef Dataset dataset
    dataset.n_rows = X.shape[0]
    dataset.n_cols = X.shape[1]
    dataset.data = <void*>X.data
    dataset.dtype = np.dtype(X.dtype).num
    return dataset

cdef MDDataset to_md_dataset(cnp.ndarray X):
    cdef MDDataset dataset
    dataset.dtype = np.dtype(X.dtype).num
    dataset.data = <void*>X.data
    dataset.n_dims = X.ndim
    for i in range(7):
        dataset.dims[i] = X.shape[i] if i < dataset.n_dims else 0
    return dataset

cdef Labels to_labels(cnp.ndarray y):
    cdef Labels labels
    labels.n_rows = y.shape[0]
    labels.data = <target_t*>y.data
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
        self.config.ordered_queue = False
        self.config.class_weights = NULL

    cdef TreeConfig get_c_config(self):
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
    property ordered_queue:
        def __get__(self): return self.config.ordered_queue
        def __set__(self, value): self.config.ordered_queue = value
    property class_weights:
        def __get__(self):
            return np.asarray(<float[:self.config.n_classes:1]>self.config.class_weights)
        def __set__(self, cnp.ndarray value):
            assert(value.dtype == np.float)
            self.config.n_classes = len(value)
            self.config.class_weights = <float*>value.data


cdef class ForestConfiguration:
    cdef ForestConfig config

    def __init__(self):
        self.config.type = RANDOM_FOREST
        self.config.task = CLASSIFICATION_TASK
        self.config.n_classes = 2
        self.config.score_metric = MLOG_LOSS
        self.config.n_iter = 100
        self.config.max_n_trees = 100
        self.config.max_n_nodes = <size_t>1e+15
        self.config.max_n_features = 9999
        self.config.n_leaves = <size_t>1e+15
        self.config.n_jobs = 1
        self.config.n_samples_per_leaf = 1
        self.config.regularization = REG_L1
        self.config.bagging_fraction = 1.0
        self.config.early_stopping_round = 300
        self.config.boosting_method = GRADIENT_BOOSTING
        self.config.max_depth = <size_t>1e+15
        self.config.l1_lambda = 0.1
        self.config.l2_lambda = 0.1
        self.config.seed = 4.0
        self.config.verbose = True
        self.config.nan_value = numeric_limits[data_t].quiet_NaN()
        self.config.min_threshold = 1e-06
        self.config.ordered_queue = False
        self.config.partitioning = 100

    cdef ForestConfig get_c_config(self):
        return self.config
    cdef void set_c_config(self, ForestConfig fconfig):
        self.config = fconfig

    property type:
        def __get__(self): return self.config.type
        def __set__(self, value): self.config.type = value
    property task:
        def __get__(self): return self.config.task
        def __set__(self, value): self.config.task = value
    property n_classes:
        def __get__(self): return self.config.n_classes
        def __set__(self, value): self.config.n_classes = value
    property score_metric:
        def __get__(self): return self.config.score_metric
        def __set__(self, value): self.config.score_metric = value
    property max_n_trees:
        def __get__(self): return self.config.max_n_trees
        def __set__(self, value): self.config.max_n_trees = self.config.n_iter = value
    property max_n_nodes:
        def __get__(self): return self.config.max_n_nodes
        def __set__(self, value): self.config.max_n_nodes = value
    property max_n_features:
        def __get__(self): return self.config.max_n_features
        def __set__(self, value): self.config.max_n_features = value
    property n_leaves:
        def __get__(self): return self.config.n_leaves
        def __set__(self, value): self.config.n_leaves = value
    property n_jobs:
        def __get__(self): return self.config.n_jobs
        def __set__(self, value): self.config.n_jobs = value
    property n_samples_per_leaf:
        def __get__(self): return self.config.n_samples_per_leaf
        def __set__(self, value): self.config.n_samples_per_leaf = value
    property regularization:
        def __get__(self): return self.config.regularization
        def __set__(self, value): self.config.regularization = value
    property bagging_fraction:
        def __get__(self): return self.config.bagging_fraction
        def __set__(self, value): self.config.bagging_fraction = value
    property early_stopping_round:
        def __get__(self): return self.config.early_stopping_round
        def __set__(self, value): self.config.early_stopping_round = value
    property boosting_method:
        def __get__(self): return self.config.boosting_method
        def __set__(self, value): self.config.boosting_method = value
    property max_depth:
        def __get__(self): return self.config.max_depth
        def __set__(self, value): self.config.max_depth = value
    property l1_lambda:
        def __get__(self): return self.config.l1_lambda
        def __set__(self, value): self.config.l1_lambda = value
    property l2_lambda:
        def __get__(self): return self.config.l2_lambda
        def __set__(self, value): self.config.l2_lambda = value
    property seed:
        def __get__(self): return self.config.seed
        def __set__(self, value): self.config.seed = value
    property verbose:
        def __get__(self): return self.config.verbose
        def __set__(self, value): self.config.verbose = value
    property nan_value:
        def __get__(self): return self.config.nan_value
        def __set__(self, value): self.config.nan_value = value
    property min_threshold:
        def __get__(self): return self.config.min_threshold
        def __set__(self, value): self.config.min_threshold = value
    property ordered_queue:
        def __get__(self): return self.config.ordered_queue
        def __set__(self, value): self.config.ordered_queue = value
    property partitioning:
        def __get__(self): return self.config.partitioning
        def __set__(self, value): self.config.partitioning = value


cdef class Tree:
    cdef TreeConfig config
    cdef void* predictor_p
    cdef size_t n_features

    def __init__(self, TreeConfiguration cy_config, task):
        cy_config.task = TASKS[task]
        self.config = cy_config.get_c_config()
    def fit(self, X, y):
        cdef cnp.ndarray cX = np.ascontiguousarray(X, dtype = data_np)
        cdef cnp.ndarray cy = np.ascontiguousarray(y, dtype = target_np)
        cdef Dataset dataset = to_dataset(cX)
        cdef Labels labels = to_labels(cy)
        if self.config.task == REGRESSION_TASK:
            self.predictor_p = fit_regression_tree(
                &dataset, &labels, &self.config)
        else:
            self.predictor_p = fit_classification_tree(
                &dataset, &labels, &self.config)
        self.n_features = cX.shape[1]
    def predict(self, X):
        cdef cnp.ndarray cX = np.ascontiguousarray(X, dtype = data_np)
        cdef Dataset dataset = to_dataset(cX)
        n_rows = len(X)
        assert(self.n_features == cX.shape[1])
        if self.config.task == REGRESSION_TASK:
            preds = ptr_to_reg_predictions(
                tree_predict(&dataset, self.predictor_p, &self.config), n_rows)
        else:
            n_classes = self.config.n_classes
            preds = ptr_to_cls_predictions(
                tree_classify(&dataset, self.predictor_p, &self.config),
                n_rows, n_classes)
        return preds
    def getFeatureImportances(self):
        importances = tree_get_feature_importances(self.predictor_p)
        self.n_features = importances.length
        return np.asarray(<double[:self.n_features:1]>importances.data)



cdef class Forest:
    cdef ForestConfig config
    cdef void* predictor_p
    cdef size_t n_features

    def __init__(self, ForestConfiguration cy_config, task, forest_type):
        cy_config.task = TASKS[task]
        cy_config.type = FOREST_TYPES[forest_type.lower()]
        self.config = cy_config.get_c_config()
    def set_predictor_p(self, ptr):
        self.predictor_p = <void*>ptr
    def fit(self, X, y):
        cdef cnp.ndarray cX = np.ascontiguousarray(X, dtype = data_np)
        cdef cnp.ndarray cy = np.ascontiguousarray(y, dtype = target_np)
        cdef Dataset dataset = to_dataset(cX)
        cdef Labels labels = to_labels(cy)
        if self.config.task == REGRESSION_TASK:
            raise NotImplementedError()
        else:
            self.predictor_p = fit_classification_forest(
                &dataset, &labels, &self.config)
        self.n_features = cX.shape[1]
    def predict(self, X):
        cdef cnp.ndarray cX = np.ascontiguousarray(X, dtype = data_np)
        cdef Dataset dataset = to_dataset(cX)
        n_classes = self.config.n_classes
        n_rows = len(X)
        assert(self.n_features == cX.shape[1])
        if self.config.task == REGRESSION_TASK:
            raise NotImplementedError()
        else:
            n_classes = self.config.n_classes
            preds = ptr_to_cls_predictions(
                forest_classify(&dataset, self.predictor_p, &self.config),
                n_rows, n_classes)
        return preds
    def getFeatureImportances(self):
        importances = forest_get_feature_importances(self.predictor_p)
        self.n_features = importances.length
        return np.asarray(<double[:self.n_features:1]>importances.data)

cdef class Scythe:
    cdef void* scythe_p

    def __init__(self):
        self.scythe_p = create_scythe()

    def prune_forest_height(self, Forest forest, max_height):
        return forest_prune_height(self.scythe_p, forest.predictor_p, max_height)

    def restore(self, int pruning_id):
        restore_pruning(self.scythe_p, pruning_id)
    
    def prune(self, int pruning_id):
        prune(self.scythe_p, pruning_id)

def py_api_test():
    cdef Dataset* dataset_p = <Dataset*>malloc(sizeof(Dataset))
    dataset_p.n_rows = 42
    # cdef Labels* labels_p = <Labels*>malloc(sizeof(Labels))
    # cdef TreeConfig* config_p = <TreeConfig*>malloc(sizeof(TreeConfig))
    # fit_classification_tree(dataset_p, labels_p, config_p)
    api_test(dataset_p)


cdef class LayerConfiguration:
    cdef LayerConfig config

    def __init__(self, ForestConfiguration cy_config, n_forests, forest_type):
        self.config.fconfig = cy_config.get_c_config()
        self.config.n_forests = n_forests
        self.config.forest_type = forest_type

    cdef LayerConfig get_c_config(self):
        return self.config

    property n_forests:
        def __get__(self): return self.config.n_forests
    # property fconfig:
    #     def __get__(self): return self.config.fconfig


NO_ID = -1

cdef class Layer(object):
    cdef int layer_id
    cdef int owner_id

    def __init__(self, **kwargs):
        self.layer_id = NO_ID
        self.owner_id = NO_ID
    def getForests(self):
        assert(self.layer_id is not None)
        cdef void* forest_ptr
        forests = list()
        for i in range(self.lconfig.n_forests):
            
            forest_ptr = c_get_forest(self.owner_id, self.layer_id, i)
            forest_configuration = ForestConfiguration()
            # forest_configuration.set_c_config(self.lconfig.fconfig) # TODO
            current_forest = Forest(forest_configuration, "classification", "rf")
            current_forest.set_predictor_p(<object>forest_ptr)
            forests.append(current_forest)
        return forests

    def addToGraph(self, graph_id):
        raise NotImplementedError()

    property layer_id:
        def __get__(self): return self.layer_id
        def __set__(self, value): self.layer_id = value
    property owner_id:
        def __get__(self): return self.owner_id
        def __set__(self, value): self.owner_id = value


cdef class DirectLayer(Layer):
    cdef LayerConfiguration lconfig

    def __init__(self, LayerConfiguration lconfig, **kwargs):
        Layer.__init__(self, **kwargs)
        assert(isinstance(lconfig, LayerConfiguration))
        self.lconfig = lconfig
    def addToGraph(self, graph_id):
        raise NotImplementedError()


cdef class CascadeLayer(Layer):
    cdef LayerConfiguration lconfig

    def __init__(self, LayerConfiguration lconfig, **kwargs):
        Layer.__init__(self, **kwargs)
        assert(isinstance(lconfig, LayerConfiguration))
        self.lconfig = lconfig
    def addToGraph(self, graph_id):
        self.layer_id = c_add_cascade_layer(graph_id, self.lconfig.get_c_config())
        return self.layer_id


cdef class MultiGrainedScanner2D(Layer):
    cdef LayerConfiguration lconfig
    cdef object kernel_shape

    def __init__(self, LayerConfiguration lconfig, kernel_shape, **kwargs):
        Layer.__init__(self, **kwargs)
        assert(isinstance(lconfig, LayerConfiguration))
        assert(isinstance(kernel_shape, tuple))
        assert(len(kernel_shape) == 2)
        self.kernel_shape = kernel_shape
        self.lconfig = lconfig
    def addToGraph(self, graph_id):
        self.layer_id = c_add_scanner_2d(graph_id, self.lconfig.get_c_config(), self.kernel_shape[0], self.kernel_shape[1])
        return self.layer_id


class DeepForest:
    def __init__(self, n_classes = 2, task = "classification"):
        self.n_classes = n_classes
        self.task = task
        self.deep_forest_id = c_create_deep_forest(TASKS[task])
        self.layers = list()
    def add(self, layer):
        layer_id = layer.addToGraph(self.deep_forest_id)
        layer.owner_id = self.deep_forest_id
        self.layers.append(layer)
        return layer_id
    def connect(self, parent_id, child_id):
        c_connect_nodes(self.deep_forest_id, parent_id, child_id)
    def fit(self, X, y):
        cdef cnp.ndarray cX = np.ascontiguousarray(X)
        cdef cnp.ndarray cy = np.ascontiguousarray(y, dtype = target_np)
        cdef MDDataset dataset = to_md_dataset(cX)
        cdef Labels labels = to_labels(cy)
        c_fit_deep_forest(dataset, &labels, self.deep_forest_id)

    def classify(self, X):
        cdef cnp.ndarray cX = np.ascontiguousarray(X)
        cdef MDDataset dataset = to_md_dataset(cX)
        n_instances, n_classes = len(X), self.n_classes
        preds = ptr_to_cls_predictions(
            c_deep_forest_classify(dataset, self.deep_forest_id),
            n_instances, n_classes)
        return preds
