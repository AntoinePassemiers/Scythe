# -*- coding: utf-8 -*-
# core.py : Interface to the C API
# author : Antoine Passemiers

import abc

from structures import *
from layers import *

REGRESSION     = "regression"
CLASSIFICATION = "classification"

RF_FOREST  = ["random forest", "rf"]
CRF_FOREST = ["complete random forest", "crf"]
GB_FOREST  = ["gradient boosting", "gb"]


class Model:
    def __init__(self, config, task):
        assert(task.lower() in [CLASSIFICATION, REGRESSION])
        self.config_p = config
        self.task = task
        self.predictor_p = None
        if self.task == CLASSIFICATION:
            config.task = CLASSIFICATION_TASK
        else:
            config.task = REGRESSION_TASK
    @abc.abstractmethod
    def fit(self, X, y):
        return None
    @abc.abstractmethod
    def predict(self, X):
        return None


class Tree(Model):
    def __init__(self, config, task):
        assert(isinstance(config, TreeConfig))
        Model.__init__(self, config, task)
    def fit(self, X, y):
        assert(isinstance(X, Dataset))
        assert(isinstance(y, Labels))
        if self.task == REGRESSION:
            self.predictor_p = scythe.fit_regression_tree(
                ctypes.byref(X), 
                ctypes.byref(y), 
                ctypes.byref(self.config_p))
        else:
            self.predictor_p = scythe.fit_classification_tree(
                ctypes.byref(X), 
                ctypes.byref(y), 
                ctypes.byref(self.config_p))
    def predict(self, X):
        assert(isinstance(X, Dataset))
        n_rows = len(X)
        if self.task == REGRESSION:
            preds_addr = scythe.tree_predict(
                ctypes.byref(X),
                ctypes.c_void_p(self.predictor_p),
                ctypes.byref(self.config_p))
            preds_p = ctypes.cast(preds_addr, c_double_p)
            preds = np.ctypeslib.as_array(preds_p, shape = (n_rows,))
        else:
            n_classes = self.config_p.n_classes
            preds_addr = scythe.tree_classify(
                ctypes.byref(X),
                ctypes.c_void_p(self.predictor_p),
                ctypes.byref(self.config_p))
            preds_p = ctypes.cast(preds_addr, c_float_p)
            preds = np.ctypeslib.as_array(preds_p, shape = (n_rows, n_classes))
        return preds


class Forest(Model):
    def __init__(self, config, task, forest_type):
        forest_type = forest_type.lower()
        assert(isinstance(config, ForestConfig))
        assert(forest_type in RF_FOREST + CRF_FOREST + GB_FOREST)
        Model.__init__(self, config, task)
        if forest_type in RF_FOREST:
            config.type = RANDOM_FOREST
        elif forest_type in CRF_FOREST:
            config.type = COMPLETE_RANDOM_FOREST
        else:
            config.type = GRADIENT_BOOSTING
    def fit(self, X, y):
        assert(isinstance(X, Dataset))
        assert(isinstance(y, Labels))
        if self.task == REGRESSION:
            raise NotImplementedError()
        else:
            self.predictor_p = scythe.fit_classification_forest(
                ctypes.byref(X), 
                ctypes.byref(y), 
                ctypes.byref(self.config_p))
    def predict(self, X):
        assert(isinstance(X, Dataset))
        n_classes = self.config_p.n_classes
        n_rows = len(X)
        if self.task == REGRESSION:
            raise NotImplementedError()
        else:
            n_classes = self.config_p.n_classes
            preds_addr = scythe.forest_classify(
                ctypes.byref(X),
                ctypes.c_void_p(self.predictor_p),
                ctypes.byref(self.config_p))
            preds_p = ctypes.cast(preds_addr, c_float_p)
            preds = np.ctypeslib.as_array(preds_p, shape = (n_rows, n_classes))
        return preds


class DeepForest:
    def __init__(self, task = "classification"):
        assert(task in [REGRESSION, CLASSIFICATION])
        self.task = task
        d = CLASSIFICATION_TASK if (self.task == CLASSIFICATION) else REGRESSION_TASK
        self.deep_forest_p = scythe.c_create_deep_forest(d)
    def add(self, layer):
        assert(isinstance(layer, Layer))
        layer.addToGraph(self.deep_forest_p)