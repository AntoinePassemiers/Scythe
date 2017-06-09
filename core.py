# -*- coding: utf-8 -*-
# core.py : Interface to the C API
# author : Antoine Passemiers

from structures import * # TODO
from utils import *


REGRESSION     = "regression"
CLASSIFICATION = "classification"


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
        Model.__init__(self, config, task)
    def fit(self, X, y):
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
    def __init__(self, config, task):
        Model.__init__(self, config, task)
