# -*- coding: utf-8 -*-
# distutils: language=c++

import numpy as np
cimport numpy as cnp
cnp.import_array()

import abc

from scythe.utils import *


cdef class LayerConfiguration:
    cdef LayerConfig config

    def __init__(self, cy_config, n_forests, forest_type):
        self.config.fconfig = cy_config.get_c_config()
        self.config.n_forests = n_forests
        self.config.forest_type = forest_type

    cpdef LayerConfig get_c_config(self):
        return self.config

class Layer(object):
    def __init__(self, **kwargs):
        self.config = None
        # TODO
    @abc.abstractmethod
    def addToGraph(self, graph_id):
        return None


class DirectLayer(Layer):
    def __init__(self, lconfig, **kwargs):
        Layer.__init__(self, **kwargs)
        assert(isinstance(lconfig, LayerConfiguration))
        self.lconfig = lconfig
    def addToGraph(self, graph_id):
        raise NotImplementedError()


class CascadeLayer(Layer):
    def __init__(self, lconfig, **kwargs):
        Layer.__init__(self, **kwargs)
        assert(isinstance(lconfig, LayerConfiguration))
        self.lconfig = lconfig
    def addToGraph(self, graph_id):
        c_add_cascade_layer(graph_id, self.lconfig.get_c_config())


class MultiGrainedScanner1D(Layer):
    def __init__(self, lconfig, kernel_shape, **kwargs):
        Layer.__init__(self, **kwargs)
        assert(isinstance(lconfig, LayerConfiguration))
        assert(isinstance(kernel_shape, tuple))
        assert(len(kernel_shape) == 1)
        self.kernel_shape = kernel_shape
        self.lconfig = lconfig
    def addToGraph(self, graph_id):
        c_add_scanner_1d(graph_id, self.lconfig.get_c_config(), self.kernel_shape[0])


class MultiGrainedScanner2D(Layer):
    def __init__(self, lconfig, kernel_shape, **kwargs):
        Layer.__init__(self, **kwargs)
        assert(isinstance(lconfig, LayerConfiguration))
        assert(isinstance(kernel_shape, tuple))
        assert(len(kernel_shape) == 2)
        self.kernel_shape = kernel_shape
        self.lconfig = lconfig
    def addToGraph(self, graph_id):
        c_add_scanner_2d(graph_id, self.lconfig.get_c_config(), self.kernel_shape[0], self.kernel_shape[1])
    @staticmethod
    def estimateRequiredBufferSize(N, M, P, kc, kr, n_classes, n_forests):
        sc = M - kc + 1
        sr = P - kr + 1
        return N * sc * sr * n_forests * n_classes


class MultiGrainedScanner3D(Layer):
    def __init__(self, lconfig, kernel_shape, **kwargs):
        Layer.__init__(self, **kwargs)
        assert(isinstance(lconfig, LayerConfiguration))
        assert(isinstance(kernel_shape, tuple))
        assert(len(kernel_shape) == 3)
        self.kernel_shape = kernel_shape
        self.lconfig = lconfig
    def addToGraph(self, graph_id):
        c_add_scanner_3d(graph_id, self.lconfig.get_c_config(), self.kernel_shape[0], self.kernel_shape[1], self.kernel_shape[2])


class DeepForest:
    def __init__(self, task = "classification"):
        assert(task in [REGRESSION, CLASSIFICATION])
        self.n_classes = 0
        self.task = task
        d = CLASSIFICATION_TASK if (self.task == CLASSIFICATION) else REGRESSION_TASK
        self.deep_forest_id = c_create_deep_forest(d)
    def add(self, layer):
        layer.addToGraph(self.deep_forest_id)
        self.n_classes = layer.get_c_config().fconfig.n_classes
    def fit(self, X, y):
        cdef cnp.ndarray cX = np.ascontiguousarray(X, dtype = data_np)
        cdef cnp.ndarray cy = np.ascontiguousarray(y, dtype = target_np)
        cdef MDDataset dataset = to_md_dataset(cX)
        cdef Labels labels = to_labels(cy)
        c_fit_deep_forest(dataset, &labels, self.deep_forest_id)
    def classify(self, X):
        cdef cnp.ndarray cX = np.ascontiguousarray(X, dtype = data_np)
        cdef MDDataset dataset = to_md_dataset(cX)
        n_instances, n_classes = len(X), self.n_classes
        preds = ptr_to_cls_predictions(
	        c_deep_forest_classify(dataset, self.deep_forest_id),
	        n_instances, n_classes)
        return preds
