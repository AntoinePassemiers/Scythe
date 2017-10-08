# -*- coding: utf-8 -*-
# distutils: language=c++

import numpy as np
cimport numpy as cnp
cnp.import_array()

import abc

from scythe.core cimport *
from scythe.core import *
from scythe.utils import *


cdef class LayerConfiguration:
    cdef LayerConfig config

    def __init__(self, cy_config, n_forests, forest_type):
        self.config.fconfig = cy_config.get_c_config()
        self.config.n_forests = n_forests
        self.config.forest_type = forest_type

    cpdef LayerConfig get_c_config(self):
        return self.config

    property n_forests:
        def __get__(self): return self.config.n_forests
    property fconfig:
        def __get__(self): return self.config.fconfig

class Layer(object):
    def __init__(self, **kwargs):
        self.layer_id = None
        self.owner_id = None
    def getForests(self):
        assert(self.layer_id is not None)
        cdef void* forest_ptr
        forests = list()
        for i in range(self.lconfig.n_forests):
            
            forest_ptr = c_get_forest(self.owner_id, self.layer_id, i)
            forest_configuration = ForestConfiguration()
            forest_configuration.set_c_config(self.lconfig.fconfig)
            current_forest = Forest(forest_configuration, "classification", "rf")
            current_forest.set_predictor_p(<object>forest_ptr)
            forests.append(current_forest)
        return forests

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
        self.layer_id = c_add_cascade_layer(graph_id, self.lconfig.get_c_config())
        return self.layer_id


class MultiGrainedScanner1D(Layer):
    def __init__(self, lconfig, kernel_shape, **kwargs):
        Layer.__init__(self, **kwargs)
        assert(isinstance(lconfig, LayerConfiguration))
        assert(isinstance(kernel_shape, tuple))
        assert(len(kernel_shape) == 1)
        self.kernel_shape = kernel_shape
        self.lconfig = lconfig
    def addToGraph(self, graph_id):
        self.layer_id = c_add_scanner_1d(graph_id, self.lconfig.get_c_config(), self.kernel_shape[0])
        return self.layer_id


class MultiGrainedScanner2D(Layer):
    def __init__(self, lconfig, kernel_shape, **kwargs):
        Layer.__init__(self, **kwargs)
        assert(isinstance(lconfig, LayerConfiguration))
        assert(isinstance(kernel_shape, tuple))
        assert(len(kernel_shape) == 2)
        self.kernel_shape = kernel_shape
        self.lconfig = lconfig
    def addToGraph(self, graph_id):
        self.layer_id = c_add_scanner_2d(graph_id, self.lconfig.get_c_config(), self.kernel_shape[0], self.kernel_shape[1])
        return self.layer_id


class MultiGrainedScanner3D(Layer):
    def __init__(self, lconfig, kernel_shape, **kwargs):
        Layer.__init__(self, **kwargs)
        assert(isinstance(lconfig, LayerConfiguration))
        assert(isinstance(kernel_shape, tuple))
        assert(len(kernel_shape) == 3)
        self.kernel_shape = kernel_shape
        self.lconfig = lconfig
    def addToGraph(self, graph_id):
        self.layer_id = c_add_scanner_3d(graph_id, self.lconfig.get_c_config().fconfig, self.kernel_shape[0], self.kernel_shape[1], self.kernel_shape[2])
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
