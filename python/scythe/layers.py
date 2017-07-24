# -*- coding: utf-8 -*-
# core.py : Scythe's layer classes
# author : Antoine Passemiers

import abc

from structures import *


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
        assert(isinstance(lconfig, LayerConfig))
        self.lconfig = lconfig
    def addToGraph(self, graph_id):
        raise NotImplementedError()


class CascadeLayer(Layer):
    def __init__(self, lconfig, **kwargs):
        Layer.__init__(self, **kwargs)
        assert(isinstance(lconfig, LayerConfig))
        self.lconfig = lconfig
    def addToGraph(self, graph_id):
        scythe.c_add_cascade_layer(
            ctypes.c_size_t(int(graph_id)),
            self.lconfig)


class MultiGrainedScanner1D(Layer):
    def __init__(self, lconfig, kernel_shape, **kwargs):
        Layer.__init__(self, **kwargs)
        assert(isinstance(lconfig, LayerConfig))
        assert(isinstance(kernel_shape, tuple))
        assert(len(kernel_shape) == 1)
        self.kernel_shape = kernel_shape
        self.lconfig = lconfig
    def addToGraph(self, graph_id):
        scythe.c_add_scanner_1d(
            ctypes.c_size_t(int(graph_id)),
            self.lconfig,
            self.kernel_shape[0])


class MultiGrainedScanner2D(Layer):
    def __init__(self, lconfig, kernel_shape, **kwargs):
        Layer.__init__(self, **kwargs)
        assert(isinstance(lconfig, LayerConfig))
        assert(isinstance(kernel_shape, tuple))
        assert(len(kernel_shape) == 2)
        self.kernel_shape = kernel_shape
        self.lconfig = lconfig
    def addToGraph(self, graph_id):
        scythe.c_add_scanner_2d(
            ctypes.c_size_t(int(graph_id)),
            self.lconfig,
            self.kernel_shape[0],
            self.kernel_shape[1])
    @staticmethod
    def estimateRequiredBufferSize(N, M, P, kc, kr, n_classes, n_forests):
        sc = M - kc + 1
        sr = P - kr + 1
        return N * sc * sr * n_forests * n_classes


class MultiGrainedScanner3D(Layer):
    def __init__(self, lconfig, kernel_shape, **kwargs):
        Layer.__init__(self, **kwargs)
        assert(isinstance(lconfig, LayerConfig))
        assert(isinstance(kernel_shape, tuple))
        assert(len(kernel_shape) == 3)
        self.kernel_shape = kernel_shape
        self.lconfig = lconfig
    def addToGraph(self, graph_id):
        scythe.c_add_scanner_3d(
            ctypes.c_size_t(int(graph_id)),
            self.lconfig,
            self.kernel_shape[0],
            self.kernel_shape[1],
            self.kernel_shape[2])