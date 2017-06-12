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
    def addToGraph(self, graph):
        return None

class DirectLayer(Layer):
    def __init__(self, **kwargs):
        Layer.__init__(self, **kwargs)
        # TODO
    def addToGraph(self, graph):
        raise NotImplementedError()

class CascadeLayer(Layer):
    def __init__(self, **kwargs):
        Layer.__init__(self, **kwargs)
        # TODO
    def addToGraph(self, graph):
        raise NotImplementedError()

class MultiGrainedScanner1D(Layer):
    def __init__(self, **kwargs):
        Layer.__init__(self, **kwargs)
        # TODO
    def addToGraph(self, graph):
        scythe.c_add_scanner_1d(
            ctypes.c_void_p(graph),
            None, # TODO
            2 )   # TODO

class MultiGrainedScanner2D(Layer):
    def __init__(self, **kwargs):
        Layer.__init__(self, **kwargs)
        # TODO
    def addToGraph(self, graph):
        raise NotImplementedError()

class MultiGrainedScanner3D(Layer):
    def __init__(self, **kwargs):
        Layer.__init__(self, **kwargs)
        # TODO
    def addToGraph(self, graph):
        raise NotImplementedError()