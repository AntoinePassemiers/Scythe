# -*- coding: utf-8 -*-
# structures.py : Function signatures from the C-API
# author : Antoine Passemiers

from scythe.structures import *


scythe.c_create_deep_forest.restype = ctypes.c_size_t
scythe.c_create_deep_forest.argtypes = [ctypes.c_int]

scythe.c_fit_deep_forest.restype = None
scythe.c_fit_deep_forest.argtypes = [
    MDDataset,
    ctypes.POINTER(Labels),
    ctypes.c_size_t]

scythe.c_add_scanner_1d.restype = None
scythe.c_add_scanner_1d.argtypes = [
    ctypes.c_size_t,
    LayerConfig,
    ctypes.c_size_t]

scythe.c_add_scanner_2d.restype = None
scythe.c_add_scanner_2d.argtypes = [
    ctypes.c_size_t,
    LayerConfig,
    ctypes.c_size_t,
    ctypes.c_size_t]

scythe.c_add_scanner_3d.restype = None
scythe.c_add_scanner_3d.argtypes = [
    ctypes.c_size_t,
    LayerConfig,
    ctypes.c_size_t,
    ctypes.c_size_t,
    ctypes.c_size_t]