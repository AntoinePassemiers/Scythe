# -*- coding: utf-8 -*-

import numpy as np
import ctypes

scythe = ctypes.cdll.LoadLibrary("scythe/scythe.lib")

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

        self.c_double_p = ctypes.POINTER(ctypes.c_double)
        self.data = data.ctypes.data_as(self.c_double_p)

class Labels(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.POINTER(ctypes.c_int)),
        ("n_rows", ctypes.c_size_t)
    ]

    def __init__(self, data):
        data = np.asarray(data, dtype = np.int)
        self.np_data = data.astype(np.int)
        self.n_rows = self.np_data.shape[0]

        self.c_int_p = ctypes.POINTER(ctypes.c_int)
        self.data = data.ctypes.data_as(self.c_int_p)


if __name__ == "__main__":
    X = np.random.rand(1000, 8)
    y = np.random.randint(0, 3, size = 1000)

    dataset = Dataset(X)
    labels  = Labels(y)
    scythe.fit(ctypes.byref(dataset), ctypes.byref(labels))