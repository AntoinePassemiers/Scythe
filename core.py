# -*- coding: utf-8 -*-

import numpy as np
import ctypes

scythe = ctypes.cdll.LoadLibrary("scythe/scythe.lib")

class Dataset(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.POINTER(ctypes.c_double)),
        ("n_rows", ctypes.c_uint),
        ("n_cols", ctypes.c_uint)]

    def __init__(self, data):

        data = np.asarray(data, dtype = np.double)
        self.np_data = data.astype(np.double)

        self.n_rows = self.np_data.shape[0]
        self.n_cols = self.np_data.shape[1]

        self.c_double_p = ctypes.POINTER(ctypes.c_double)
        self.data = data.ctypes.data_as(self.c_double_p)


if __name__ == "__main__":
    dataset = Dataset(np.array([[5, 7], [0, 1], [9, 3]]))
    scythe.printArray(ctypes.byref(dataset))