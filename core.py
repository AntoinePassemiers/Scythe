# -*- coding: utf-8 -*-

import numpy as np
import ctypes

scythe = ctypes.cdll.LoadLibrary("scythe/scythe.lib")


c_float_p = ctypes.POINTER(ctypes.c_float)

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

class TreeConfig(ctypes.Structure):
    _fields_ = [
        ("is_incremental", ctypes.c_int),
        ("min_threshold", ctypes.c_double),
        ("max_height", ctypes.c_size_t),
        ("n_classes", ctypes.c_size_t),
        ("max_nodes", ctypes.c_size_t),
        ("partitioning", ctypes.c_int),
        ("nan_value", ctypes.c_double)
    ]
    def __init__(self):
        pass # TODO


if __name__ == "__main__":
    X_train = np.asarray(np.array([
        [0, 0, 0], # 0
        [0, 0, 1], # 0
        [1, 0, 0], # 1
        [2, 0, 0], # 1
        [2, 1, 0], # 2
        [2, 1, 1], # 0
        [1, 1, 1], # 1
        [0, 0, 0], # 2
        [0, 1, 0], # 2
        [2, 1, 0], # 1
        [0, 1, 1], # 1
        [1, 0, 1], # 1
        [1, 1, 0], # 1
        [2, 0, 1]  # 0
    ]), dtype = np.double)
    y_train = np.array([0, 0, 1, 1, 2, 0, 1, 2, 2, 1, 1, 1, 1, 0])
    X_test = X_train

    dataset = Dataset(X_train)
    labels  = Labels(y_train)
    testset = Dataset(X_test)
    tree_addr = scythe.fit(ctypes.byref(dataset), ctypes.byref(labels))
    preds_addr = scythe.predict(ctypes.byref(testset), ctypes.c_void_p(tree_addr))
    preds_p = ctypes.cast(preds_addr, c_float_p)
    preds = np.ctypeslib.as_array(preds_p, shape = (14, 3))
    print(preds)

    print("Finished")