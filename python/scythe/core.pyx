# -*- coding: utf-8 -*-
# distutils: language=c++
# distutils: sources = ../../src/scythe.cpp
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=True

from libc.stdlib cimport *


def py_api_test():
    cdef Dataset* dataset_p = <Dataset*>malloc(sizeof(Dataset))
    # cdef Labels* labels_p = <Labels*>malloc(sizeof(Labels))
    # cdef TreeConfig* config_p = <TreeConfig*>malloc(sizeof(TreeConfig))
    # fit_classification_tree(dataset_p, labels_p, config_p)
    api_test(dataset_p)