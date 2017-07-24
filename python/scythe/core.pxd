# -*- coding: utf-8 -*-
# distutils: language=c++


cdef extern from "../../src/tree/cart.hpp" namespace "scythe":
    cdef struct TreeConfig:
        pass

cdef extern from "../../src/misc/sets.hpp" namespace "scythe":
    cdef struct Dataset:
        pass
    cdef struct Labels:
        pass

cdef extern from "../../src/scythe.cpp":
    # void* fit_classification_tree(Dataset*, Labels*, TreeConfig*)
    void api_test(Dataset*)