# -*- coding: utf-8 -*-
# distutils: language=c++

cdef extern from "../../src/scythe.hpp":
    struct Dataset:
        pass
    struct Labels:
        pass
    struct TreeConfig:
        pass
    void* fit_classification_tree(Dataset*, Labels*, TreeConfig*)
    void api_test()