# -*- coding: utf-8 -*-
# distutils: language=c++

import numpy as np
cimport numpy as cnp
cnp.import_array()

from scythe.core cimport *
from scythe.core import *


cdef extern from "../../src/deep_learning/layers/layer.hpp" namespace "scythe":
    struct LayerConfig:
        ForestConfig fconfig
        size_t       n_forests
        int          forest_type

cdef extern from "../../src/deep_scythe.hpp":
    size_t c_create_deep_forest(int task)
    void c_fit_deep_forest(MDDataset dataset, Labels* labels, size_t forest_id)
    float* c_deep_forest_classify(MDDataset dataset, size_t forest_id)
    void c_add_cascade_layer(size_t forest_id, LayerConfig lconfig)
    void c_add_scanner_1d(size_t forest_id, LayerConfig lconfig, size_t kc)
    void c_add_scanner_2d(size_t forest_id, LayerConfig lconfig, size_t kc, size_t kr)
    void c_add_scanner_3d(size_t forest_id, LayerConfig lconfig, size_t kc, size_t kr, size_t kd)
