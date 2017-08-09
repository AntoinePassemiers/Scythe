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
    size_t c_add_cascade_layer(size_t forest_id, LayerConfig lconfig)
    size_t c_add_scanner_1d(size_t forest_id, LayerConfig lconfig, size_t kc)
    size_t c_add_scanner_2d(size_t forest_id, LayerConfig lconfig, size_t kc, size_t kr)
    size_t c_add_scanner_3d(size_t forest_id, LayerConfig lconfig, size_t kc, size_t kr, size_t kd)
    void* c_get_forest(size_t deep_forest_id, size_t layer_id, size_t forest_id)