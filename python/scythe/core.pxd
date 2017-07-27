# -*- coding: utf-8 -*-
# distutils: language=c++

from libcpp cimport bool
from libcpp.limits cimport numeric_limits

import numpy as np
cimport numpy as cnp
cnp.import_array()


ctypedef double data_t

target_np = np.double
data_np = np.double
ctypedef cnp.double_t cy_target_np
ctypedef cnp.double_t cy_data_np

# Magic numbers
cpdef int CLASSIFICATION_TASK = 0xF55A90
cpdef int REGRESSION_TASK     = 0xF55A91

cpdef int MLOG_LOSS = 0x7711A0
cpdef int MSE       = 0xC97B00

cpdef int QUARTILE_PARTITIONING   = 0xB23A40
cpdef int DECILE_PARTITIONING     = 0xB23A41
cpdef int PERCENTILE_PARTITIONING = 0xB23A42

cpdef int RANDOM_FOREST          = 0
cpdef int COMPLETE_RANDOM_FOREST = 1
cpdef int GRADIENT_BOOSTING      = 2

cpdef int REG_L1 = 0x778C10
cpdef int REG_L2 = 0x778C11

cpdef int ADABOOST          = 0x28FE90
cpdef int GRADIENT_BOOSTING = 0x28FE91

cpdef int DTYPE_PROBA  = 0
cpdef int DTYPE_DATA   = 1
cpdef int DTYPE_UINT_8 = 2

cpdef object CLASSIFICATION = "classification"
cpdef object REGRESSION     = "regression"
cpdef object RF_FOREST  = ["random forest", "rf"]
cpdef object CRF_FOREST = ["complete random forest", "crf"]
cpdef object GB_FOREST  = ["gradient boosting", "gb"]

cdef extern from "../../src/misc/sets.hpp" namespace "scythe":
    ctypedef unsigned int uint
    ctypedef double       data_t
    ctypedef float        fast_data_t
    ctypedef double       target_t
    ctypedef float        proba_t
    ctypedef int          label_t

    struct Dataset:
        void* data
        size_t n_rows
        size_t n_cols

    struct MDDataset:
        void* data
        size_t n_dims
        size_t dims[7]
        int dtype

    struct Labels:
        target_t* data
        size_t n_rows

cdef extern from "../../src/tree/cart.hpp" namespace "scythe":
    struct TreeConfig:
        int    task
        bool   is_incremental
        double min_threshold
        size_t max_height
        size_t max_n_features
        size_t n_classes
        size_t max_nodes
        int    partitioning
        data_t nan_value
        bool   is_complete_random

cdef extern from "../../src/forest/forest.hpp" namespace "scythe":
    struct ForestConfig:
        int       type                 
        int       task                 
        size_t    n_classes            
        int       score_metric         
        size_t    n_iter               
        size_t    max_n_trees          
        size_t    max_n_nodes          
        size_t    max_n_features       
        float     learning_rate        
        size_t    n_leaves             
        size_t    n_jobs               
        size_t    n_samples_per_leaf   
        int       regularization       
        size_t    bag_size             
        size_t    early_stopping_round 
        int       boosting_method      
        int       max_depth            
        float     l1_lambda            
        float     l2_lambda            
        float     seed                 
        int       verbose              
        data_t    nan_value            
        double    min_threshold        

cdef extern from "../../src/scythe.hpp":
    void* fit_classification_tree(Dataset*, Labels*, TreeConfig*)
    void* fit_regression_tree(Dataset*, Labels*, TreeConfig*)
    float* tree_classify(Dataset*, void*, TreeConfig*)
    data_t* tree_predict(Dataset*, void*, TreeConfig*)
    void* fit_classification_forest(Dataset*, Labels*, ForestConfig*)
    float* forest_classify(Dataset* dataset, void* forest_p, ForestConfig* config)
    void api_test(Dataset*)


cdef Dataset to_dataset(object X)
cdef MDDataset to_md_dataset(object X)
cdef Labels to_labels(object y)
cdef cnp.ndarray ptr_to_cls_predictions(float* predictions, size_t n_rows, size_t n_classes)
cdef cnp.ndarray ptr_to_reg_predictions(data_t* predictions, size_t n_rows)
