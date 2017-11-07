# -*- coding: utf-8 -*-
# distutils: language=c++

from libcpp cimport bool
from libcpp.limits cimport numeric_limits

import numpy as np
cimport numpy as cnp
cnp.import_array()

ctypedef double data_t

ctypedef cnp.double_t cy_target_np
ctypedef cnp.double_t cy_data_np

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
        int dtype

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
        bool   ordered_queue

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
        float     bagging_fraction             
        size_t    early_stopping_round 
        int       boosting_method      
        int       max_depth            
        float     l1_lambda
        float     l2_lambda            
        float     seed                 
        int       verbose              
        data_t    nan_value            
        double    min_threshold
        bool      ordered_queue
        int       partitioning

cdef extern from "../../src/scythe.hpp":
    struct double_vec_t:
        double* data
        size_t length

    void* fit_classification_tree(Dataset*, Labels*, TreeConfig*)
    void* fit_regression_tree(Dataset*, Labels*, TreeConfig*)
    float* tree_classify(Dataset*, void*, TreeConfig*)
    data_t* tree_predict(Dataset*, void*, TreeConfig*)
    double_vec_t tree_get_feature_importances(void*)

    void* fit_classification_forest(Dataset*, Labels*, ForestConfig*)
    float* forest_classify(Dataset* dataset, void* forest_p, ForestConfig* config)
    double_vec_t forest_get_feature_importances(void* forest_p)

    void* create_scythe()
    void add_tree_to_scythe(void* scythe_p, void* tree_p)
    int forest_prune_height(void* scythe_p, void* forest_p, size_t max_height)
    void restore_pruning(void* scythe_p, int pruning_id)
    void prune(void* scythe_p, int pruning_id)

    void api_test(Dataset*)


cdef Dataset to_dataset(cnp.ndarray X)
cdef MDDataset to_md_dataset(cnp.ndarray X)
cdef Labels to_labels(cnp.ndarray y)
cdef cnp.ndarray ptr_to_cls_predictions(float* predictions, size_t n_rows, size_t n_classes)
cdef cnp.ndarray ptr_to_reg_predictions(data_t* predictions, size_t n_rows)
