/**
    scythe.cpp
    Scythe's C API

    @author Antoine Passemiers
    @version 1.0 12/04/2017
*/

#ifndef SCYTHE_HPP_
#define SCYTHE_HPP_

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string>

#include "tree/pruning.hpp"
#include "misc/sets.hpp"
#include "forest/forest.hpp"
#include "forest/classification_gb.hpp"
#include "forest/classification_rf.hpp"
#include "forest/classification_complete_rf.hpp"
#include "forest/regression_gb.hpp"
#include "forest/regression_rf.hpp"
#include "forest/regression_complete_rf.hpp"


using namespace scythe;

extern "C" {

    struct double_vec_t {
        double* data;
        size_t length;
    };

    void* fit_classification_tree(Dataset*, Labels*, TreeConfig*);

    void* fit_regression_tree(Dataset*, Labels*, TreeConfig*);

    float* tree_classify(Dataset*, void*, TreeConfig*);

    data_t* tree_predict(Dataset*, void*, TreeConfig*);

    double_vec_t tree_get_feature_importances(void*);

    void* fit_classification_forest(Dataset*, Labels*, ForestConfig*);

    float* forest_classify(Dataset* dataset, void* forest_p, ForestConfig* config);

    double_vec_t forest_get_feature_importances(void* forest_p);

    void* create_scythe();

    void add_tree_to_scythe(void* scythe_p, void* tree_p);

    int forest_prune_height(void* scythe_p, void* forest_p, size_t max_height);

    void restore_pruning(void* scythe_p, int pruning_id);

    void prune(void* scythe_p, int pruning_id);

    void api_test(Dataset* dataset);
}

#endif // SCYTHE_HPP_
