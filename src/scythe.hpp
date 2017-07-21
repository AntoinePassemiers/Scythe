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

#include "misc/sets.hpp"
#include "forest/forest.hpp"
#include "forest/classification_gb.hpp"
#include "forest/classification_rf.hpp"
#include "forest/classification_complete_rf.hpp"
#include "forest/regression_gb.hpp"
#include "forest/regression_rf.hpp"
#include "forest/regression_complete_rf.hpp"



extern "C" {
    void* fit_classification_tree(scythe::Dataset*, scythe::Labels*, scythe::TreeConfig*);

    void* fit_regression_tree(scythe::Dataset*, scythe::Labels*, scythe::TreeConfig*);

    float* tree_classify(scythe::Dataset*, void*, scythe::TreeConfig*);

    scythe::data_t* tree_predict(scythe::Dataset*, void*, scythe::TreeConfig*);

    void* fit_classification_forest(scythe::Dataset*, scythe::Labels*, scythe::ForestConfig*);
}

#endif // SCYTHE_HPP_
