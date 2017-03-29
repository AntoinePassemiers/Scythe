#ifndef SCYTHE_HPP_
#define SCYTHE_HPP_

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string>

#include "forest.hpp"

struct Dataset {
    data_t* data;
    size_t n_rows;
    size_t n_cols;
};

struct Labels {
    target_t* data;
    size_t n_rows;
};

struct GroundTruth {
    data_t* data;
    size_t n_rows;
};

extern "C" {
    void* fit(Dataset* dataset, Labels* labels, TreeConfig* config);

    float* predict(Dataset* dataset, void* tree_p, TreeConfig* config);
}

#endif // SCYTHE_HPP_
