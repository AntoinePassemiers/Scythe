#ifndef ID3_HPP_
#define ID3_HPP_

#include <assert.h>
#include <cmath>
#include <math.h>
#include <queue>  
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NO_FEATURE       -1
#define NO_INSTANCE       0
#define NO_SPLIT_VALUE    INFINITY
#define NUM_SPLIT_LABELS  3
#define COST_OF_EMPTINESS INFINITY

typedef unsigned int uint;
typedef double data_t;
typedef int target_t;

namespace gbdf_part {
    const int QUARTILE_PARTITIONING   = 0xB23A40;
    const int DECILE_PARTITIONING     = 0xB23A41;
    const int PERCENTILE_PARTITIONING = 0xB23A42;
}

namespace gbdf_task {
    const int CLASSIFICATION_TASK = 0xF55A90;
    const int REGRESSION_TASK     = 0xF55A91;
}

struct Node {
    int     id;
    int     feature_id;
    size_t* counters;
    size_t  n_instances;
    double  score;
    data_t  split_value;
    struct  Node* left_child;
    struct  Node* right_child;
};

struct TreeConfig {
    int    task;
    bool   is_incremental;
    double min_threshold;
    size_t max_height;
    size_t n_classes;
    size_t max_nodes;
    int    partitioning;
    data_t nan_value;
};

struct Density {
    bool    is_categorical;
    data_t  split_value;
    data_t* quartiles;
    data_t* deciles;
    data_t* percentiles;
    size_t* counters_left;
    size_t* counters_right;
    size_t* counters_nan;
};

template <typename T>
struct Splitter {
    int     task;
    struct  Node* node;
    size_t  n_instances;
    data_t* partition_values;
    size_t  n_classes;
    size_t* belongs_to;
    size_t  feature_id;
    size_t  n_features;
    T*      targets; 
    data_t  nan_value;
};

struct Tree {
    struct Node* root;
    size_t n_nodes;
    size_t n_classes;
    size_t n_features;
    struct TreeConfig* config;
};

#include "id3.tpp"

#endif // ID3_HPP_