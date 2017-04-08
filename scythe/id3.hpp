#ifndef ID3_HPP_
#define ID3_HPP_

#include <cassert>
#include <cmath>
#include <math.h>
#include <numeric>
#include <queue>
#include <cstddef>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits>

constexpr int NO_FEATURE = -1;
constexpr int NO_INSTANCE = 0;
constexpr int NO_SPLIT_VALUE = std::numeric_limits<int>::max();
constexpr int NUM_SPLIT_LABELS = 3;
constexpr int COST_OF_EMPTINESS = std::numeric_limits<int>::max();
constexpr int INFINITE_DEPTH = -1;

typedef unsigned int uint;
typedef double data_t;
typedef double target_t;

namespace gbdf {
    // Partitioning of the input's density function
    constexpr int QUARTILE_PARTITIONING   = 0xB23A40;
    constexpr int DECILE_PARTITIONING     = 0xB23A41;
    constexpr int PERCENTILE_PARTITIONING = 0xB23A42;

    // Task of the tree / forest
    constexpr int CLASSIFICATION_TASK = 0xF55A90;
    constexpr int REGRESSION_TASK     = 0xF55A91;
}

struct TrainingSet {
    data_t*   const data;
    target_t* const targets;
    size_t    n_instances;
    size_t    n_features;
};

struct ValidationSet {
    data_t*   const data;
    target_t* const targets;
    size_t    n_instances;
    size_t    n_features;
};

struct Node {
    int     id;
    int     feature_id = NO_FEATURE;
    size_t* counters;
    size_t  n_instances = NO_INSTANCE;
    double  score = INFINITY;
    data_t  split_value = NO_SPLIT_VALUE;
    data_t  mean;
    Node*   left_child = nullptr;
    Node*   right_child = nullptr;

    Node(size_t n_classes = 0);
};

struct NodeSpace {
    Node*   owner;
    size_t  current_depth;
    data_t* feature_left_bounds;
    data_t* feature_right_bounds;
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
    Node*   node;
    size_t  n_instances;
    data_t* partition_values;
    size_t  n_classes;
    double  mean_left;
    double  mean_right;
    size_t  n_left;
    size_t  n_right;
    size_t* belongs_to;
    size_t  feature_id;
    size_t  n_features;
    T*      targets;
    data_t  nan_value;
};

struct Tree {
    Node*       root;
    size_t      n_nodes;
    size_t      n_classes;
    size_t      n_features;
    TreeConfig* config;
    ptrdiff_t   level;
};

inline size_t sum_counts(size_t* counters, size_t n_counters);

Density* computeDensities(data_t* data, size_t n_instances, size_t n_features,
                                 size_t n_classes, data_t nan_value);

inline float ShannonEntropy(float probability);

inline float GiniCoefficient(float probability);

double getFeatureCost(Density* density, size_t n_classes);

void initRoot(Node* root, target_t* const targets, size_t n_instances, size_t n_classes);

Tree* ID3(TrainingSet dataset, TreeConfig* config);

float* classify(data_t* const data, size_t n_instances, size_t n_features,
                Tree* const tree, TreeConfig* config);

data_t* predict(data_t* const data, size_t n_instances, size_t n_features,
                Tree* const tree, TreeConfig* config);
template <typename T>
inline double evaluatePartitions(data_t* data, Density* density,
                                 Splitter<T>* splitter, size_t k);

template <typename T>
inline double evaluatePartitionsWithRegression(data_t* data, Density* density,
                                 Splitter<T>* splitter, size_t k);

template <typename T>
double evaluateByThreshold(Splitter<T>* splitter, Density* density,
                           data_t* const data, int partition_value_type);

#include "id3.tpp"

#endif // ID3_HPP_
