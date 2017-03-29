#ifndef ID3_HPP_
#define ID3_HPP_

#include <assert.h>
#include <cmath>
#include <math.h>
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

typedef unsigned int uint;
typedef double data_t;
typedef int target_t;

namespace gbdf_part {
    constexpr int QUARTILE_PARTITIONING   = 0xB23A40;
    constexpr int DECILE_PARTITIONING     = 0xB23A41;
    constexpr int PERCENTILE_PARTITIONING = 0xB23A42;
}

namespace gbdf_task {
    constexpr int CLASSIFICATION_TASK = 0xF55A90;
    constexpr int REGRESSION_TASK     = 0xF55A91;
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

inline size_t sum_counts(size_t* counters, size_t n_counters);

Node* newNode(size_t n_classes);

Density* computeDensities(data_t* data, size_t n_instances, size_t n_features,
                                 size_t n_classes, data_t nan_value);

inline float ShannonEntropy(float probability);

inline float GiniCoefficient(float probability);

inline double getFeatureCost(struct Density* density, size_t n_classes);

void initRoot(Node* root, target_t* const targets, size_t n_instances, size_t n_classes);

Tree* ID3(data_t* const data, target_t* const targets, size_t n_instances,
                 size_t n_features, TreeConfig* config);

float* classify(data_t* const data, size_t n_instances, size_t n_features,
                Tree* const tree, TreeConfig* config);

data_t* regress(data_t* const data, size_t n_instances, size_t n_features,
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
