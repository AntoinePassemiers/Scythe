#ifndef ID3_HPP_
#define ID3_HPP_

#include <stddef.h>
#include <queue>

#define NO_FEATURE       -1
#define NO_INSTANCE       0
#define NO_SPLIT_VALUE    INFINITY
#define NUM_SPLIT_LABELS  3
#define COST_OF_EMPTINESS INFINITY


namespace gbdf_part {
    const int QUARTILE_PARTITIONING   = 0xB23A40;
    const int DECILE_PARTITIONING     = 0xB23A41;
    const int PERCENTILE_PARTITIONING = 0xB23A42;
}

namespace gbdf_task {
    const int CLASSIFICATION_TASK = 0xF55A90;
    const int REGRESSION_TASK     = 0xF55A91;
}

typedef double data_t;
typedef int target_t;

struct Node {
    int id;
    int feature_id;
    size_t* counters;
    size_t n_instances;
    double score;
    double split_value;
    struct Node* left_child;
    struct Node* right_child;
};

struct TreeConfig {
    bool   is_incremental;
    double min_threshold;
    size_t max_height;
    size_t n_classes;
    size_t max_nodes;
    int partitioning;
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

struct Splitter {
    struct Node* node;
    size_t n_instances;
    data_t* partition_values;
    size_t n_classes;
    size_t* belongs_to;
    size_t feature_id;
    size_t n_features;
    target_t* targets; 
    data_t nan_value;
};

struct Tree {
    struct Node* root;
    size_t n_nodes;
    size_t n_classes;
    size_t n_features;
    struct TreeConfig* config;
};


inline float log_2(float value);

inline size_t sum_counts(size_t* counters, size_t n_counters);

struct Node* newNode(size_t n_classes);

extern inline float ShannonEntropy(float probability);

extern inline float GiniCoefficient(float probability);

struct Density* computeDensities(data_t* data, size_t n_instances, size_t n_features,
                                 size_t n_classes, data_t nan_value);

double evaluatePartitions(data_t* data, struct Density* density,
                          struct Splitter* splitter, size_t k);

extern inline double getFeatureCost(struct Density* density, size_t n_classes);

double evaluateByThreshold(struct Splitter* splitter, struct Density* density, 
                           data_t* data, int partition_value_type);

struct Tree* ID3(data_t* data, target_t* targets, size_t n_instances, size_t n_features,
                 struct TreeConfig* config);

float* classify(data_t* data, size_t n_instances, size_t n_features,
                struct Tree* tree, struct TreeConfig* config);

#endif // ID3_HPP_