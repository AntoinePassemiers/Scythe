/**
    cart.hpp
    Grow classification trees and regression trees

    @author Antoine Passemiers
    @version 1.3 12/04/2017
*/

#ifndef CART_HPP_
#define CART_HPP_

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

#include "../misc/sets.hpp"
#include "heuristics.hpp"
#include "../densities/continuous.hpp"

constexpr int NO_FEATURE = -1;
constexpr int NO_INSTANCE = 0;
constexpr int NO_SPLIT_VALUE = std::numeric_limits<int>::max();
constexpr int NUM_SPLIT_LABELS = 3;
constexpr int COST_OF_EMPTINESS = std::numeric_limits<int>::max();
constexpr int INFINITE_DEPTH = -1;

namespace gbdf {
    // Task of the tree / forest
    constexpr int CLASSIFICATION_TASK = 0xF55A90;
    constexpr int REGRESSION_TASK     = 0xF55A91;
}

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
    size_t* feature_left_bounds;
    size_t* feature_right_bounds;
};

struct TreeConfig {
    int    task;
    bool   is_incremental;
    double min_threshold;
    size_t max_height;
    size_t max_n_features;
    size_t n_classes;
    size_t max_nodes;
    int    partitioning;
    data_t nan_value;
    bool   is_complete_random;
};

template <typename T>
struct Splitter {
    int             task;
    Node*           node;
    size_t          n_instances;
    data_t*         partition_values;
    size_t          n_classes;
    double          mean_left;
    double          mean_right;
    size_t          n_left;
    size_t          n_right;
    size_t*         belongs_to;
    size_t          feature_id;
    size_t          n_features;
    VirtualTargets* targets;
    data_t          nan_value;
    int             best_split_id;
    NodeSpace       node_space;
    bool            is_complete_random;
};

struct Tree {
    Node*       root;
    size_t      n_nodes;
    size_t      n_classes;
    size_t      n_features;
    TreeConfig* config;
    ptrdiff_t   level;
};

NodeSpace newNodeSpace(Node* owner, size_t n_features, Density* densities);

NodeSpace copyNodeSpace(const NodeSpace& node_space, size_t n_features);

inline size_t sum_counts(size_t* counters, size_t n_counters);

inline float ShannonEntropy(float probability);

inline float GiniCoefficient(float probability);

double getFeatureCost(Density* density, size_t n_classes);

void initRoot(Node* root, VirtualTargets* const targets, size_t n_instances, size_t n_classes);

Tree* CART(VirtualDataset* dataset, VirtualTargets* const targets, TreeConfig* config, Density* densities);

Tree* CART(VirtualDataset* dataset, VirtualTargets* const targets, TreeConfig* config, Density* densities, size_t* belongs_to);

float* classifyFromTree(VirtualDataset* data, size_t n_instances, size_t n_features,
                        Tree* const tree, TreeConfig* config);

data_t* predict(VirtualDataset* data, size_t n_instances, size_t n_features,
                Tree* const tree, TreeConfig* config);

template <typename T>
inline double evaluatePartitions(VirtualDataset* data, Density* density, Splitter<T>* splitter, size_t k);

template <typename T>
inline double evaluatePartitionsWithRegression(VirtualDataset* data, Density* density,
                                 Splitter<T>* splitter, size_t k);

template <typename T>
double evaluateByThreshold(Splitter<T>* splitter, Density* density, const VirtualDataset* data);

#include "cart.tpp"

#endif // CART_HPP_
