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
    size_t  n_instances = NO_INSTANCE;
    data_t  split_value = NO_SPLIT_VALUE;
    union { // Anonymous union
        size_t* counters = nullptr; // Classification case
        data_t  mean;               // Regression case
    };
    Node*   left_child = nullptr;
    Node*   right_child = nullptr;

    explicit Node() = default;
    explicit Node(size_t n_classes, int id, size_t n_instances);
};

// Nodes with reasonable size
static_assert(sizeof(Node) < 100, "Node size is too large");


struct NodeSpace {
    Node*   owner;
    size_t  current_depth;
    size_t* feature_left_bounds;
    size_t* feature_right_bounds;

    explicit NodeSpace(Node* owner, size_t n_features, Density* densities);
    explicit NodeSpace(const NodeSpace& node_space, size_t n_features);
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

struct Splitter {
    int             task;
    Node*           node;
    size_t          n_instances;
    size_t          n_instances_in_node;
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

    explicit Splitter(NodeSpace nodespace, TreeConfig* config, size_t n_instances,
        size_t n_features, size_t* belongs_to, VirtualTargets* targets);
};

struct Tree {
    Node*       root;
    size_t      n_nodes;
    size_t      n_classes;
    size_t      n_features;
    TreeConfig* config;
    ptrdiff_t   level;

    explicit Tree();
    explicit Tree(Node* root, TreeConfig* config, size_t n_features);
};

NodeSpace newNodeSpace(Node* owner, size_t n_features, Density* densities);

NodeSpace copyNodeSpace(const NodeSpace& node_space, size_t n_features);

double getFeatureCost(Density* density, size_t n_classes);

void initRoot(Node* root, VirtualTargets* const targets, size_t n_instances, size_t n_classes);

Tree* CART(VirtualDataset* dataset, VirtualTargets* const targets, TreeConfig* config, Density* densities);

Tree* CART(VirtualDataset* dataset, VirtualTargets* const targets, TreeConfig* config, Density* densities, size_t* belongs_to);

float* classifyFromTree(VirtualDataset* data, size_t n_instances, size_t n_features,
                        Tree* const tree, TreeConfig* config);

data_t* predict(VirtualDataset* data, size_t n_instances, size_t n_features,
                Tree* const tree, TreeConfig* config);

double evaluatePartitions(VirtualDataset* data, Density* density, Splitter* splitter, size_t k);

double evaluatePartitionsWithRegression(VirtualDataset* data, Density* density,
                                 Splitter* splitter, size_t k);

double evaluateByThreshold(Splitter* splitter, Density* density, const VirtualDataset* data);


inline size_t sum_counts(size_t* counters, size_t n_counters) {
    /**
        Sums the integer values stored in counters

        @param counters
            Counters of instances per class (for classification)
        @param n_counters
            Number of counters
        @return Sum of the counters
    */
    return std::accumulate(counters, counters + n_counters, 0);
}

inline float ShannonEntropy(float probability) {
    /**
        Computes a single term of the Shannon entropy

        @param probability
            Probability of belonging to a certain class
        @return The ith term of the Shannon entropy
    */
    return -probability * std::log2(probability);
}

inline float pow2(float probability) {
    /**
        Computes a single term of the Gini coefficient

        @param probability
            Probability of belonging to a certain class
        @return The ith term of the Gini coefficient
    */
    return probability * probability;
}

#endif // CART_HPP_
