/**
    cart.hpp
    Grow classification trees and regression trees

    @author Antoine Passemiers
    @version 1.3 12/04/2017
*/

#ifndef CART_HPP_
#define CART_HPP_

#include <queue>
#include <list>

#include "opt.hpp"
#include "../misc/sets.hpp"
#include "../misc/utils.hpp"
#include "heuristics.hpp"
#include "../densities/continuous.hpp"


namespace scythe {

constexpr size_t MAX_N_CLASSES = 100;

constexpr int NO_FEATURE = -1;
constexpr int NO_INSTANCE = 0;
constexpr int NO_SPLIT_VALUE = std::numeric_limits<int>::max();
constexpr int NUM_SPLIT_LABELS = 3;
constexpr int COST_OF_EMPTINESS = std::numeric_limits<int>::max();
constexpr int INFINITE_DEPTH = -1;

// Task of the tree / forest
constexpr int CLASSIFICATION_TASK = 0xF55A90;
constexpr int REGRESSION_TASK     = 0xF55A91;


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

// https://www.codeproject.com/Articles/4795/C-Standard-Allocator-An-Introduction-and-Implement

struct NodeSpace {
    Node*   owner;
    size_t  current_depth;
    size_t* feature_left_bounds;
    size_t* feature_right_bounds;
    double  information_gain;

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
    bool   ordered_queue;
};

struct Splitter {
    int             task;
    Node*           node;
    size_t          n_instances;
    size_t          n_instances_in_node;
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
    SplitManager*   split_manager;

    explicit Splitter(NodeSpace nodespace, TreeConfig* config, size_t n_instances,
        size_t n_features, size_t* belongs_to, VirtualTargets* targets, SplitManager* split_manager);
};

struct Tree {
    Node*         root;
    size_t        n_nodes;
    size_t        n_classes;
    size_t        n_features;
    TreeConfig*   config;
    ptrdiff_t     level;
    SplitManager* split_manager;

    explicit Tree();
    explicit Tree(Node* root, TreeConfig* config, size_t n_features);
    explicit Tree(const Tree&);
};

void ordered_push(std::list<NodeSpace>& queue, NodeSpace nodespace, bool ordered);

NodeSpace newNodeSpace(Node* owner, size_t n_features, Density* densities);

NodeSpace copyNodeSpace(const NodeSpace& node_space, size_t n_features);

double getFeatureCost(size_t* const, size_t* const, size_t n_classes);

double informationGain(size_t*, size_t*, size_t*, size_t);

void initRoot(Node* root, VirtualTargets* const targets, size_t n_instances, size_t n_classes);

Tree* CART(VirtualDataset* dataset, VirtualTargets* const targets, TreeConfig* config, Density* densities);

Tree* CART(VirtualDataset* dataset, VirtualTargets* const targets, TreeConfig* config, Density* densities, size_t* belongs_to);

float* classifyFromTree(VirtualDataset* data, size_t n_instances, size_t n_features,
                        Tree* const tree, TreeConfig* config);

data_t* predict(VirtualDataset* data, size_t n_instances, size_t n_features,
                Tree* const tree, TreeConfig* config);

double evaluatePartitions(const VirtualDataset* RESTRICT data, const Density* RESTRICT density, 
    const Splitter* RESTRICT splitter, size_t k);

double evaluatePartitionsWithRegression(VirtualDataset* data, Density* density,
                                 Splitter* splitter, size_t k);

double evaluateBySingleThreshold(Splitter* splitter, Density* density, const VirtualDataset* data);

double evaluateByThreshold(Splitter* splitter, Density* density, const VirtualDataset* data);

double fastEvaluateByThreshold(Splitter* splitter, Density* density, VirtualDataset* data);

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

inline float pow2(float probability) {
    /**
        Computes a single term of the Gini coefficient

        @param probability
            Probability of belonging to a certain class
        @return The ith term of the Gini coefficient
    */
    return probability * probability;
}

template<typename T>
void count_instances(T* RESTRICT contiguous_data, label_t* RESTRICT contiguous_labels, size_t* counter_ptrs[2], size_t n_instances_in_node, double split_value) {
    #ifdef _OMP
        #pragma omp simd aligned(contiguous_data : 32)
    #endif
    // Canonical loop form (signed variable, no data dependency, no virtual call...)
    for (signed int j = 0; j < n_instances_in_node; j++) {
        /**
        if (contiguous_data[j] >= split_value) {
            counters_right[contiguous_labels[j]]++;
        }
        else {
            counters_left[contiguous_labels[j]]++;
        }
        */
        counter_ptrs[(contiguous_data[j] >= static_cast<T>(split_value))][contiguous_labels[j]]++;
    }
}

} // namespace

#endif // CART_HPP_