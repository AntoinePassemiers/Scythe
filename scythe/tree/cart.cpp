/**
    cart.cpp
    Grow classification trees and regression trees

    @author Antoine Passemiers
    @version 1.3 12/04/2017
*/

#include "cart.hpp"

// Constructor for struct Node
Node::Node(size_t n_classes):
    id(0),
    counters(n_classes > 0 ? new size_t[n_classes] : nullptr),
    mean(0) {}

NodeSpace newNodeSpace(Node* owner, size_t n_features, Density* densities) {
    /**
        Factory function for struct NodeSpace. This is being called while
        instantiating the tree's root. When evaluating the split values for
        the root, all the partition values of each features are being considered,
        this is why the left and right bounds of the node space are first set 
        to their maxima.

        @param owner
            Node related to the current space (root of the tree)
        @param n_features
            Number of features in the training set
        @param densities
            Density functions of the features
            There are supposed to be n_features densities
        @return A new NodeSpace for the tree's root
    */
    NodeSpace new_space;
    new_space.current_depth = 1;
    new_space.owner = owner;
    size_t n_bytes = n_features * sizeof(size_t);
    new_space.feature_left_bounds = static_cast<size_t*>(malloc(n_bytes));
    new_space.feature_right_bounds = static_cast<size_t*>(malloc(n_bytes));
    for (uint f = 0; f < n_features; f++) {
        new_space.feature_left_bounds[f] = 1;
        new_space.feature_right_bounds[f] = densities[f].n_values;
    }
    return new_space;
}

NodeSpace copyNodeSpace(const NodeSpace& node_space, size_t n_features) {
    NodeSpace new_space;
    new_space.owner = node_space.owner;
    new_space.current_depth = node_space.current_depth;
    size_t n_bytes = n_features * sizeof(size_t);
    new_space.feature_left_bounds = static_cast<size_t*>(malloc(n_bytes));
    new_space.feature_right_bounds = static_cast<size_t*>(malloc(n_bytes));
    memcpy(new_space.feature_left_bounds, node_space.feature_left_bounds, n_bytes);
    memcpy(new_space.feature_right_bounds, node_space.feature_right_bounds, n_bytes);
    return new_space;
}

Density* computeDensities(data_t* data, size_t n_instances, size_t n_features,
                                 size_t n_classes, data_t nan_value, int partitioning) {
    Density* densities = new Density[n_features];
    data_t* sorted_values;
    for (uint f = 0; f < n_features; f++) {
        sorted_values = new data_t[n_instances];
        densities[f].quartiles = new data_t[4];
        densities[f].deciles = new data_t[10];
        densities[f].percentiles = new data_t[100];
        densities[f].counters_left = new size_t[n_classes];
        densities[f].counters_right = new size_t[n_classes];
        densities[f].counters_nan = new size_t[n_classes];
        // Putting nan values aside
        bool is_categorical = true;
        size_t n_acceptable = 0;
        data_t data_point;
        for (uint i = 0; i < n_instances; i++) {
            data_point = data[i * n_features + f];
            if (data_point != nan_value) {
                sorted_values[n_acceptable] = data_point;
                n_acceptable++;
                if (is_categorical && !(round(data_point) == data_point)) {
                    is_categorical = false;
                }
            }
        }
        densities[f].is_categorical = is_categorical;
        // Sorting acceptable values
        size_t k;
        data_t x;
        for (uint i = 0; i < n_acceptable; i++) {
            x = sorted_values[i];
            k = i;
            while (k > 0 && sorted_values[k - 1] > x) {
                sorted_values[k] = sorted_values[k - 1];
                k--;
            }
            sorted_values[k] = x;
        }
        // Computing quartiles, deciles, percentiles
        float step_size = static_cast<float>(n_acceptable) / 100.0f;
        float current_index = 0.0;
        int rounded_index = 0;
        for (uint i = 0; i < 10; i++) {
            densities[f].deciles[i] = sorted_values[rounded_index];
            for (uint k = 0; k < 10; k++) {
                rounded_index = static_cast<int>(floor(current_index));
                densities[f].percentiles[10 * i + k] = sorted_values[rounded_index];
                current_index += step_size;
            }
        }

        size_t n_distinct = 1;
        x = sorted_values[0];
        for (uint i = 1; i < n_acceptable; i++) {
            if (sorted_values[n_distinct - 1] != sorted_values[i]) {
                sorted_values[n_distinct++] = sorted_values[i];
            }
            x = sorted_values[i];
        }

        size_t n_partition_values;
        switch(partitioning) {
            case gbdf::QUARTILE_PARTITIONING:
                densities[f].values = densities[f].quartiles;
                n_partition_values = 4; break;
            case gbdf::DECILE_PARTITIONING:
                densities[f].values = densities[f].deciles;
                n_partition_values = 10; break;
            case gbdf::PERCENTILE_PARTITIONING:
                densities[f].values = densities[f].percentiles;
                n_partition_values = 100; break;
            default:
                densities[f].values = densities[f].percentiles;
                n_partition_values = 100; break;
        }
        if (n_distinct < n_partition_values) {
            densities[f].n_values = n_distinct;
            densities[f].values = sorted_values;
        }
        else {
            densities[f].n_values = n_partition_values;
            // delete[] sorted_values;
        }
        printf("%i - %i, ", densities[f].n_values, densities[f].is_categorical);
    }
    return densities;
}

double getFeatureCost(Density* density, size_t n_classes) {
    size_t n_left = sum_counts(density->counters_left, n_classes);
    size_t n_right = sum_counts(density->counters_right, n_classes);
    size_t total = n_left + n_right;
    float left_rate = static_cast<float>(n_left) / static_cast<float>(total);
    float right_rate = static_cast<float>(n_right) / static_cast<float>(total);
    if (n_left == 0 || n_right == 0) {
        return COST_OF_EMPTINESS;
    }
    double left_cost = 0.0, right_cost = 0.0;
    size_t* counters_left = density->counters_left;
    size_t* counters_right = density->counters_right;
    if (n_left > 0) {
        size_t n_p;
        for (uint i = 0; i < n_classes; i++) {
            n_p = counters_left[i];
            if (n_p > 0) {
                left_cost += ShannonEntropy(static_cast<float>(n_p) / static_cast<float>(n_left));
            }
        }
        left_cost *= left_rate;
    }
    if (n_right > 0) {
        size_t n_n;
        for (uint i = 0; i < n_classes; i++) {
            n_n = counters_right[i];
            if (n_n > 0) {
                right_cost += ShannonEntropy(static_cast<float>(n_n) / static_cast<float>(n_right));
            }
        }
        right_cost *= right_rate;
    }
    return left_cost + right_cost;
}

Tree* CART(TrainingSet dataset, TreeConfig* config, Density* densities) {
    size_t n_instances = dataset.n_instances;
    size_t* belongs_to = static_cast<size_t*>(calloc(n_instances, sizeof(size_t)));
    return CART(dataset, config, densities, belongs_to);
}

Tree* CART(TrainingSet dataset, TreeConfig* config, Density* densities, size_t* belongs_to) {
    data_t* const data = dataset.data;
    target_t* const targets = dataset.targets;
    size_t n_instances = dataset.n_instances;
    size_t n_features  = dataset.n_features;
    Node* current_node = new Node(config->n_classes);
    current_node->id = 0;
    current_node->n_instances = n_instances;
    current_node->mean = 0.0; // TODO : mean of all the samples
    if (config->task == gbdf::CLASSIFICATION_TASK) {
        memset(current_node->counters, 0x00, config->n_classes * sizeof(size_t));
        for (uint i = 0; i < n_instances; i++) {
            current_node->counters[static_cast<size_t>(targets[i])]++;
        }
    }
    Node* child_node;
    Tree* tree = new Tree;
    tree->root = current_node;
    tree->config = config;
    tree->n_nodes = 1;
    tree->n_classes = config->n_classes;
    tree->n_features = n_features;
    bool still_going = 1;
    size_t** split_sides = new size_t*[2];
    Density* next_density;
    NodeSpace current_node_space = newNodeSpace(current_node, n_features, densities);
    Splitter<target_t> splitter = {
        config->task,
        current_node,
        n_instances,
        nullptr,
        config->n_classes,
        0.0, 0.0,
        0, 0,
        belongs_to,
        static_cast<size_t>(NO_FEATURE),
        n_features,
        targets,
        config->nan_value,
        0,
        current_node_space
    };

    if (config->max_n_features > n_features) { config->max_n_features = n_features; }
    size_t max_n_features = config->max_n_features;
    size_t* features_to_use = static_cast<size_t*>(malloc(n_features * sizeof(size_t)));
    memset(features_to_use, 0x01, n_features * sizeof(size_t));
    uint best_feature = 0;

    std::queue<NodeSpace> queue;
    queue.push(current_node_space);
    while ((tree->n_nodes < config->max_nodes) && !queue.empty() && still_going) {
        current_node_space = queue.front(); queue.pop();
        current_node = current_node_space.owner;
        double e_cost = INFINITY;
        double lowest_e_cost = INFINITY;
        splitter.node = current_node;
        splitter.node_space = current_node_space;

        selectFeaturesToConsider(features_to_use, n_features, config->max_n_features);
        for (uint f = 0; f < n_features; f++) {
            splitter.feature_id = f;
            e_cost = evaluateByThreshold(&splitter, &densities[f], data);
            if (e_cost < lowest_e_cost) {
                lowest_e_cost = e_cost;
                best_feature = f;
            }
        }
        splitter.feature_id = best_feature;
        evaluateByThreshold(&splitter, &densities[best_feature], data); // TODO : redundant calculus
        next_density = &densities[best_feature];
        if ((best_feature != static_cast<uint>(current_node->feature_id))
            || (next_density->split_value != current_node->split_value)) { // TO REMOVE ?
            next_density = &densities[best_feature];
            size_t split_totals[2] = {
                sum_counts(next_density->counters_left, config->n_classes),
                sum_counts(next_density->counters_right, config->n_classes)
            };
            if ((tree->n_nodes < config->max_nodes) &&
                (current_node_space.current_depth < config->max_height) &&
                (((split_totals[0] && split_totals[1])
                    && (config->task == gbdf::CLASSIFICATION_TASK))
                    || ((config->task == gbdf::REGRESSION_TASK)
                    && (splitter.n_left > 0) && (splitter.n_right > 0)))) { 
                Node* new_children = new Node[2];
                data_t split_value = next_density->split_value;
                current_node->feature_id = static_cast<int>(best_feature);
                current_node->split_value = split_value;
                current_node->left_child = &new_children[0];
                current_node->right_child = &new_children[1];

                split_sides[0] = next_density->counters_left;
                split_sides[1] = next_density->counters_right;
                int new_left_bounds[2] = { 
                    current_node_space.feature_left_bounds[best_feature],
                    splitter.best_split_id + 1};
                int new_right_bounds[2] = { 
                    splitter.best_split_id,
                    current_node_space.feature_right_bounds[best_feature]};
                for (uint i = 0; i < 2; i++) {
                    for (uint j = 0; j < n_instances; j++) {
                        bool is_on_the_left = (data[j * n_features + best_feature] < split_value) ? 1 : 0;
                        if (belongs_to[j] == static_cast<size_t>(current_node->id)) {
                            if (is_on_the_left * (1 - i) + (1 - is_on_the_left) * i) {
                                belongs_to[j] = tree->n_nodes;
                            }
                        }
                    }
                    child_node = &new_children[i];
                    child_node->id = static_cast<int>(tree->n_nodes);
                    child_node->split_value = split_value;
                    child_node->n_instances = split_totals[i];
                    child_node->score = COST_OF_EMPTINESS;
                    child_node->feature_id = static_cast<int>(best_feature);
                    child_node->left_child = nullptr;
                    child_node->right_child = nullptr;
                    child_node->counters = new size_t[config->n_classes];
                    memcpy(child_node->counters, split_sides[i], config->n_classes * sizeof(size_t));
                    if (lowest_e_cost > config->min_threshold) {
                        NodeSpace child_space = copyNodeSpace(current_node_space, n_features);
                        child_space.owner = child_node;
                        child_space.current_depth = current_node_space.current_depth + 1;
                        child_space.feature_left_bounds[best_feature] = new_left_bounds[i];
                        child_space.feature_right_bounds[best_feature] = new_right_bounds[i];
                        queue.push(child_space);
                    }
                    ++tree->n_nodes;
                }
                new_children[0].mean = splitter.mean_left;
                new_children[1].mean = splitter.mean_right;
            }
        }
    }
    delete[] features_to_use;
    delete[] split_sides;
    return tree;
}

float* classify(data_t* const data, size_t n_instances, size_t n_features,
                Tree* const tree, TreeConfig* config) {
    assert(config->task == gbdf::CLASSIFICATION_TASK);
    Node *current_node;
    size_t feature;
    size_t n_classes = config->n_classes;
    float* predictions = new float[n_instances * n_classes];
    for (uint k = 0; k < n_instances; k++) {
        bool improving = true;
        current_node = tree->root;
        while (improving) {
            feature = current_node->feature_id;
            if (current_node->left_child != NULL) {
                if (data[k * n_features + feature] >= current_node->split_value) {
                    current_node = current_node->right_child;
                }
                else {
                    current_node = current_node->left_child;
                }
            }
            else {
                improving = false;
            }
        }
        size_t node_instances = current_node->n_instances;
        for (uint c = 0; c < n_classes; c++) {
            predictions[k * n_classes + c] = static_cast<float>(current_node->counters[c]) / static_cast<float>(node_instances);
        }
    }
    return predictions;
}

data_t* predict(data_t* const data, size_t n_instances, size_t n_features,
                Tree* const tree, TreeConfig* config) {
    assert(config->task == gbdf::REGRESSION_TASK);
    Node *current_node;
    size_t feature;
    data_t* predictions = new data_t[n_instances];

    for (uint k = 0; k < n_instances; k++) {
        bool improving = true;
        current_node = tree->root;
        while (improving) {
            feature = current_node->feature_id;
            if (current_node->left_child != NULL) {
                if (data[k * n_features + feature] >= current_node->split_value) {
                    current_node = current_node->right_child;
                }
                else {
                    current_node = current_node->left_child;
                }
            }
            else {
                improving = false;
            }
        }
        predictions[k] = current_node->mean;
        // TODO : define a new type of struct
    }
    return predictions;
}
