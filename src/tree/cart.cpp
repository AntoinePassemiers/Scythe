/**
    cart.cpp
    Grow classification trees and regression trees

    @author Antoine Passemiers
    @version 1.3 12/04/2017
*/

#include "cart.hpp"


namespace scythe {

Tree::Tree() :
    root(nullptr), n_nodes(0), n_classes(0), 
    n_features(0), config(nullptr), level(1), split_manager(nullptr) {}

Tree::Tree(Node* root, TreeConfig* config, size_t n_features) :
    root(root), n_nodes(1), n_classes(config->n_classes),
    n_features(n_features), config(config), level(1), split_manager(nullptr) {}

Tree::Tree(const Tree& other) {
    Node* nodes = new Node[other.n_nodes];
    this->root = &nodes[0];
    this->n_nodes = other.n_nodes;
    this->n_classes = other.n_classes;
    this->n_features = other.n_features;
    this->config = other.config;
    this->level = other.level;
    this->split_manager = other.split_manager;

    size_t current_node_id = 0;
    std::queue<Node*> queue;
    queue.push(other.root);
    while (!queue.empty()) {
        Node* next_node = queue.front(); queue.pop();
        std::memcpy(&nodes[current_node_id], next_node, sizeof(Node));
        current_node_id++;
        if (next_node->left_child != nullptr) {
            queue.push(next_node->left_child);
            queue.push(next_node->right_child);
        }
    }
}

Node::Node(size_t n_classes, int id, size_t n_instances):
    id(id),
    n_instances(n_instances),
    counters(n_classes > 0 ? new (std::nothrow) size_t[n_classes] : nullptr) {

    memset(counters, 0x00, n_classes * sizeof(size_t));
    }

NodeSpace::NodeSpace(Node* owner, size_t n_features, Density* densities) :
    owner(owner),
    current_depth(1),
    feature_left_bounds(new size_t[n_features]),
    feature_right_bounds(new size_t[n_features]) {
    /**
        This constructor is called while instantiating the tree's root. 
        When evaluating the split values for
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
    for (uint f = 0; f < n_features; f++) {
        feature_left_bounds[f] = 1;
        feature_right_bounds[f] = densities[f].n_values;
    }
}

NodeSpace::NodeSpace(const NodeSpace& node_space, size_t n_features) :
    owner(node_space.owner), 
    current_depth(node_space.current_depth),
    feature_left_bounds(new size_t[n_features]),
    feature_right_bounds(new size_t[n_features]) {
    size_t n_bytes = n_features * sizeof(size_t);
    memcpy(feature_left_bounds, node_space.feature_left_bounds, n_bytes);
    memcpy(feature_right_bounds, node_space.feature_right_bounds, n_bytes);
}

Splitter::Splitter(NodeSpace node_space, TreeConfig* config, size_t n_instances,
        size_t n_features, size_t* belongs_to, VirtualTargets* targets, SplitManager* split_manager) :
    task(config->task), 
    node(node_space.owner),
    n_instances(n_instances),
    n_instances_in_node(n_instances),
    n_classes(config->n_classes),
    mean_left(0.0),
    mean_right(0.0),
    n_left(0),
    n_right(0),
    belongs_to(belongs_to),
    feature_id(static_cast<size_t>(NO_FEATURE)),
    n_features(n_features),
    targets(targets),
    nan_value(config->nan_value),
    best_split_id(0),
    node_space(node_space),
    is_complete_random(config->is_complete_random),
    split_manager(split_manager) {}


double getFeatureCost(size_t* RESTRICT const counters_left, 
    size_t* RESTRICT const counters_right, size_t n_classes) {

    size_t n_left = sum_counts(counters_left, n_classes);
    size_t n_right = sum_counts(counters_right, n_classes);
    if (n_left == 0 || n_right == 0) {
        return COST_OF_EMPTINESS;
    }
    double left_cost = 0.0;
    for (uint i = 0; i < n_classes; i++) {
        left_cost += pow2(static_cast<float>(counters_left[i]) / static_cast<float>(n_left));
    }
    left_cost = (1.0 - left_cost);
    double right_cost = 0.0;
    for (uint i = 0; i < n_classes; i++) {
        right_cost += pow2(static_cast<float>(counters_right[i]) / static_cast<float>(n_right));
    }
    right_cost = (1.0 - right_cost);
    float left_rate = static_cast<float>(n_left) / static_cast<float>(n_left + n_right);
    return left_cost * left_rate + right_cost * (1.0 - left_rate);
}

double informationGain(
    size_t* counters, size_t* counters_left, size_t* counters_right, size_t n_classes) {

    double gini = getFeatureCost(counters_left, counters_right, n_classes);
    size_t n_total = sum_counts(counters, n_classes);
    double cost = 0.0;
    for (uint i = 0; i < n_classes; i++) {
        cost += pow2(static_cast<float>(counters[i]) / static_cast<float>(n_total));
    }
    cost = (1.0 - cost);
    double gain = cost - gini;
    gain = (gain < 0.0) ? 0.0 : gain;
    return gain;
}

double evaluatePartitions(
    VirtualDataset* RESTRICT data, const Density* RESTRICT density, 
    const Splitter* RESTRICT splitter, size_t k) {
    size_t* counters_left = density->counters_left;
    size_t* counters_right = density->counters_right;
    std::fill(counters_left, counters_left + splitter->n_classes, 0);
    std::fill(counters_right, counters_right + splitter->n_classes, 0);
    // std::fill(density->counters_nan, density->counters_nan + splitter->n_classes, 0);
    fast_data_t split_value = static_cast<fast_data_t>(density->values[k]);
    
    fast_data_t* RESTRICT contiguous_data = data->retrieveContiguousData();
    label_t* RESTRICT contiguous_labels = (*(splitter->targets)).retrieveContiguousData();

    size_t* counter_ptrs[2] = { counters_left, counters_right };
    #ifdef _OMP
        #pragma omp simd aligned(contiguous_data : 32)
    #endif
    // Canonical loop form (signed variable, no data dependency, no virtual call...)
    for (signed int j = 0; j < splitter->n_instances_in_node; j++) {
        /**
        if (contiguous_data[j] >= split_value) {
            counters_right[contiguous_labels[j]]++;
        }
        else {
            counters_left[contiguous_labels[j]]++;
        }
        */
        counter_ptrs[(contiguous_data[j] >= split_value)][contiguous_labels[j]]++;
    }    
    return getFeatureCost(counters_left, counters_right, splitter->n_classes);
}

double evaluatePartitionsWithRegression(VirtualDataset* data, Density* density,
                                 Splitter* splitter, size_t k) {

    size_t i = splitter->feature_id;
    data_t data_point, nan_value = splitter->nan_value;
    double y;
    size_t id = splitter->node->id;
    size_t* belongs_to = splitter->belongs_to;
    size_t n_left = 0, n_right = 0;
    density->split_value = density->values[k];
    VirtualTargets* targets = splitter->targets;
    data_t split_value = density->split_value;
    double mean_left = 0.0, mean_right = 0.0;
    double cost = 0.0;

    for (uint j = 0; j < splitter->n_instances; j++) {
        if (belongs_to[j] == id) {
            data_point = (*data)(j, i);
            y = static_cast<double>((*targets)[j]);
            // if (data_point == nan_value) {}
            if (data_point >= split_value) {
                mean_right += y;
                n_right++;
            }
            else {
                mean_left += y;
                n_left++;
            }
        }
    }
    mean_left /= static_cast<double>(n_left);
    mean_right /= static_cast<double>(n_right);
    splitter->mean_left  = mean_left;
    splitter->mean_right = mean_right;
    splitter->n_left = n_left;
    splitter->n_right = n_right;
    if ((n_left == 0) || (n_right == 0)) { return INFINITY; }
    for (uint j = 0; j < splitter->n_instances; j++) {
        if (splitter->belongs_to[j] == id) {
            data_point = (*data)(j, i);
            y = (*targets)[j];
            // if (data_point == splitter->nan_value) {}
            if (data_point >= split_value) {
                cost += abs(mean_right - y); // TODO : use squared error ?
            }
            else {
                cost += abs(mean_left - y); // TODO : use squared error ?
            }
        }
    }
    // printf("Cost : %f", cost);
    return cost;
}

double evaluateBySingleThreshold(Splitter* splitter, Density* density, const VirtualDataset* data) {
    return 0.0; // TODO
}

double evaluateByThreshold(Splitter* splitter, Density* density, VirtualDataset* data) {
    size_t lower_bound = splitter->node_space.feature_left_bounds[splitter->feature_id];
    size_t upper_bound = splitter->node_space.feature_right_bounds[splitter->feature_id];
    if (lower_bound == upper_bound) { return INFINITY; }
    if (splitter->is_complete_random) {
        // return evaluateBySingleThreshold(splitter, density, data);
        size_t random_bound = lower_bound + (rand() % (upper_bound - lower_bound));
        lower_bound = random_bound;
        upper_bound = random_bound + 1;
    }
    size_t n_classes = splitter->n_classes;

    data->allocateFromSampleMask(
        splitter->belongs_to,
        splitter->node->id,
        splitter->feature_id,
        splitter->n_instances_in_node,
        splitter->n_instances);

    splitter->targets->allocateFromSampleMask(
        splitter->belongs_to,
        splitter->node->id,
        splitter->n_instances_in_node,
        splitter->n_instances);

    size_t best_split_id = 0;
    double lowest_cost = INFINITY;
    size_t best_counters_left[MAX_N_CLASSES];
    size_t best_counters_right[MAX_N_CLASSES];
    for (uint k = lower_bound; k < upper_bound; k++) {
        if (splitter->split_manager->shouldEvaluate(splitter->feature_id, k)) {
            double cost;
            if (splitter->task == CLASSIFICATION_TASK) {
                cost = evaluatePartitions(data, density, splitter, k);
                density->split_value = density->values[k];
            }
            else {
                cost = evaluatePartitionsWithRegression(data, density, splitter, k);
            }
            if (cost < lowest_cost) {
                lowest_cost = cost;
                best_split_id = k;
                memcpy(best_counters_left, density->counters_left, n_classes * sizeof(size_t));
                memcpy(best_counters_right, density->counters_right, n_classes * sizeof(size_t));
            }
            splitter->best_split_id = best_split_id;
            density->split_value = density->values[best_split_id];
        }
    }
    if (splitter->task == CLASSIFICATION_TASK) {
        memcpy(density->counters_left, best_counters_left, n_classes * sizeof(size_t));
        memcpy(density->counters_right, best_counters_right, n_classes * sizeof(size_t));
    }
    else {
        // TODO: Remove function call and save side effects
        evaluatePartitionsWithRegression(data, density, splitter, best_split_id);
    }
    return lowest_cost;
}

Tree* CART(VirtualDataset* dataset, VirtualTargets* targets, TreeConfig* config, Density* densities) {
    size_t n_instances = dataset->getNumInstances();
    size_t* belongs_to = static_cast<size_t*>(calloc(n_instances, sizeof(size_t)));
    return CART(dataset, targets, config, densities, belongs_to);
}

Tree* CART(VirtualDataset* dataset, VirtualTargets* targets, TreeConfig* config, Density* densities, size_t* belongs_to) {
    size_t n_instances = dataset->getNumInstances();
    size_t n_features  = dataset->getNumFeatures();
    Node* current_node = new Node(config->n_classes, 0, n_instances);
    if (config->task == CLASSIFICATION_TASK) {
        for (uint i = 0; i < n_instances; i++) {
            current_node->counters[static_cast<size_t>((*targets)[i])]++;
        }
    }
    Node* child_node;
    Tree* tree = new Tree(current_node, config, n_features);
    size_t** split_sides = new size_t*[2];
    Density* next_density;
    NodeSpace current_node_space(current_node, n_features, densities);
    SplitManager* split_manager = densities[0].owner;
    tree->split_manager = split_manager;
    Splitter splitter(current_node_space, config, n_instances, n_features, 
        belongs_to, targets, split_manager);
    Splitter best_splitter = splitter;

    if (config->max_n_features > n_features) { config->max_n_features = n_features; }
    size_t max_n_features = config->max_n_features;
    uint best_feature = 0;

    std::queue<NodeSpace> queue;
    queue.push(current_node_space);
    while ((tree->n_nodes < config->max_nodes) && !queue.empty()) {

        current_node_space = queue.front(); queue.pop();
        current_node = current_node_space.owner;
        double e_cost = INFINITY;
        double lowest_e_cost = INFINITY;
        splitter.node = current_node;
        splitter.node_space = current_node_space;
        splitter.n_instances_in_node = current_node->n_instances;
        for (size_t f : selectFeaturesToConsider(n_features, max_n_features)) {
            splitter.feature_id = f;
            e_cost = evaluateByThreshold(&splitter, &densities[f], dataset);
            if (e_cost < lowest_e_cost) {
                lowest_e_cost = e_cost;
                best_feature = f;
                best_splitter = splitter;
            }
        }
        splitter.feature_id = best_feature;
        next_density = &densities[best_feature];
        double information_gain = informationGain(current_node->counters, next_density->counters_left,
            next_density->counters_right, config->n_classes);
        split_manager->updateCurrentBestSplit(
            best_feature, splitter.best_split_id, lowest_e_cost, information_gain,
            static_cast<double>(best_splitter.n_instances_in_node) / static_cast<double>(best_splitter.n_instances));

        size_t split_totals[2] = {
            sum_counts(next_density->counters_left, config->n_classes),
            sum_counts(next_density->counters_right, config->n_classes)
        };
        if ((tree->n_nodes < config->max_nodes) && (!std::isinf(lowest_e_cost)) && (information_gain > 0.0) &&
            (current_node_space.current_depth < config->max_height) &&
            (((split_totals[0] && split_totals[1])
                && (config->task == CLASSIFICATION_TASK))
                || ((config->task == REGRESSION_TASK)
                && (best_splitter.n_left > 0) && (best_splitter.n_right > 0)))) {
            Node* new_children[2];
            data_t split_value = next_density->split_value;
            current_node->feature_id = static_cast<int>(best_feature);
            current_node->split_value = split_value;
            current_node->left_child = new_children[0] = static_cast<Node*>(malloc(sizeof(Node)));
            current_node->right_child = new_children[1] = static_cast<Node*>(malloc(sizeof(Node)));

            split_sides[0] = next_density->counters_left;
            split_sides[1] = next_density->counters_right;
            int new_left_bounds[2] = {
                current_node_space.feature_left_bounds[best_feature],
                best_splitter.best_split_id + 1};
            int new_right_bounds[2] = { 
                best_splitter.best_split_id,
                current_node_space.feature_right_bounds[best_feature]};

            dataset->_iterator_begin(best_feature);
            #ifdef _OMP
                #pragma omp simd
            #endif
            for (uint j = 0; j < n_instances; j++) {
                if (belongs_to[j] == static_cast<size_t>(current_node->id)) {
                    // Left child  : belongs_to[j] = tree->n_nodes
                    // Right child : belongs_to[j] = tree->n_nodes + 1
                    belongs_to[j] = tree->n_nodes + (dataset->_iterator_deref() >= split_value);
                }
                dataset->_iterator_inc();
            }
            for (uint i = 0; i < 2; i++) {
                child_node = new_children[i];
                child_node->id = static_cast<int>(tree->n_nodes);
                child_node->split_value = split_value;
                child_node->n_instances = split_totals[i];
                child_node->feature_id = static_cast<int>(best_feature);
                child_node->left_child = nullptr;
                child_node->right_child = nullptr;
                child_node->counters = new (std::nothrow) size_t[config->n_classes];
                memcpy(child_node->counters, split_sides[i], config->n_classes * sizeof(size_t));
                if (lowest_e_cost > config->min_threshold) {
                    NodeSpace child_space(current_node_space, n_features);
                    child_space.owner = child_node;
                    child_space.current_depth = current_node_space.current_depth + 1;
                    child_space.feature_left_bounds[best_feature] = new_left_bounds[i];
                    child_space.feature_right_bounds[best_feature] = new_right_bounds[i];
                    queue.push(child_space);
                    if (child_space.current_depth > tree->level) {
                        tree->level = child_space.current_depth;
                    }
                }
                ++tree->n_nodes;
            }
            if (config->task == REGRESSION_TASK) {
                current_node->left_child->mean = best_splitter.mean_left;
                current_node->right_child->mean = best_splitter.mean_right;
            }
        }
    }
    split_manager->notifyGrownTree();
    // std::cout << "Tree depth : " << tree->level << std::endl;
    // std::cout << "Node count : " << tree->n_nodes << std::endl;
    delete[] split_sides;
    return tree;
}

float* classifyFromTree(VirtualDataset* dataset, size_t n_instances, size_t n_features,
                Tree* const tree, TreeConfig* config) {
    assert(config->task == CLASSIFICATION_TASK);
    size_t n_classes = config->n_classes;
    float* predictions = new float[n_instances * n_classes];
    for (uint k = 0; k < n_instances; k++) {
        bool improving = true;
        Node* current_node = tree->root;
        while (improving) {
            size_t feature = current_node->feature_id;
            if (current_node->left_child != NULL) {
                if ((*dataset)(k, feature) >= current_node->split_value) {
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

data_t* predict(VirtualDataset* data, size_t n_instances, size_t n_features,
                Tree* const tree, TreeConfig* config) {
    assert(config->task == REGRESSION_TASK);
    data_t* predictions = new data_t[n_instances];
    for (uint k = 0; k < n_instances; k++) {
        bool improving = true;
        Node* current_node = tree->root;
        while (improving) {
            size_t feature = current_node->feature_id;
            if (current_node->left_child != NULL) {
                if ((*data)(k, feature) >= current_node->split_value) {
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

} // namespace