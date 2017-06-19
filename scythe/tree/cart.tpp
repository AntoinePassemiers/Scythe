/**
    cart.tpp
    Grow classification trees and regression trees

    @author Antoine Passemiers
    @version 1.3 12/04/2017
*/

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

inline float GiniCoefficient(float probability) {
    /**
        Computes a single term of the Gini coefficient

        @param probability
            Probability of belonging to a certain class
        @return The ith term of the Gini coefficient
    */
    return 1.f - probability * probability;
}

template <typename T>
double evaluatePartitions(VirtualDataset* data, Density* density,
                          Splitter<T>* splitter, size_t k) {
    size_t i = splitter->feature_id;
    size_t n_features = splitter->n_features;
    data_t data_point;
    target_t target_value;
    size_t id = splitter->node->id;
    std::fill(density->counters_left, density->counters_left + splitter->n_classes, 0);
    std::fill(density->counters_right, density->counters_right + splitter->n_classes, 0);
    std::fill(density->counters_nan, density->counters_nan + splitter->n_classes, 0);
    density->split_value = splitter->partition_values[k];
    for (uint j = 0; j < splitter->n_instances; j++) {
        if (splitter->belongs_to[j] == id) {
            target_value = (*splitter->targets)[j];
            data_point = (*data)(j, i);
            if (data_point == splitter->nan_value) {
                density->counters_nan[static_cast<size_t>(target_value)]++;
            }
            else if (data_point >= density->split_value) {
                density->counters_right[static_cast<size_t>(target_value)]++;
            }
            else {
                density->counters_left[static_cast<size_t>(target_value)]++;
            }   
        }
    }
    return getFeatureCost(density, splitter->n_classes);
}

template <typename T>
double evaluatePartitionsWithRegression(VirtualDataset* data, Density* density,
                                 Splitter<T>* splitter, size_t k) {

    size_t i = splitter->feature_id;
    size_t n_features = splitter->n_features;
    data_t data_point, nan_value = splitter->nan_value;
    double y;
    size_t id = splitter->node->id;
    size_t* belongs_to = splitter->belongs_to;
    size_t n_left = 0, n_right = 0;
    density->split_value = splitter->partition_values[k];
    VirtualTargets* targets = splitter->targets;
    data_t split_value = density->split_value;
    double mean_left = 0, mean_right = 0;
    double cost = 0.0;

    for (uint j = 0; j < splitter->n_instances; j++) {
        if (belongs_to[j] == id) {
            data_point = (*data)(j, i);
            y = static_cast<double>((*splitter->targets)[j]);
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
    if ((n_left == 0) or (n_right == 0)) { return INFINITY; }
    for (uint j = 0; j < splitter->n_instances; j++) {
        if (splitter->belongs_to[j] == id) {
            data_point = (*data)(j, i);
            y = (*splitter->targets)[j];
            // if (data_point == splitter->nan_value) {}
            if (data_point >= split_value) {
                cost += std::abs(mean_right - y); // TODO : use squared error ?
            }
            else {
                cost += std::abs(mean_left - y); // TODO : use squared error ?
            }
        }
    }
    // printf("Cost : %f", cost);
    return cost;
}

template <typename T>
double evaluateByThreshold(Splitter<T>* splitter, Density* density, VirtualDataset* data) {
    size_t feature_id = splitter->feature_id;
    size_t best_split_id = 0;
    double lowest_cost = INFINITY;
    double cost;
    splitter->partition_values = density->values;

    size_t lower_bound = splitter->node_space.feature_left_bounds[feature_id];
    size_t upper_bound = splitter->node_space.feature_right_bounds[feature_id];
    if (splitter->is_complete_random) {
        size_t random_bound = lower_bound + (rand() % (upper_bound - lower_bound));
        lower_bound = random_bound;
        upper_bound = random_bound + 1;
    }
    for (uint k = lower_bound; k < upper_bound; k++) {
        if (splitter->task == gbdf::CLASSIFICATION_TASK) {
            cost = evaluatePartitions(data, density, splitter, k);
        }
        else {
            cost = evaluatePartitionsWithRegression(data, density, splitter, k);
        }
        if (cost < lowest_cost) {
            lowest_cost = cost;
            best_split_id = k;
        }
    splitter->best_split_id = best_split_id;
    }
    if (splitter->task == gbdf::CLASSIFICATION_TASK) {
        evaluatePartitions(data, density, splitter, best_split_id);
    }
    else {
        evaluatePartitionsWithRegression(data, density, splitter, best_split_id);
    }
    return lowest_cost;
}