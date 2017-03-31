inline size_t sum_counts(size_t* counters, size_t n_counters) {
    return std::accumulate(counters, counters + n_counters, 0);
}

inline float ShannonEntropy(float probability) {
    return -probability * std::log2(probability);
}

inline float GiniCoefficient(float probability) {
    return 1.f - probability * probability;
}

template <typename T>
double evaluatePartitions(data_t* data, Density* density,
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
            target_value = splitter->targets[j];
            data_point = data[j * n_features + i];
            if (data_point == splitter->nan_value) {
                density->counters_nan[target_value]++;
            }
            else if (data_point >= density->split_value) {
                density->counters_right[target_value]++;
            }
            else {
                density->counters_left[target_value]++;
            }
        }
    }
    return getFeatureCost(density, splitter->n_classes);
}

template <typename T>
double evaluatePartitionsWithRegression(data_t* data, Density* density,
                                 Splitter<T>* splitter, size_t k) {

    size_t i = splitter->feature_id;
    size_t n_features = splitter->n_features;
    data_t data_point;
    T y;
    size_t id = splitter->node->id;
    size_t n_left = 0, n_right = 0;
    density->split_value = splitter->partition_values[k];
    data_t split_value = density->split_value;
    T mean_left = 0, mean_right = 0;
    double cost = 0.0;
    for (uint j = 0; j < splitter->n_instances; j++) {
        if (splitter->belongs_to[j] == id) {
            data_point = data[j * n_features + i];
            y = splitter->targets[j];
            if (data_point == splitter->nan_value) {}
            else if (data_point >= split_value) {
                mean_right += y;
                n_right++;
            }
            else {
                mean_left += y;
                n_left++;
            }
        }
    }
    if ((n_left == 0) or (n_right == 0)) { return INFINITY; }
    mean_left /= static_cast<T>(n_left);
    mean_right /= static_cast<T>(n_right);
    for (uint j = 0; j < splitter->n_instances; j++) {
        if (splitter->belongs_to[j] == id) {
            data_point = data[j * n_features + i];
            y = splitter->targets[j];
            if (data_point == splitter->nan_value) {}
            else if (data_point >= split_value) {
                cost += std::abs(mean_right - y); // TODO : use squared error ?
            }
            else {
                cost += std::abs(mean_right - y); // TODO : use squared error ?
            }
        }
    }
    return cost;
}

template <typename T>
double evaluateByThreshold(Splitter<T>* splitter, Density* density,
                           data_t* const data, int partition_value_type) {
    size_t best_split_id = 0;
    double lowest_cost = INFINITY;
    double cost;
    size_t n_partition_values;
    switch(partition_value_type) {
        case gbdf_part::QUARTILE_PARTITIONING:
            splitter->partition_values = density->quartiles;
            n_partition_values = 4;
            break;
        case gbdf_part::DECILE_PARTITIONING:
            splitter->partition_values = density->deciles;
            n_partition_values = 10;
            break;
        case gbdf_part::PERCENTILE_PARTITIONING:
            splitter->partition_values = density->percentiles;
            n_partition_values = 100;
            break;
        default:
            splitter->partition_values = density->percentiles;
            n_partition_values = 100;
    }
    for (uint k = 1; k < n_partition_values - 1; k++) {
        if (splitter->task == gbdf_task::CLASSIFICATION_TASK) {
            cost = evaluatePartitions(data, density, splitter, k);
        }
        else {
            cost = evaluatePartitionsWithRegression(data, density, splitter, k);
        }
        if (cost < lowest_cost) {
            lowest_cost = cost;
            best_split_id = k;
        }
    }
    if (splitter->task == gbdf_task::CLASSIFICATION_TASK) {
        evaluatePartitions(data, density, splitter, best_split_id);
    }
    else {
        evaluatePartitionsWithRegression(data, density, splitter, best_split_id);
    }
    return lowest_cost;
}
