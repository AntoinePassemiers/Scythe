template <typename T>
inline double evaluatePartitions(data_t* data, struct Density* density,
                                 struct Splitter<T>* splitter, size_t k) {
    size_t i = splitter->feature_id;
    size_t n_features = splitter->n_features;
    data_t data_point;
    target_t target_value;
    size_t id = splitter->node->id;
    memset(static_cast<void*>(density->counters_left), 0x00, splitter->n_classes * sizeof(size_t));
    memset(static_cast<void*>(density->counters_right), 0x00, splitter->n_classes * sizeof(size_t));
    memset(static_cast<void*>(density->counters_nan), 0x00, splitter->n_classes * sizeof(size_t));
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
inline double evaluatePartitionsWithRegression(data_t* data, struct Density* density,
                                 struct Splitter<T>* splitter, size_t k) {

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
    mean_left /= n_left;
    mean_right /= n_right;
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
double evaluateByThreshold(struct Splitter<T>* splitter, struct Density* density,
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
