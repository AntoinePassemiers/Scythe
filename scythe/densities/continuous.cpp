/**
    continuous.cpp
    Compute densities of continuous variables

    @author Antoine Passemiers
    @version 1.0 23/06/2017
*/

#include "continuous.hpp"


Density* computeDensities(VirtualDataset* data, size_t n_classes, data_t nan_value, int partitioning) {
    /**
        TODO
        ----
        Use STL containers such as vectors
        std::vector<data_t> data;
        data.assign(data_ptr, data_ptr + len);
        std::sort(data.begin(), data.end());
    */

    size_t n_instances = data->getNumInstances();
    size_t n_features  = data->getNumFeatures();
    Density* densities = new Density[n_features];
    #pragma omp parallel for num_threads(parameters.n_jobs)
    for (uint f = 0; f < n_features; f++) {
        data_t* sorted_values = new data_t[n_instances];
        densities[f].quartiles = new data_t[4];
        densities[f].deciles = new data_t[10];
        densities[f].percentiles = new data_t[100];
        densities[f].counters_left = new size_t[n_classes];
        densities[f].counters_right = new size_t[n_classes];
        densities[f].counters_nan = new size_t[n_classes];
        // Putting nan values aside
        bool is_categorical = true;
        size_t n_acceptable = 0;
        for (uint i = 0; i < n_instances; i++) {
            data_t data_point = (*data)(i, f);
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
        for (uint i = 0; i < n_acceptable; i++) {
            data_t x = sorted_values[i];
            size_t k = i;
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
        data_t x = sorted_values[0];
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