/**
    continuous.cpp
    Compute densities of continuous variables

    @author Antoine Passemiers
    @version 1.0 23/06/2017
*/

#include "continuous.hpp"

#include <algorithm>

namespace scythe {

Density* computeDensities(VirtualDataset* data, size_t n_classes, data_t nan_value, int partitioning) {

    size_t n_instances = data->getNumInstances();
    size_t n_features  = data->getNumFeatures();
    Density* densities = new Density[n_features];

    #ifdef _OMP
        #pragma omp parallel for num_threads(parameters.n_jobs)
    #endif
    for (uint f = 0; f < n_features; f++) {

        size_t nsad = 100; // TODO

        densities[f].values = new data_t[nsad];
        densities[f].n_values = nsad;
        densities[f].counters_left = new size_t[n_classes];
        densities[f].counters_right = new size_t[n_classes];
        densities[f].counters_nan = new size_t[n_classes];
        densities[f].is_categorical = true; // TODO

        std::vector<data_t> vec;
        data->_iterator_begin(f);
        for (uint i = 0; i < n_instances; i++) {
            data_t value = data->_iterator_deref();
            if (value != nan_value) {
                vec.push_back(value);
            }
            data->_iterator_inc();
        }
        std::sort(vec.begin(), vec.end());

        float step_size = static_cast<float>(n_instances) / static_cast<float>(nsad);
        float current_index = 0.0;
        for (uint i = 0; i < nsad; i++) {
            int rounded_index = static_cast<int>(floor(current_index));
            densities[f].values[i] = vec.at(rounded_index);
            current_index += step_size;
        }
    }
    return densities;
}

} // namespace