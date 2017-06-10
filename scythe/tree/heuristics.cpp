/**
    heuristics.cpp
    Meta-heuristics for increasing CART's speed

    @author Antoine Passemiers
    @version 1.3 10/06/2017
*/

#include "heuristics.hpp"


void selectFeaturesToConsider(size_t* to_use, size_t n_features, size_t max_n_features) {
    memset(to_use, 0x00, n_features * sizeof(size_t));
    for (size_t i = 0; i < max_n_features; i++) {
        size_t n_remaining_features = rand() % (n_features - i);
        size_t feature_id = 0;
        for (size_t j = 0; j < n_features; j++) {
            if (!to_use[j]) {
                if (feature_id == n_remaining_features) {
                    to_use[j] = 1;
                    break;
                }
                ++feature_id;
            }
        }
    }
}