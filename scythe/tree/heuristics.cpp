/**
    heuristics.cpp
    Meta-heuristics for increasing CART's speed

    @author Antoine Passemiers
    @version 1.3 10/06/2017
*/

#include "heuristics.hpp"


void selectFeaturesToConsider(size_t* to_use, size_t n_features, size_t max_n_features) {
    if (max_n_features > n_features) { max_n_features = n_features; }
    memset(to_use, 0x00, n_features * sizeof(size_t));
    std::vector<size_t> indices;
    for (size_t f = 0; f < n_features; f++) {
        indices.push_back(f);
    }
    for (size_t i = 0; i < max_n_features; i++) {
        size_t rng = rand() % (n_features - i);
        size_t random_feature_id = indices.at(rng);
        to_use[random_feature_id] = true;
        indices.erase(indices.begin() + rng);
    }
}