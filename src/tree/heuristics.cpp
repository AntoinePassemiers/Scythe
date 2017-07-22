/**
    heuristics.cpp
    Meta-heuristics for increasing CART's speed

    @author Antoine Passemiers
    @version 1.3 10/06/2017
*/

#include "heuristics.hpp"


namespace scythe {

SplitManager::SplitManager(Density* const densities, size_t n_features) : 
    n_features(n_features), features() {

    for (size_t f = 0; f < n_features; f++) {
        std::shared_ptr<FeatureInfo> feature(
            static_cast<FeatureInfo*>(malloc(sizeof(FeatureInfo))));
        size_t n_values = densities[f].n_values;
        feature->ntimes_best = new size_t[n_values]();
        feature->n_values = n_values;
        features.push_back(feature);
    }
}

void SplitManager::updateCurrentBestSplit(size_t feature_id, size_t split_id, double score) {
    // TODO : adapt the manager's behavior as a function of the given score
    features.at(feature_id)->ntimes_best[split_id]++;
}

bool SplitManager::shouldEvaluate(size_t feature_id, size_t split_id) {
    // TODO
    return true;
}

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

} // namespace