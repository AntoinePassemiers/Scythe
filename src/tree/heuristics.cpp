/**
    heuristics.cpp
    Meta-heuristics for increasing CART's speed

    @author Antoine Passemiers
    @version 1.3 10/06/2017
*/

#include "heuristics.hpp"


namespace scythe {

SplitManager::SplitManager(Density* const densities, size_t n_features) : 
    n_features(n_features), n_grown_trees(0), features() {

    feature_importances = new double[n_features]();
    for (size_t f = 0; f < n_features; f++) {
        std::shared_ptr<FeatureInfo> feature(
            static_cast<FeatureInfo*>(malloc(sizeof(FeatureInfo))));
        size_t n_values = densities[f].n_values;
        feature->ntimes_best = new size_t[n_values]();
        feature->n_values = n_values;
        features.push_back(feature);
    }
}

void SplitManager::updateCurrentBestSplit(size_t feature_id, size_t split_id, double score, 
    double information_gain, double weight) {
    // TODO : adapt the manager's behavior as a function of the given score
    features.at(feature_id)->ntimes_best[split_id]++;
    feature_importances[feature_id] += weight * information_gain;
}

bool SplitManager::shouldEvaluate(size_t feature_id, size_t split_id) {
    // TODO
    if ((n_grown_trees > 0) && (features.at(feature_id)->ntimes_best[split_id] == 0)) {
        return true; // false
    }
    return true;
}

std::vector<size_t> selectFeaturesToConsider(size_t n_features, size_t max_n_features) {
    if (max_n_features > n_features) { max_n_features = n_features; }
    std::vector<size_t> features_to_use;
    std::vector<size_t> indices;
    for (size_t f = 0; f < n_features; f++) {
        indices.push_back(f);
    }
    for (size_t i = 0; i < max_n_features; i++) {
        size_t rng = rand() % (n_features - i);
        size_t random_feature_id = indices.at(rng);
        features_to_use.push_back(random_feature_id);
        indices.erase(indices.begin() + rng);
    }
    return features_to_use;
}

} // namespace