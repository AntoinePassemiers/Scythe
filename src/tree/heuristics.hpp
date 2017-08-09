/**
    heuristics.hpp
    Meta-heuristics for increasing CART's speed

    @author Antoine Passemiers
    @version 1.3 10/06/2017
*/

#ifndef HEURISTICS_HPP_
#define HEURISTICS_HPP_

#include <cassert>
#include <math.h>
#include <numeric>
#include <queue>
#include <cstddef>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits>

#include "../densities/density.hpp"


namespace scythe {

struct FeatureInfo {
    size_t* ntimes_best;
    size_t  n_values;
};

class SplitManager {
private:
    size_t n_features;
    size_t n_grown_trees;
    std::vector<std::shared_ptr<FeatureInfo>> features;
    double* feature_importances;
public:
    SplitManager(Density* const densities, size_t n_densities);
    SplitManager(const SplitManager&) = default;
    SplitManager& operator=(const SplitManager&) = default;

    void notifyGrownTree() { n_grown_trees++; }
    void updateCurrentBestSplit(size_t, size_t, double, double, double);
    bool shouldEvaluate(size_t feature_id, size_t split_id);

    size_t getNumFeatures() { return n_features; }
    double* getFeatureImportances() { return feature_importances; }
};

void selectFeaturesToConsider(size_t* to_use, size_t n_features, size_t max_n_features);

} // namespace

#endif // HEURISTICS_HPP_