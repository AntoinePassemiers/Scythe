/**
    heuristics.hpp
    Meta-heuristics for increasing CART's speed

    @author Antoine Passemiers
    @version 1.3 10/06/2017
*/

#ifndef HEURISTICS_HPP_
#define HEURISTICS_HPP_

#include <cassert>
#include <cmath>
#include <math.h>
#include <numeric>
#include <queue>
#include <cstddef>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits>

#include "../densities/continuous.hpp"


namespace scythe {

struct FeatureInfo {
    size_t* ntimes_best;
    size_t n_values;
};

class SplitManager {
private:
    size_t n_features;
    std::vector<std::shared_ptr<FeatureInfo>> features;
public:
    SplitManager(Density* const densities, size_t n_densities);
    SplitManager(const SplitManager&) = default;
    SplitManager& operator=(const SplitManager&) = default;

    void updateCurrentBestSplit(size_t feature_id, size_t split_id, double score);
    bool shouldEvaluate(size_t feature_id, size_t split_id);
};

void selectFeaturesToConsider(size_t* to_use, size_t n_features, size_t max_n_features);

} // namespace

#endif // HEURISTICS_HPP_