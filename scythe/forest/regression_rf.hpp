/**
    regression_rf.hpp
    Regression random forests

    @author Antoine Passemiers
    @version 1.0 12/04/2017
*/

#ifndef REGRESSION_RF_HPP_
#define REGRESSION_RF_HPP_

#include "forest.hpp"
#include "../metrics/regression_metrics.hpp"


class RegressionRF : public Forest {
private:
    std::shared_ptr<RegressionError> score_metric;
public:
    RegressionRF(ForestConfig*, size_t, size_t);
    void init();
    void preprocessDensities(VirtualDataset* dataset);
    void fit(VirtualDataset* dataset, VirtualTargets* targets);
    void fitNewTree(VirtualDataset* dataset, VirtualTargets* targets);
    ~RegressionRF() = default;
};

#endif // REGRESSION_RF_HPP_