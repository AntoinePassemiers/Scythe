/**
    regression_complete_rf.hpp
    Regression complete random forests

    @author Antoine Passemiers
    @version 1.0 12/04/2017
*/

#ifndef REGRESSION_COMPLETE_RF_HPP_
#define REGRESSION_COMPLETE_RF_HPP_

#include "forest.hpp"
#include "../metrics/regression_metrics.hpp"


class RegressionCompleteRF : public Forest {
private:
    std::shared_ptr<RegressionError> score_metric;
public:
    RegressionCompleteRF(ForestConfig*, size_t, size_t);
    void init();
    void fit(TrainingSet dataset);
    float* fitBaseTree(TrainingSet dataset);
    void fitNewTree(TrainingSet dataset, data_t* gradient);
    data_t* predictGradient(std::shared_ptr<Tree> tree, TrainingSet dataset);
    ~RegressionCompleteRF() = default;
};

#endif // REGRESSION_COMPLETE_RF_HPP_