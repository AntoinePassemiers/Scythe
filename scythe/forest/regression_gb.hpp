/**
    regression_gb.hpp
    Regression gradient boosting

    @author Antoine Passemiers
    @version 1.0 12/04/2017
*/

#ifndef REGRESSION_GB_HPP_
#define REGRESSION_GB_HPP_

#include "forest.hpp"
#include "../metrics/regression_metrics.hpp"


class RegressionGB : public Forest {
private:
    std::shared_ptr<RegressionError> score_metric;
public:
    RegressionGB(ForestConfig*, size_t, size_t);
    void init();
    void fit(TrainingSet dataset);
    float* fitBaseTree(TrainingSet dataset);
    void fitNewTree(TrainingSet dataset, data_t* gradient);
    data_t* predictGradient(std::shared_ptr<Tree> tree, TrainingSet dataset);
    ~RegressionGB() = default;
};

#endif // REGRESSION_GB_HPP_