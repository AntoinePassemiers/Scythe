#ifndef REGRESSION_METRICS_HPP_
#define REGRESSION_METRICS_HPP_

#include "metrics.hpp"

class RegressionError {
protected:
    size_t n_classes;
    size_t n_instances;
    std::shared_ptr<data_t> gradient;
public:
    inline size_t getNumberOfClasses() { return this->n_classes; }
    inline size_t getNumberOfInstances() { return this->n_instances; }
    virtual size_t getNumberOfRequiredTrees() = 0;
    virtual void computeGradient(float* const, target_t* const) = 0;
    virtual loss_t computeLoss(float* const, target_t* const) = 0;
    RegressionError();
    RegressionError(size_t n_classes, size_t n_instances);
    virtual ~RegressionError();
};

#endif // REGRESSION_METRICS_HPP_