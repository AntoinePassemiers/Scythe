#ifndef METRICS_HPP_
#define METRICS_HPP_

#include <iostream>
#include <memory>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>

#include "id3.hpp"

/*
Reference
---------

http://luthuli.cs.uiuc.edu/~daf/courses/optimization/papers/2699986.pdf
pg 1201
*/

namespace gbdf {
    // Classification
    constexpr int MLOG_LOSS = 0x7711A0;

    // Regression
    constexpr int MSE = 0xC97B00; // Mean squared error
}

typedef double loss_t;

class ClassificationError {
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
    ClassificationError();
    ClassificationError(size_t n_classes, size_t n_instances);
    virtual ~ClassificationError();
};

class MultiLogLossError : public ClassificationError {
private:
    double stability_threshold = 10.0e-15;
public:
    MultiLogLossError(size_t n_classes, size_t n_instances) {
        ClassificationError::n_classes = n_classes;
        ClassificationError::n_instances = n_instances;
        ClassificationError::gradient = std::move(std::shared_ptr<data_t>(new data_t[n_classes * n_instances]));
    }

    inline size_t getNumberOfRequiredTrees() { return this->n_classes; }

    inline double getStabilityThreshold() { return this->stability_threshold; }

    loss_t computeLoss(float* const probabilities, target_t* const targets) {
        loss_t loss = 0.0;
        for (uint i = 0; i < this->n_instances; i++) {
            for (uint j = 0; j < this->n_classes; j++) {
                if (static_cast<size_t>(targets[i]) == j) {
                    data_t prob = probabilities[i * this->n_classes + j];
                    prob = std::max(std::min(prob, 1.0 - this->stability_threshold), this->stability_threshold);
                    loss -= std::log(prob);
                }
            }
        }
        return loss / static_cast<double>(n_instances);
    }

    void computeGradient(float* const probabilities, target_t* const targets) {
        for (uint i = 0; i < this->n_instances; i++) {
            for (uint j = 0; j < this->n_classes; j++) {
                data_t prob = probabilities[i * this->n_classes + j];
                prob = std::max(std::min(prob, 1.0 - this->stability_threshold), this->stability_threshold);
                if (static_cast<size_t>(targets[i]) == j) {
                    this->gradient.get()[j * this->n_instances + i] = -(1.0 - prob);
                    // printf("%f, ", this->gradient.get()[j * this->n_instances + i]);
                }
                else {
                    this->gradient.get()[j * this->n_instances + i] = -prob;
                }
            }
        }
    }

    inline data_t* getGradientAt(size_t class_id) {
        return this->gradient.get() + this->n_instances * class_id;
    }

    ~MultiLogLossError() = default;
};

#endif // METRICS_H_