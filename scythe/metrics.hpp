#ifndef METRICS_HPP_
#define METRICS_HPP_

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>
#include <boost/shared_ptr.hpp>

#include "id3.hpp"


typedef float score_t;

class ClassificationError {
private:
	size_t n_classes;
	size_t n_instances;
	boost::shared_ptr<data_t> gradient;
public:
	inline size_t getNumberOfClasses() { return this->n_classes; }
	inline size_t getNumberOfInstances() { return this->n_instances; }
	inline boost::shared_ptr<data_t> getGradient();
	virtual size_t getNumberOfRequiredTrees();
	virtual void computeGradient(float* const, target_t* const);
	virtual ~ClassificationError() = 0;
};

class MultiLogLossError : public ClassificationError {
private:
	double stability_threshold = 10.0e-15;
public:
    MultiLogLossError(size_t n_classes, size_t n_instances) : 
    	n_classes(n_classes), n_instances(n_instances) {
    	this->gradient(new data_t[n_classes * n_instances]);
    }
    inline size_t getNumberOfRequiredTrees() { return this->n_classes; }
    inline double getStabilityThreshold() { return this->stability_threshold; }
    void computeGradient(float* probabilities const, target_t* targets const) {
    	for (uint i = 0; i < this->n_instances; i++) {
    		for (uint j = 0; j < this->n_classes; j++) {
    			if (static_cast<size_t>(targets[i]) == j) {
    				data_t prob = probabilities[i * this->n_classes + j];
    				prob = std::max(std::min(prob, 1.0 - this->stability_threshold), this->stability_threshold);
    				this->gradient[j * this->n_instances + i] = -1.0 / prob;
    			}
    			else {
    				this->gradient[j * this->n_instances + i] = 0.0;
    			}
			}
    	}
    }
    ~MultiLogLossError() = default;
};

#endif // METRICS_H_