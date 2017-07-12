/**
    classification_metrics.cpp
    Metrics for evaluating classification forests

    @author Antoine Passemiers
    @version 1.0 12/04/2017
*/

#include "classification_metrics.hpp"


namespace scythe {

ClassificationError::ClassificationError() : 
    n_classes(0), n_instances(0), gradient(nullptr) {}

ClassificationError::ClassificationError(size_t n_classes, size_t n_instances) : 
    n_classes(n_classes), n_instances(n_instances), gradient(nullptr) {}

ClassificationError::~ClassificationError() {}

}