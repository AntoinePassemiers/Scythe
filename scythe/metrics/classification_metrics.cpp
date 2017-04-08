#include "classification_metrics.hpp"

ClassificationError::ClassificationError() : 
    n_classes(0), n_instances(0), gradient(nullptr) {}

ClassificationError::ClassificationError(size_t n_classes, size_t n_instances) : 
    n_classes(n_classes), n_instances(n_instances), gradient(nullptr) {}

ClassificationError::~ClassificationError() {}