/**
    classification_rf.cpp
    Classification random forests
    
    @author Antoine Passemiers
    @version 1.0 12/04/2017
*/

#ifndef CLASSIFICATION_RF_HPP_
#define CLASSIFICATION_RF_HPP_

#include "../misc/sets.hpp"
#include "forest.hpp"
#include "../metrics/classification_metrics.hpp"
#include "../misc/bagging.hpp"


class ClassificationRF : public ClassificationForest {
private:
    std::shared_ptr<ClassificationError> score_metric;
    std::shared_ptr<Density> densities;
public:
    ClassificationRF(ForestConfig*, size_t, size_t);
    void init();
    void preprocessDensities(VirtualDataset* dataset);
    void fit(VirtualDataset* dataset, target_t* targets);
    void fitNewTree(VirtualDataset* dataset, target_t* targets);
    float* classify(VirtualDataset* dataset);
    ~ClassificationRF() = default;
};

#endif // CLASSIFICATION_RF_HPP_