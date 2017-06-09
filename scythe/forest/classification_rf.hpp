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


class ClassificationRF : public Forest {
private:
    std::shared_ptr<ClassificationError> score_metric;
    std::shared_ptr<Density> densities;
public:
    ClassificationRF(ForestConfig*, size_t, size_t);
    void init();
    void preprocessDensities(TrainingSet dataset);
    void fit(TrainingSet dataset);
    void fitNewTree(TrainingSet dataset, std::shared_ptr<size_t> subset);
    float* classify(Dataset dataset);
    ~ClassificationRF() = default;
};

#endif // CLASSIFICATION_RF_HPP_