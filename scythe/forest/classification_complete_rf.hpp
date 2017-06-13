/**
    classification_complete_rf.hpp
    Classification completely-random forests
    
    @author Antoine Passemiers
    @version 1.0 12/04/2017
*/

#ifndef CLASSIFICATION_COMPLETE_RF_HPP_
#define CLASSIFICATION_COMPLETE_RF_HPP_

#include "../misc/sets.hpp"
#include "forest.hpp"
#include "../metrics/classification_metrics.hpp"
#include "../misc/bagging.hpp"


class ClassificationCompleteRF : public ClassificationForest {
private:
    std::shared_ptr<ClassificationError> score_metric;
    std::shared_ptr<Density> densities;
public:
    ClassificationCompleteRF(ForestConfig*, size_t, size_t);
    void init();
    void preprocessDensities(TrainingSet dataset);
    void fit(TrainingSet dataset);
    void fitNewTree(TrainingSet dataset);
    float* classify(Dataset dataset);
    ~ClassificationCompleteRF() = default;
};

#endif // CLASSIFICATION_COMPLETE_RF_HPP_