/**
    classification_rf.cpp
    Classification complete random forests
    
    @author Antoine Passemiers
    @version 1.0 12/04/2017
*/

#ifndef CLASSIFICATION_COMPLETE_RF_HPP_
#define CLASSIFICATION_COMPLETE_RF_HPP_

#include "forest.hpp"
#include "../metrics/classification_metrics.hpp"


class ClassificationCompleteRF : public Forest {
private:
    std::shared_ptr<ClassificationError> score_metric;
    std::shared_ptr<Density> densities;
public:
    ClassificationCompleteRF(ForestConfig*, size_t, size_t);
    void init();
    void preprocessDensities(TrainingSet dataset);
    void fit(TrainingSet dataset);
    float* fitBaseTree(TrainingSet dataset);
    void fitNewTree(TrainingSet dataset, data_t* gradient);
    data_t* predictGradient(std::shared_ptr<Tree> tree, TrainingSet dataset);
    void applySoftmax(float* probabilities, data_t* F_k);
    ~ClassificationCompleteRF() = default;
};

#endif // CLASSIFICATION_COMPLETE_RF_HPP_