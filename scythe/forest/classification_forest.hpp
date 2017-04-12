#ifndef CLASSIFICATION_FOREST_HPP_
#define CLASSIFICATION_FOREST_HPP_

#include "forest.hpp"
#include "../metrics/classification_metrics.hpp"

class ClassificationForest : public Forest {
private:
    std::shared_ptr<ClassificationError> score_metric;
    std::shared_ptr<Density> densities;
public:
    ClassificationForest(ForestConfig*, size_t, size_t);
    void init();
    void preprocessDensities(TrainingSet dataset);
    void fit(TrainingSet dataset);
    float* fitBaseTree(TrainingSet dataset);
    void fitNewTree(TrainingSet dataset, data_t* gradient);
    data_t* predictGradient(std::shared_ptr<Tree> tree, TrainingSet dataset);
    void applySoftmax(float* probabilities, data_t* F_k);
    ~ClassificationForest() = default;
};

// https://eric.univ-lyon2.fr/~ricco/cours/slides/gradient_boosting.pdf


#endif // CLASSIFICATION_FOREST_HPP_