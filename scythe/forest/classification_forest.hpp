#ifndef CLASSIFICATION_FOREST_HPP_
#define CLASSIFICATION_FOREST_HPP_

#include "forest.hpp"
#include "../metrics/classification_metrics.hpp"

class ClassificationForest : public Forest {
private:
    std::shared_ptr<ClassificationError> score_metric;
public:
    ClassificationForest(ForestConfig*, size_t, size_t);
    void init();
    void fit(TrainingSet dataset);
    float* fitBaseTree(TrainingSet dataset);
    void fitNewTree(TrainingSet dataset, data_t* gradient);
    data_t* predictGradient(std::shared_ptr<Tree> tree, TrainingSet dataset);
    void applySoftmax(float* probabilities, data_t* F_k);
    ~ClassificationForest() = default;
};

// https://eric.univ-lyon2.fr/~ricco/cours/slides/gradient_boosting.pdf
/*
mu = mean(Y)
dY = Y - mu
for k in range(n_boost):
    Learner[k] = train_regressor(X, dY)
    alpha[k] = 1 # TODO
    dY = dY - alpha[k] * predict(Learner[k], X)

(n_test, D) = X_test.shape
predict = zeros(n_test, 1)
for k in range(n_boost):
    predict = predict + alpha[k] * predict(Learner[k], X_test)
*/


#endif // CLASSIFICATION_FOREST_HPP_