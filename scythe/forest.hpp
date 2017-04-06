#ifndef FOREST_HPP_
#define FOREST_HPP_

#include <iostream>
#include <memory>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>

#include "metrics.hpp"
#include "id3.hpp"


namespace gbdf {
    // Regularization method
    constexpr int REG_L1 = 0x778C10;
    constexpr int REG_L2 = 0x778C11;

    // Boosting method
    constexpr int ADABOOST          = 0x28FE90;
    constexpr int GRADIENT_BOOSTING = 0x28FE91;
}

struct ForestConfig {
    int       task                 = gbdf::CLASSIFICATION_TASK;
    size_t    n_classes            = 2;
    int       score_metric         = gbdf::MLOG_LOSS;
    size_t    n_iter               = 100;
    size_t    max_n_trees          = 150;
    float     learning_rate        = 0.1f;
    size_t    n_leaves             = 1023;
    size_t    n_jobs               = 1;
    size_t    n_samples_per_leaf   = 50;
    int       regularization       = gbdf::REG_L1;
    float     bagging_fraction     = 0.1f;
    size_t    early_stopping_round = 300;
    int       boosting_method      = gbdf::GRADIENT_BOOSTING;
    int       max_depth            = INFINITE_DEPTH;
    float     l1_lambda            = 0.1f;
    float     l2_lambda            = 0.1f;
    float     seed                 = 4.f;
    int       verbose              = true;
    data_t    nan_value            = std::numeric_limits<data_t>::quiet_NaN();
};

struct TrainingSet {
    data_t*   const data;
    target_t* const targets;
    size_t    n_instances;
    size_t    n_features;
};

class Forest {
protected:
    size_t n_instances;
    size_t n_features;

    ForestConfig config;

    Tree base_tree;
    std::vector<Tree*> trees;

    ptrdiff_t prediction_state;

public:
    TreeConfig base_tree_config;

    Forest(size_t n_instances, size_t n_features) : 
        n_instances(n_instances),
        n_features(n_features),
        config(), 
        base_tree(), 
        base_tree_config(), 
        trees(), 
        prediction_state(0) {};
    virtual void init() = 0;
    virtual void fit(TrainingSet) = 0;
    virtual ~Forest() = default;
};

class ClassificationForest : public Forest {
private:
    std::unique_ptr<ClassificationError> score_metric;
public:
    ClassificationForest(ForestConfig*, size_t, size_t);
    void init();
    void fit(TrainingSet dataset);
    ~ClassificationForest() = default;
};

#endif // FOREST_HPP_

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
    predict = predict + alpha[k] = predict(Learner[k], X_test)
*/
