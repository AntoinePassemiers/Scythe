#ifndef FOREST_HPP_
#define FOREST_HPP_

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>

#include "id3.hpp"


namespace reg {
    constexpr int REG_L1 = 0x778C10;
    constexpr int REG_L2 = 0x778C11;
}

namespace gbdf_boost {
    constexpr int ADABOOST          = 0x28FE90;
    constexpr int GRADIENT_BOOSTING = 0x28FE91;
}

struct ForestConfig {
    int       task                 = gbdf_task::CLASSIFICATION_TASK;
    size_t    n_classes            = 2;
    size_t    n_iter               = 100;
    float     learning_rate        = 0.1f;
    size_t    n_leaves             = 1023;
    size_t    n_jobs               = 1;
    size_t    n_samples_per_leaf   = 50;
    int       regularization       = reg::REG_L1;
    float     bagging_fraction     = 0.1f;
    size_t    early_stopping_round = 300;
    int       boosting_method      = gbdf_boost::GRADIENT_BOOSTING;
    int       max_depth            = INFINITE_DEPTH;
    float     l1_lambda            = 0.1f;
    float     l2_lambda            = 0.1f;
    float     seed                 = 4.f;
    int       verbose              = true;
};

class Forest {
private:
    size_t max_n_trees;

    Tree base_tree;
    std::vector<Tree*> trees;

public:
    Forest();
    virtual void initForest() = 0;
    virtual ~Forest() = default;
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
    predict = predict + alpha[k] + predict(Learner[k], X_test)
*/
