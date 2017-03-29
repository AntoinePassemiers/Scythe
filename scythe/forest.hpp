#ifndef FOREST_HPP_
#define FOREST_HPP_

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>

#include "id3.hpp"


class Forest {
private:
    size_t max_n_trees;

    struct Tree* base_tree;
    std::vector<struct Tree*> trees;

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
