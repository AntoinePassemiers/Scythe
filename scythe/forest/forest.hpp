/**
    forest.hpp
    Forest abstract class and configurations

    @author Antoine Passemiers
    @version 1.0 12/04/2017
*/

#ifndef FOREST_HPP_
#define FOREST_HPP_

#include "../metrics/metrics.hpp"
#include "../tree/cart.hpp"


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
    size_t    max_n_nodes          = 30;
    float     learning_rate        = 0.001f;
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

class Forest {
protected:
    size_t n_instances;
    size_t n_features;

    ForestConfig config;

    Tree base_tree;
    std::vector<std::shared_ptr<Tree>> trees;

    ptrdiff_t prediction_state;

public:
    TreeConfig base_tree_config;
    TreeConfig grad_trees_config;

    Forest(size_t n_instances, size_t n_features) : 
        n_instances(n_instances),
        n_features(n_features),
        config(), 
        base_tree(), 
        base_tree_config(), 
        grad_trees_config(),
        trees(), 
        prediction_state(0) {};
    virtual void init() = 0;
    virtual float* fitBaseTree(TrainingSet) = 0;
    virtual void fit(TrainingSet) = 0;
    virtual ~Forest() = default;
};

#endif // FOREST_HPP_