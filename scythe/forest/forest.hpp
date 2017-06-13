/**
    forest.hpp
    Forest abstract class and configurations

    @author Antoine Passemiers
    @version 1.0 12/04/2017
*/

#ifndef FOREST_HPP_
#define FOREST_HPP_

#include <pthread.h>
#include <omp.h>

#include "../metrics/metrics.hpp"
#include "../tree/cart.hpp"


namespace gbdf {
    // Regularization method
    constexpr int REG_L1 = 0x778C10;
    constexpr int REG_L2 = 0x778C11;

    // Boosting method
    constexpr int ADABOOST          = 0x28FE90;
    constexpr int GRADIENT_BOOSTING = 0x28FE91;

    // Forest type
    constexpr int RANDOM_FOREST          = 0;
    constexpr int COMPLETE_RANDOM_FOREST = 1;
    constexpr int GB_FOREST              = 2;
}

struct ForestConfig {
    int       type                 = gbdf::RANDOM_FOREST;
    int       task                 = gbdf::CLASSIFICATION_TASK;
    size_t    n_classes            = 2;
    int       score_metric         = gbdf::MLOG_LOSS;
    size_t    n_iter               = 100;
    size_t    max_n_trees          = 150;
    size_t    max_n_nodes          = 30;
    size_t    max_n_features       = std::numeric_limits<size_t>::max();
    float     learning_rate        = 0.001f;
    size_t    n_leaves             = 1023;
    size_t    n_jobs               = 1;
    size_t    n_samples_per_leaf   = 50;
    int       regularization       = gbdf::REG_L1;
    size_t    bag_size             = 100;
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

    Tree base_tree; // TODO : keep it ?
    std::vector<std::shared_ptr<Tree>> trees;

    ptrdiff_t prediction_state;
    TreeConfig base_tree_config;
public:

    Forest(ForestConfig* config, size_t n_instances, size_t n_features) : 
        n_instances(n_instances),
        n_features(n_features),
        config(),
        base_tree(), 
        trees(),
        prediction_state(0),
        base_tree_config() {
            this->config = *config;
            base_tree_config.nan_value = config->nan_value;
            base_tree_config.n_classes = config->n_classes;
            base_tree_config.is_incremental = false;
            base_tree_config.min_threshold = 1e-06;
            base_tree_config.max_height = config->max_depth;
            base_tree_config.max_nodes = config->max_n_nodes;
            base_tree_config.max_n_features = config->max_n_features;
            base_tree_config.partitioning = gbdf::PERCENTILE_PARTITIONING;
    }
    virtual void fit(VirtualDataset* dataset, target_t* targets) = 0;
    virtual ~Forest() = default;
};


class ClassificationForest : public Forest {
public:
    ClassificationForest(ForestConfig* config, size_t n_instances, size_t n_features) :
        Forest(config, n_instances, n_features) {}
    ~ClassificationForest() = default;
    virtual float* classify(VirtualDataset* dataset) = 0;
};

#endif // FOREST_HPP_