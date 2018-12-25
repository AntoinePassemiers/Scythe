/**
    forest.hpp
    Forest abstract class and configurations

    @author Antoine Passemiers
    @version 1.0 12/04/2017
*/

#ifndef FOREST_HPP_
#define FOREST_HPP_

#include <limits>

#include "../metrics/metrics.hpp"
#include "../tree/cart.hpp"
#include "../tree/pruning.hpp"
#include "../densities/continuous.hpp"
#include "../densities/grayscale.hpp"
#include "../densities/proba.hpp"


namespace scythe {

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

struct ForestConfig {
    int       type                 = RANDOM_FOREST;
    int       task                 = CLASSIFICATION_TASK;
    size_t    n_classes            = 2;
    int       score_metric         = MLOG_LOSS;
    size_t    n_iter               = 100;
    size_t    max_n_trees          = 150;
    size_t    max_n_nodes          = 30;
    size_t    max_n_features       = std::numeric_limits<size_t>::max();
    float     learning_rate        = 0.001f;
    size_t    n_leaves             = 1023;
    size_t    n_jobs               = 1;
    size_t    n_samples_per_leaf   = 50;
    int       regularization       = REG_L1;
    float     bagging_fraction     = 1.0;
    size_t    early_stopping_round = 300;
    int       boosting_method      = GRADIENT_BOOSTING;
    int       max_depth            = INFINITE_DEPTH;
    float     l1_lambda            = 0.1f;
    float     l2_lambda            = 0.1f;
    float     seed                 = 4.f;
    int       verbose              = true;
    data_t    nan_value            = std::numeric_limits<data_t>::quiet_NaN();
    double    min_threshold        = 1e-06;
    bool      ordered_queue        = false;
    int       partitioning         = 100;
    float*    class_weights        = NO_CLASS_WEIGHT;
};

class Forest {
protected:
    size_t n_instances;
    size_t n_features;

    ForestConfig config;

    Tree base_tree; // TODO : keep it ?
    std::vector<std::shared_ptr<Tree>> trees;

    std::ptrdiff_t prediction_state;
    TreeConfig base_tree_config;
    std::shared_ptr<Density> densities;
public:

    Forest(ForestConfig* config, size_t n_instances, size_t n_features) : 
        n_instances(n_instances),
        n_features(n_features),
        config(),
        base_tree(), 
        trees(),
        prediction_state(0),
        base_tree_config(),
        densities() {
            this->config = *config;
            base_tree_config.nan_value = config->nan_value;
            base_tree_config.n_classes = config->n_classes;
            base_tree_config.is_incremental = false;
            base_tree_config.min_threshold = config->min_threshold;
            base_tree_config.max_height = config->max_depth;
            base_tree_config.max_nodes = config->max_n_nodes;
            base_tree_config.max_n_features = config->max_n_features;
            base_tree_config.partitioning = config->partitioning;
            base_tree_config.ordered_queue = config->ordered_queue;
    }
    virtual ~Forest() = default;
    virtual void fit(VirtualDataset* dataset, VirtualTargets* targets) = 0;
    void preprocessDensities(VirtualDataset* dataset);
    virtual size_t getInstanceStride() = 0;
    void save(std::ofstream& file);
    void load(std::ifstream& file);

    std::vector<std::shared_ptr<Tree>>& getTrees() { return trees; }
};

class ClassificationForest : public Forest {
public:
    ClassificationForest(ForestConfig* config, size_t n_instances, size_t n_features) :
        Forest(config, n_instances, n_features) {}
    ~ClassificationForest() = default;
    virtual float* classify(VirtualDataset* dataset) = 0;
    virtual size_t getInstanceStride() { return config.n_classes; }
};

} // namespace

#endif // FOREST_HPP_
