#include "forest.hpp"

ClassificationForest::ClassificationForest
    (ForestConfig* config, size_t n_instances, size_t n_features) :
    Forest(n_instances, n_features) {
    Forest::config = *config;
    Forest::base_tree_config.task = Forest::config.task;
    Forest::base_tree_config.nan_value = Forest::config.nan_value;
    Forest::base_tree_config.n_classes = Forest::config.n_classes;
    // TODO : other parameters

    score_metric = std::move(
        std::unique_ptr<ClassificationError>(
            new MultiLogLossError(n_instances, n_features)));
}

void ClassificationForest::init() {
}

void ClassificationForest::fit(TrainingSet dataset) {
    // Fit the base classification tree
    this->prediction_state = 0;
    this->base_tree = *ID3(
        dataset.data,
        dataset.targets,
        dataset.n_instances,
        dataset.n_features,
        &(Forest::base_tree_config));

    // Predict with the base tree and compute the gradient of the error
    float* probabilities = classify(
        dataset.data, 
        dataset.n_instances,
        dataset.n_features,
        &(this->base_tree),
        &(Forest::base_tree_config));
    loss_t loss = score_metric.get()->computeLoss(probabilities, dataset.targets);
    printf("mlog-loss error : %f\n", static_cast<double>(loss));

    // score_metric.get()->computeGradient(probabilities, dataset.targets);

    uint n_boost = 0;
    while (n_boost++ < Forest::config.n_iter) {
        // Fit the boosting trees on the gradient
    }
}