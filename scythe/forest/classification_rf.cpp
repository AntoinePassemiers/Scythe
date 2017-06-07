/**
    classification_forest.cpp
    Classification random forests

    @author Antoine Passemiers
    @version 1.0 07/06/2017
*/

#include "classification_rf.hpp"


ClassificationRF::ClassificationRF
        (ForestConfig* config, size_t n_instances, size_t n_features) :
        Forest::Forest(config, n_instances, n_features) {
    this->score_metric = std::move(
        std::shared_ptr<ClassificationError>(
            new MultiLogLossError(config->n_classes, n_instances)));
}

void ClassificationRF::init() {}

void ClassificationRF::fitNewTree(TrainingSet dataset, data_t* gradient) {
    std::shared_ptr<Tree> new_tree = std::shared_ptr<Tree>(CART(
        { dataset.data, gradient, dataset.n_instances, dataset.n_features },
        &(Forest::grad_trees_config),
        this->densities.get()));
    Forest::trees.push_back(new_tree);
}

void ClassificationRF::preprocessDensities(TrainingSet dataset) {
    this->densities = std::move(std::shared_ptr<Density>(computeDensities(
        dataset.data, 
        dataset.n_instances, 
        dataset.n_features,
        Forest::base_tree_config.n_classes, 
        Forest::base_tree_config.nan_value, 
        Forest::base_tree_config.partitioning)));
}

void ClassificationRF::fit(TrainingSet dataset) {
    // Compute density functions of all features
    this->preprocessDensities(dataset);

    // Fit the base classification tree

}