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

void ClassificationRF::fitNewTree(TrainingSet dataset, std::shared_ptr<size_t> subset) {
    std::shared_ptr<Tree> new_tree = std::shared_ptr<Tree>(CART(
        dataset,
        &(Forest::base_tree_config),
        this->densities.get(),
        subset.get()));
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

    uint n_trees = 0;
    while (n_trees++ < Forest::config.n_iter) {
        std::shared_ptr<size_t> subset = createSubsetWithReplacement(
            dataset.n_instances, config.bag_size);
        this->fitNewTree(dataset, subset);
    }
}

float* ClassificationRF::classify(TrainingSet dataset) {
    size_t n_classes = Forest::config.n_classes;
    size_t n_instances = dataset.n_instances;
    size_t n_probs = n_classes * n_instances;
    size_t n_trees = trees.size();

    float* probabilities = new float[n_probs]();
    for (int i = 0; i < n_trees; i++) {
        std::shared_ptr<Tree> tree = trees.at(i);
        data_t* predictions = predict(
            dataset.data,
            dataset.n_instances, 
            dataset.n_features,
            tree.get(),
            &base_tree_config);
        for (int k = 0; k < n_probs; k++) {
            probabilities[k] += predictions[k];
        }
    }
    for (int k = 0; k < n_probs; k++) {
        probabilities[k] /= n_trees;
    }
    return probabilities;
}