/**
    classification_complete_rf.cpp
    Classification completely-random forests

    @author Antoine Passemiers
    @version 1.0 13/06/2017
*/

#include "classification_complete_rf.hpp"

ClassificationCompleteRF::ClassificationCompleteRF
        (ForestConfig* config, size_t n_instances, size_t n_features) :
        Forest::Forest(config, n_instances, n_features) {
    Forest::base_tree_config.task = gbdf::CLASSIFICATION_TASK;
    Forest::base_tree_config.is_complete_random = true;
    this->score_metric = std::move(
        std::shared_ptr<ClassificationError>(
            new MultiLogLossError(config->n_classes, n_instances)));
}

void ClassificationCompleteRF::fitNewTree(TrainingSet dataset) {
    std::shared_ptr<size_t> subset = createSubsetWithReplacement(
        dataset.n_instances, config.bag_size);
    DirectDataset* direct_dataset = new DirectDataset(
        dataset.data, dataset.n_instances, dataset.n_features);
    std::shared_ptr<Tree> new_tree = std::shared_ptr<Tree>(CART(
        direct_dataset,
        dataset.targets, 
        &(Forest::base_tree_config),
        this->densities.get(),
        subset.get()));
    Forest::trees.push_back(new_tree);
}

void ClassificationCompleteRF::preprocessDensities(TrainingSet dataset) {
    DirectDataset* direct_dataset = new DirectDataset(
        dataset.data, dataset.n_instances, dataset.n_features);
    this->densities = std::move(std::shared_ptr<Density>(computeDensities(
        direct_dataset, 
        dataset.n_instances, 
        dataset.n_features,
        Forest::base_tree_config.n_classes, 
        Forest::base_tree_config.nan_value, 
        Forest::base_tree_config.partitioning)));
}

void ClassificationCompleteRF::fit(TrainingSet dataset) {
    // Compute density functions of all features
    this->preprocessDensities(dataset);

    // Fitting each individual tree
    #pragma omp parallel for num_threads(Forest::config.n_jobs)
    for (uint n_trees = 0; n_trees < Forest::config.n_iter; n_trees++) {
        this->fitNewTree(dataset);
    }
}

float* ClassificationCompleteRF::classify(Dataset dataset) {
    size_t n_classes = Forest::config.n_classes;
    size_t n_instances = dataset.n_rows;
    size_t n_probs = n_classes * n_instances;
    size_t n_trees = trees.size();
    DirectDataset* direct_dataset = new DirectDataset(
        dataset.data, dataset.n_rows, dataset.n_cols);
    float* probabilities = new float[n_probs]();
    for (unsigned int i = 0; i < n_trees; i++) {
        std::shared_ptr<Tree> tree = trees.at(i);
        float* predictions = classifyFromTree(
            direct_dataset,
            dataset.n_rows, 
            dataset.n_cols,
            tree.get(),
            &base_tree_config);
        for (unsigned int k = 0; k < n_probs; k++) {
            probabilities[k] += predictions[k];
        }
    }
    for (unsigned int k = 0; k < n_probs; k++) {
        probabilities[k] /= static_cast<float>(n_trees);
    }
    return probabilities;
}