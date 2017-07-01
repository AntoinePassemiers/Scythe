/**
    classification_complete_rf.cpp
    Classification completely-random forests

    @author Antoine Passemiers
    @version 1.0 13/06/2017
*/

#include "classification_complete_rf.hpp"


ClassificationCompleteRF::ClassificationCompleteRF
        (ForestConfig* config, size_t n_instances, size_t n_features) :
        ClassificationForest::ClassificationForest(config, n_instances, n_features) {
    Forest::base_tree_config.task = gbdf::CLASSIFICATION_TASK;
    Forest::base_tree_config.is_complete_random = true;
    /*
    this->score_metric = std::move(
        std::shared_ptr<ClassificationError>(
            new MultiLogLossError(config->n_classes, n_instances)));
    */
}

void ClassificationCompleteRF::fitNewTree(VirtualDataset* dataset, VirtualTargets* targets) {
    std::cout << "AAAA" << std::endl;
    std::shared_ptr<size_t> subset = createSubsetWithReplacement(
        dataset->getNumInstances(), config.bag_size);
    std::cout << "BBBB" << std::endl;
    std::cout << "v-dataset shape : " << dataset->getNumInstances() << ", ";
    std::cout << dataset->getNumFeatures() << std::endl;
    std::cout << "v-targets length : " << targets->getNumInstances() << std::endl;
    std::shared_ptr<Tree> new_tree = std::shared_ptr<Tree>(CART(
        dataset,
        targets, 
        &(Forest::base_tree_config),
        this->densities.get(),
        subset.get()));
    std::cout << "CCCC" << std::endl;
    Forest::trees.push_back(new_tree);
}

void ClassificationCompleteRF::fit(VirtualDataset* dataset, VirtualTargets* targets) {
    std::cout << "AAA" << std::endl;
    // Compute density functions of all features
    Forest::preprocessDensities(dataset);
    std::cout << "BBB" << std::endl;
    // Fitting each individual tree
    // #pragma omp parallel for num_threads(parameters.n_jobs)
    for (uint n_trees = 0; n_trees < Forest::config.n_iter; n_trees++) {
        this->fitNewTree(dataset, targets);
    }
    std::cout << "CCC" << std::endl;
}

float* ClassificationCompleteRF::classify(VirtualDataset* dataset) {
    size_t n_classes = Forest::config.n_classes;
    size_t n_instances = dataset->getNumInstances();
    size_t n_probs = n_classes * n_instances;
    size_t n_trees = trees.size();
    float* probabilities = new float[n_probs]();
    #pragma omp parallel for num_threads(parameters.n_jobs) shared(probabilities)
    for (unsigned int i = 0; i < n_trees; i++) {
        std::shared_ptr<Tree> tree = trees.at(i);
        float* predictions = classifyFromTree(
            dataset,
            dataset->getNumInstances(), 
            dataset->getNumFeatures(),
            tree.get(),
            &base_tree_config);
        for (unsigned int k = 0; k < n_probs; k++) {
            probabilities[k] += predictions[k];
        }
        delete[] predictions;
    }
    for (unsigned int k = 0; k < n_probs; k++) {
        probabilities[k] /= static_cast<float>(n_trees);
    }
    return probabilities;
}