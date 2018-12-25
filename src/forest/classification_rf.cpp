/**
    classification_forest.cpp
    Classification random forests

    @author Antoine Passemiers
    @version 1.0 07/06/2017
*/

#include "classification_rf.hpp"


namespace scythe {

ClassificationRF::ClassificationRF
        (ForestConfig* config, size_t n_instances, size_t n_features) :
        ClassificationForest::ClassificationForest(config, n_instances, n_features) {
    Forest::base_tree_config.task = CLASSIFICATION_TASK;
    Forest::base_tree_config.is_complete_random = (config->type == COMPLETE_RANDOM_FOREST);   
    if ((Forest::base_tree_config.max_n_features > n_features) ||
        (Forest::base_tree_config.max_n_features == 0)) {
        Forest::base_tree_config.max_n_features = n_features;
    }
    /*
    this->score_metric = std::move(
        std::shared_ptr<ClassificationError>(
            new MultiLogLossError(config->n_classes, n_instances)));
    */
}

void ClassificationRF::fit(VirtualDataset* dataset, VirtualTargets* targets) {
    assert(dataset->getNumInstances() == targets->getNumInstances());
    VirtualDataset* dataset_view;
    VirtualTargets* targets_view;
    std::shared_ptr<Tree> new_tree;
    size_t n_rows = targets->getNumInstances();
    size_t bag_size = config.bagging_fraction * n_rows;
    size_t n_groups = Forest::config.n_iter / config.n_jobs;
    omp_set_num_threads(config.n_jobs);
    for (uint group_id = 0; group_id < n_groups; group_id++) {
        if (config.bagging_fraction < 1.0) {
            std::vector<size_t> indexes = randomSet(bag_size, n_rows);
            dataset_view = dataset->shuffleAndCreateView(indexes);
            targets_view = targets->shuffleAndCreateView(indexes);
        } else {
            dataset_view = dataset;
            targets_view = targets;
        }
        Forest::preprocessDensities(dataset_view);

        #pragma omp parallel
        {
            #pragma omp for num_threads(config.n_jobs)
            for (uint n_trees = 0; n_trees < config.n_jobs; n_trees++) {
                new_tree = std::shared_ptr<Tree>(CART(
                    dataset_view, targets_view, &(Forest::base_tree_config), this->densities.get()));
                Forest::trees.push_back(new_tree);
            }
        }
        if (config.bagging_fraction < 1.0) {
            delete dataset_view;
            delete targets_view;
        }
    }
}

float* ClassificationRF::classify(VirtualDataset* dataset) {
    size_t n_classes = Forest::config.n_classes;
    size_t n_instances = dataset->getNumInstances();
    size_t n_probs = n_classes * n_instances;
    size_t n_trees = trees.size();
    float* probabilities = new float[n_probs]();
    #ifdef _OMP
        #pragma omp parallel for num_threads(config.n_jobs) shared(probabilities)
    #endif
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

} // namespace