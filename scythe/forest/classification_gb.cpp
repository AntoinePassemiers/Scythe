/**
    classification_forest.cpp
    Classification forests

    @author Antoine Passemiers
    @version 1.0 12/04/2017
*/

#include "classification_gb.hpp"

ClassificationGB::ClassificationGB
        (ForestConfig* config, size_t n_instances, size_t n_features) :
        Forest::Forest(config, n_instances, n_features) {
    Forest::base_tree_config.task = gbdf::CLASSIFICATION_TASK;
    Forest::base_tree_config.is_complete_random = false;
    grad_trees_config = Forest::base_tree_config;
    grad_trees_config.task = gbdf::REGRESSION_TASK;
    this->score_metric = std::move(
        std::shared_ptr<ClassificationError>(
            new MultiLogLossError(config->n_classes, n_instances)));
}

float* ClassificationGB::fitBaseTree(VirtualDataset* dataset, VirtualTargets* targets) {
    this->prediction_state = 0;
    this->base_tree = *CART(
        dataset, 
        targets,
        &(Forest::base_tree_config), 
        this->densities.get());

    // Predict with the base tree and compute the gradient of the error
    float* probabilities = classifyFromTree(
        dataset,
        dataset->getNumInstances(),
        dataset->getNumFeatures(),
        &(this->base_tree),
        &(Forest::base_tree_config));
    loss_t loss = this->score_metric.get()->computeLoss(probabilities, targets);
    printf("Iteration %3i / mlog-loss error : %f\n", 0, static_cast<double>(loss));
    return probabilities;
}

void ClassificationGB::fitNewTree(VirtualDataset* dataset, VirtualTargets* gradient) {

    std::shared_ptr<Tree> new_tree = std::shared_ptr<Tree>(CART(
        dataset, gradient, &grad_trees_config, this->densities.get()));
    Forest::trees.push_back(new_tree);
}

data_t* ClassificationGB::predictGradient(std::shared_ptr<Tree> tree, VirtualDataset* dataset) {
    data_t* predictions = predict(
        dataset,
        dataset->getNumInstances(),
        dataset->getNumFeatures(),
        tree.get(),
        &grad_trees_config);
    return predictions;
}

void ClassificationGB::applySoftmax(float* probabilities, data_t* F_k) {
    size_t n_classes = dynamic_cast<ClassificationError*>(
        this->score_metric.get())->getNumberOfClasses();
    for (uint p = 0; p < Forest::n_instances; p++) {
        data_t softmax_divisor = 0.0;
        for (uint i = 0; i < n_classes; i++) {
            softmax_divisor += std::exp(F_k[p * n_classes + i]);
        }
        for (uint i = 0; i < n_classes; i++) {
            probabilities[p * n_classes + i] = static_cast<float>(
                std::exp(F_k[p * n_classes + i]) / softmax_divisor);
        }
    }
}

void ClassificationGB::preprocessDensities(VirtualDataset* dataset) {
    this->densities = std::move(std::shared_ptr<Density>(computeDensities(
        dataset,
        dataset->getNumInstances(),
        dataset->getNumFeatures(),
        Forest::base_tree_config.n_classes, 
        Forest::base_tree_config.nan_value, 
        Forest::base_tree_config.partitioning)));
}

void ClassificationGB::fit(VirtualDataset* dataset, VirtualTargets* targets) {
    // Compute density functions of all features
    this->preprocessDensities(dataset);

    // Fit the base classification tree
    float* probabilities = this->fitBaseTree(dataset, targets);

    size_t n_classes = Forest::config.n_classes;

    data_t* F_k = static_cast<data_t*>(calloc(
        n_classes * dataset->getNumInstances(), sizeof(data_t)));
    assert(n_classes == this->score_metric.get()->getNumberOfRequiredTrees());
    uint n_boost = 0;
    while (n_boost++ < Forest::config.n_iter) {
        this->score_metric.get()->computeGradient(probabilities, targets->getValues());
        for (uint i = 0; i < n_classes; i++) {
            data_t* gradient = dynamic_cast<MultiLogLossError*>(
                this->score_metric.get())->getGradientAt(i);
            DirectTargets* vgradient = new DirectTargets(gradient, dataset->getNumInstances());
            
            // Fit new tree
            this->fitNewTree(dataset, vgradient);
            // Predict with new tree
            data_t* predictions = this->predictGradient(Forest::trees.back(), dataset);

            for (uint p = 0; p < dataset->getNumInstances(); p++) {
                // TODO : Compute gamma according to Friedman's formulas
                F_k[p * n_classes + i] -= Forest::config.learning_rate * predictions[p];
            }
            free(predictions);
        }

        this->applySoftmax(probabilities, F_k);

        loss_t loss = this->score_metric.get()->computeLoss(probabilities, targets);
        printf("Iteration %3i / mlog-loss error : %f\n", n_boost, static_cast<double>(loss));
    }
    free(probabilities);
    free(F_k);
}