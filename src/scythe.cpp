/**
    scythe.cpp
    Scythe's C API

    @author Antoine Passemiers
    @version 1.0 12/04/2017
*/

#include "scythe.hpp"

extern "C" {

    /* TREE API */

    void* fit_classification_tree(
        scythe::Dataset* dataset, scythe::Labels* labels, scythe::TreeConfig* config) {
        /**
            Grows a classification tree and returns it as a void*

            @param dataset
                Training inputs
            @param labels
                Training labels
            @param config
                Parameters of the classification tree
            @return Pointer to the new tree
        */
        config->task = scythe::CLASSIFICATION_TASK;
        config->max_n_features = dataset->n_cols;
        scythe::DirectDataset* direct_dataset = new scythe::DirectDataset(
            dataset->data, dataset->n_rows, dataset->n_cols);
        scythe::DirectTargets* direct_targets = new scythe::DirectTargets(
            labels->data, dataset->n_rows);
        scythe::Density* densities = scythe::computeDensities(
            direct_dataset, config->n_classes, config->nan_value, config->partitioning);
        return static_cast<void*>(scythe::CART(direct_dataset, direct_targets, config, densities));
    }

    void* fit_regression_tree(
        scythe::Dataset* dataset, scythe::Labels* targets, scythe::TreeConfig* config) {
        /**
            Grows a regression tree and returns it as a void*

            @param dataset
                Training inputs
            @param targets
                Training target values
            @param config
                Parameters of the regression tree
            @return Pointer to the new tree
        */
        config->task = scythe::REGRESSION_TASK;
        config->max_n_features = dataset->n_cols;
        scythe::DirectDataset* direct_dataset = new scythe::DirectDataset(
            dataset->data, dataset->n_rows, dataset->n_cols);
        scythe::DirectTargets* direct_targets = new scythe::DirectTargets(
            targets->data, dataset->n_rows);
        scythe::Density* densities = scythe::computeDensities(
            direct_dataset, config->n_classes, config->nan_value, config->partitioning);
        return static_cast<void*>(scythe::CART(
            direct_dataset, direct_targets, config, densities));
    }

    float* tree_classify(
        scythe::Dataset* dataset, void* tree_p, scythe::TreeConfig* config) {
        /**
            Classifies new data instances and estimates the probability of
            belonging to each of the classes. The probabilites are returned
            as a (n_instances, n_features) matrix, stored in C-order.

            @param dataset
                Validation/test set
            @param tree_p
                Fitted classification tree
            @param config
                Parameters of the classification tree
            @return Probabilities
                shape : (n_instances, n_features)
                order : C
        */
        scythe::Tree* tree = static_cast<scythe::Tree*>(tree_p);
        scythe::DirectDataset* direct_dataset = new scythe::DirectDataset(
            dataset->data, dataset->n_rows, dataset->n_cols);
        float* predictions = classifyFromTree(direct_dataset, dataset->n_rows, dataset->n_cols,
            tree, config);
        return predictions;
    }

    scythe::data_t* tree_predict(
        scythe::Dataset* dataset, void* tree_p, scythe::TreeConfig* config) {
        /**
            Predicts outputs based on new data instances.
            The predictions are returned as a pointer to data_t.

            @param dataset
                Validation/test set
            @param tree_p
                Fitted regression tree
            @param config
                Parameters of the classification tree
            @return Predictions
        */
        scythe::Tree* tree = static_cast<scythe::Tree*>(tree_p);
        scythe::DirectDataset* direct_dataset = new scythe::DirectDataset(
            dataset->data, dataset->n_rows, dataset->n_cols);
        scythe::data_t* predictions = scythe::predict(direct_dataset, dataset->n_rows, dataset->n_cols, tree, config);
        return predictions;
    }

    /* FOREST API */

    void* fit_classification_forest(
        scythe::Dataset* dataset, scythe::Labels* labels, scythe::ForestConfig* config) {
        /**
            Fits a classification, based on the training set 
            and the labels, and returns it as a pointer to void.

            @param dataset
                Validation/test set
            @param labels
                Training labels
            @param config
                Parameters of the classification forest
            @return Pointer to the new forest
        */
        scythe::Forest* forest;
        if (config->type == scythe::RANDOM_FOREST) {
            forest = new scythe::ClassificationRF(config, dataset->n_rows, dataset->n_cols);
        }
        else if (config->type == scythe::GB_FOREST) {
            // forest = new ClassificationGB(config, dataset->n_rows, dataset->n_cols);
            std::cout << "Error: gradient boosting is not supported" << std::endl;
        }
        else if (config->type == scythe::COMPLETE_RANDOM_FOREST) {
            forest = new scythe::ClassificationCompleteRF(config, dataset->n_rows, dataset->n_cols);
        }
        else {
            std::cout << "Error: this type of forest does not exist" << std::endl;
        }
        scythe::DirectDataset* vdataset = new scythe::DirectDataset(*dataset);
        scythe::DirectTargets* vtargets = new scythe::DirectTargets(labels->data, dataset->n_rows);
        forest->fit(vdataset, vtargets);
        return static_cast<void*>(forest);
    }

    float* forest_classify(
        scythe::Dataset* dataset, void* forest_p, scythe::ForestConfig* config) {
        float* probabilites;
        scythe::ClassificationForest* forest;
        if (config->type == scythe::RANDOM_FOREST) {
            forest = static_cast<scythe::ClassificationRF*>(forest_p);
        }
        else if (config->type == scythe::GB_FOREST) {
            std::cout << "Error: GB predict function is not implemented" << std::endl;
        }
        else {
            forest = static_cast<scythe::ClassificationCompleteRF*>(forest_p);
        }
        scythe::DirectDataset* vdataset = new scythe::DirectDataset(*dataset);
        probabilites = forest->classify(vdataset);
        return probabilites;
    }
}