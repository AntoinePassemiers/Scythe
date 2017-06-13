/**
    scythe.cpp
    Scythe's C API

    @author Antoine Passemiers
    @version 1.0 12/04/2017
*/

#include "scythe.hpp"

extern "C" {

    /* TREE API */

    void* fit_classification_tree(Dataset* dataset, Labels<target_t>* labels, TreeConfig* config) {
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
        config->task = gbdf::CLASSIFICATION_TASK;
        DirectDataset* direct_dataset = new DirectDataset(
            dataset->data, dataset->n_rows, dataset->n_cols);
        Density* densities = computeDensities(direct_dataset, dataset->n_rows, dataset->n_cols,
            config->n_classes, config->nan_value, config->partitioning);
        return static_cast<void*>(CART(direct_dataset, static_cast<target_t*>(labels->data), config, densities));
    }

    void* fit_regression_tree(Dataset* dataset, Labels<data_t>* targets, TreeConfig* config) {
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
        config->task = gbdf::REGRESSION_TASK;
        DirectDataset* direct_dataset = new DirectDataset(
            dataset->data, dataset->n_rows, dataset->n_cols);
        Density* densities = computeDensities(direct_dataset, dataset->n_rows, dataset->n_cols,
            config->n_classes, config->nan_value, config->partitioning);
        return static_cast<void*>(CART(
            direct_dataset, static_cast<target_t*>(targets->data), config, densities));
    }

    float* tree_classify(Dataset* dataset, void* tree_p, TreeConfig* config) {
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
        Tree* tree = static_cast<Tree*>(tree_p);
        DirectDataset* direct_dataset = new DirectDataset(
            dataset->data, dataset->n_rows, dataset->n_cols);
        float* predictions = classifyFromTree(direct_dataset, dataset->n_rows, dataset->n_cols,
            tree, config);
        return predictions;
    }

    data_t* tree_predict(Dataset* dataset, void* tree_p, TreeConfig* config) {
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
        Tree* tree = static_cast<Tree*>(tree_p);
        DirectDataset* direct_dataset = new DirectDataset(
            dataset->data, dataset->n_rows, dataset->n_cols);
        data_t* predictions = predict(direct_dataset, dataset->n_rows, dataset->n_cols, tree, config);
        return predictions;
    }

    /* FOREST API */

    void* fit_classification_forest(Dataset* dataset, Labels<target_t>* labels, ForestConfig* config) {
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
        Forest* forest;
        if (config->type == gbdf::RANDOM_FOREST) {
            forest = new ClassificationRF(config, dataset->n_rows, dataset->n_cols);
        }
        else if (config->type == gbdf::GB_FOREST) {
            forest = new ClassificationGB(config, dataset->n_rows, dataset->n_cols);
        }
        else if (config->type == gbdf::COMPLETE_RANDOM_FOREST) {
            forest = new ClassificationCompleteRF(config, dataset->n_rows, dataset->n_cols);
        }
        else {
            std::cout << "Error: this type of forest does not exist" << std::endl;
        }
        TrainingSet training_set = {
            dataset->data,
            labels->data,
            dataset->n_rows,
            dataset->n_cols };
        forest->fit(training_set);
        return static_cast<void*>(forest);
    }

    float* forest_classify(Dataset* dataset, void* forest_p, ForestConfig* config) {
        float* probabilites;
        ClassificationForest* forest;
        if (config->type == gbdf::RANDOM_FOREST) {
            forest = static_cast<ClassificationRF*>(forest_p);
        }
        else if (config->type == gbdf::GB_FOREST) {
            std::cout << "Error: GB predict function is not implemented" << std::endl;
        }
        else {
            forest = static_cast<ClassificationCompleteRF*>(forest_p);
        }
        probabilites = forest->classify(*dataset);
    }
}
