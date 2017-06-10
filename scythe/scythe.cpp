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
        Density* densities = computeDensities(dataset->data, dataset->n_rows, dataset->n_cols,
            config->n_classes, config->nan_value, config->partitioning);
        return static_cast<void*>(CART(
            { dataset->data, static_cast<target_t*>(labels->data), dataset->n_rows, dataset->n_cols },
            config, densities));
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
        Density* densities = computeDensities(dataset->data, dataset->n_rows, dataset->n_cols,
            config->n_classes, config->nan_value, config->partitioning);
        return static_cast<void*>(CART(
            { dataset->data, static_cast<target_t*>(targets->data), dataset->n_rows, dataset->n_cols },
            config, densities));
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
        float* predictions = classifyFromTree(dataset->data, dataset->n_rows, dataset->n_cols,
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
        data_t* predictions = predict(dataset->data, dataset->n_rows, dataset->n_cols, tree, config);
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
        else {
            // TODO
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
        if (config->type == gbdf::RANDOM_FOREST) {
            ClassificationRF* forest = static_cast<ClassificationRF*>(forest_p);
            probabilites = forest->classify(*dataset);
        }
        else {
            // TODO
        }
    }
}
