#include "scythe.hpp"

extern "C" {

    /* TREE API */

    void* fit_classification_tree(Dataset* dataset, Labels<target_t>* labels, TreeConfig* config) {
        return static_cast<void*>(ID3(
            dataset->data, 
            static_cast<target_t*>(labels->data),
            dataset->n_rows, 
            dataset->n_cols, 
            config));
    }

    // TODO : remove this function / Equivalent to fit_classifiction_tree
    void* fit_regression_tree(Dataset* dataset, Labels<data_t>* labels, TreeConfig* config) {
        return static_cast<void*>(ID3(
            dataset->data, 
            static_cast<target_t*>(labels->data),
            dataset->n_rows, 
            dataset->n_cols,
            config));
    }

    float* tree_classify(Dataset* dataset, void* tree_p, TreeConfig* config) {
        Tree* tree = static_cast<Tree*>(tree_p);
        float* predictions = classify(dataset->data, dataset->n_rows, dataset->n_cols,
            tree, config);
        return predictions;
    }

    data_t* tree_predict(Dataset* dataset, void* tree_p, TreeConfig* config) {
        Tree* tree = static_cast<Tree*>(tree_p);
        data_t* predictions = predict(dataset->data, dataset->n_rows, dataset->n_cols,
            tree, config);
        return predictions;
    }

    /* FOREST API */

    void* fit_classification_forest(Dataset* dataset, Labels<target_t>* labels, ForestConfig* config) {
        ClassificationForest* forest = new ClassificationForest(
            config,
            dataset->n_rows,
            dataset->n_cols);
        TrainingSet training_set = {
            dataset->data,
            labels->data,
            dataset->n_rows,
            dataset->n_cols };
        forest->fit(training_set);
        return static_cast<void*>(forest);
    }
}
