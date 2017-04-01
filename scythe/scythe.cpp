#include "scythe.hpp"

extern "C" {
    void* fit_classification_tree(Dataset* dataset, Labels<target_t>* labels, TreeConfig* config) {
        return static_cast<void*>(ID3(
            dataset->data, 
            static_cast<target_t*>(labels->data),
            dataset->n_rows, 
            dataset->n_cols, 
            config));
    }

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
}
