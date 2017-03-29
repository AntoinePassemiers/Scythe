#include "scythe.hpp"

extern "C" {
    void* fit(Dataset* dataset, Labels* labels, TreeConfig* config) {
        Tree* tree = ID3(dataset->data, labels->data, dataset->n_rows,
            dataset->n_cols, config);
        return (void*) tree;
    }

    float* predict(Dataset* dataset, void* tree_p, TreeConfig* config) {
        Tree* tree = static_cast<Tree*>(tree_p);
        float* predictions = classify(dataset->data, dataset->n_rows, dataset->n_cols,
            tree, config);
        return predictions;
    }
}
