#include "scythe.hpp"

struct Dataset {
    data_t* data;
    size_t n_rows;
    size_t n_cols;
};

struct Labels {
    target_t* data;
    size_t n_rows;
};

struct GroundTruth {
    data_t* data;
    size_t n_rows;
};

extern "C" {
    void* fit(Dataset* dataset, Labels* labels, TreeConfig* config) {
        struct Tree* tree = ID3(dataset->data, labels->data, dataset->n_rows,
            dataset->n_cols, config);
        return (void*) tree;
    }

    float* predict(Dataset* dataset, void* tree_p, TreeConfig* config) {
        struct Tree* tree = static_cast<struct Tree*>(tree_p);
        float* predictions = classify(dataset->data, dataset->n_rows, dataset->n_cols,
            tree, config);
        return predictions;
    }
}
