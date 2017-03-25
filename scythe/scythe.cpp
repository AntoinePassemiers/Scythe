#include "scythe.hpp"


typedef struct _Dataset {
    data_t* data;
    size_t n_rows;
    size_t n_cols;
} Dataset;

typedef struct _Labels {
    target_t* data;
    size_t n_rows;
} Labels;

typedef struct _GroundTruth {
    data_t* data;
    size_t n_rows;
} GroundTruth;

extern "C" {
    void printArray(Dataset* dataset) {
        for (int i = 0; i < dataset->n_rows; i++) {
            for (int j = 0; j < dataset->n_cols; j++) {
                std::cout << dataset->data[i * dataset->n_cols + j] << ", ";
            }
        }
        std::cout << std::endl;
    }

    void fit(Dataset* dataset, Labels* labels) {

        struct TreeConfig* config = (struct TreeConfig*) malloc(sizeof(struct TreeConfig));
        config->is_incremental = false;
        config->min_threshold  = 1e-6;
        config->max_height = 10;
        config->n_classes = 3;
        config->max_nodes = 500;
        config->partitioning = gbdf_part::PERCENTILE_PARTITIONING;
        config->nan_value = -1.0;

        struct Tree* tree = ID3(dataset->data, labels->data, dataset->n_rows, 
            dataset->n_cols, config);
    }
}