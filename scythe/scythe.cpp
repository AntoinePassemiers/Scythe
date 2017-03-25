#include "scythe.hpp"

typedef struct _Dataset {
    double* data;
    unsigned int n_rows;
    unsigned int n_cols;
} Dataset;

extern "C" {
    // Tests
    void printArray(Dataset* dataset) {
        for (int i = 0; i < dataset->n_rows; i++) {
            for (int j = 0; j < dataset->n_cols; j++) {
                std::cout << dataset->data[i * dataset->n_cols + j] << ", ";
            }
        }
        std::cout << std::endl;
    } 
}