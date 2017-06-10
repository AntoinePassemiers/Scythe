/**
    sets.cpp
    Datasets' structures
    
    @author Antoine Passemiers
    @version 1.0 09/06/2017
*/

#include "sets.hpp"


DirectDataset::DirectDataset(Dataset dataset) {
    this->data = dataset.data;
    this->n_cols = dataset.n_cols;
    this->n_rows = dataset.n_rows;
}

data_t DirectDataset::operator()(size_t i, size_t j) {
    return this->data[i * this->n_cols + j];
}