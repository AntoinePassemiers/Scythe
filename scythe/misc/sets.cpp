/**
    sets.cpp
    Datasets' structures
    
    @author Antoine Passemiers
    @version 1.0 09/06/2017
*/

#include "sets.hpp"


DirectDataset::DirectDataset(Dataset dataset) :
    data(dataset.data), n_rows(dataset.n_rows), n_cols(dataset.n_cols) {}

DirectDataset::DirectDataset(data_t* data, size_t n_instances, size_t n_features) :
    data(data), n_rows(n_instances), n_cols(n_features) {}

data_t DirectDataset::operator()(size_t i, size_t j) {
    return this->data[i * this->n_cols + j];
}