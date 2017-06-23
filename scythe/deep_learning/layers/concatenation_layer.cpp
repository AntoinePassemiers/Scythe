/**
    concatenation_layer.cpp
    Base layer with concatenation of two feature matrices

    @author Antoine Passemiers
    @version 1.0 23/06/2017
*/

#include "concatenation_layer.hpp"


ConcatenationDataset::ConcatenationDataset(size_t n_instances, size_t n_virtual_features) {
    this->data = static_cast<data_t*>(malloc(n_instances * n_virtual_features * sizeof(data_t)));
    this->n_instances = n_instances;
    this->n_virtual_cols = n_virtual_features;
    this->stride = 0;
}

ConcatenationDataset::ConcatenationDataset(const ConcatenationDataset& other) :
    data(other.data),
    n_instances(other.n_instances),
    n_virtual_cols(other.n_virtual_cols),
    stride(other.stride) {}

ConcatenationDataset& ConcatenationDataset::operator=(const ConcatenationDataset& other) {
    this->data = other.data;
    this->n_instances = other.n_instances;
    this->n_virtual_cols = other.n_virtual_cols;
    this->stride = other.stride;
    return *this;
}

void ConcatenationDataset::concatenate(float* new_data, size_t width) {
    size_t k = this->stride;
    for (unsigned int i = 0; i < this->n_instances; i++) {
        for (unsigned int j = 0; j < width; j++) {
            this->data[k + j] = static_cast<data_t>(new_data[i * width + j]);
        }
        k += this->n_virtual_cols;
    }
    this->stride += width;
}

data_t ConcatenationDataset::operator()(const size_t i, const size_t j) {
    return data[i * n_virtual_cols + j];
}