/**
    sets.cpp
    Datasets' structures
    
    @author Antoine Passemiers
    @version 1.0 09/06/2017
*/

#include "sets.hpp"


namespace scythe {

void VirtualDataset::allocateFromSampleMask(
    size_t* sample_mask, size_t node_id, size_t feature_id, size_t n_items, size_t n_instances) {
    if (contiguous_data != nullptr) {
        delete[] contiguous_data;
    }
    contiguous_data = new data_t[n_items];
    uint k = 0;
    for (uint i = 0; i < n_instances; i++) {
        if (sample_mask[i] == node_id) {
            contiguous_data[k++] = operator()(i, feature_id);
        }
    }
}

DirectDataset::DirectDataset(Dataset dataset) :
    data(dataset.data), n_rows(dataset.n_rows), n_cols(dataset.n_cols) {}

DirectDataset::DirectDataset(data_t* data, size_t n_instances, size_t n_features) :
    data(data), n_rows(n_instances), n_cols(n_features) {}

DirectDataset::DirectDataset(const DirectDataset& other) :
    data(other.data), n_rows(other.n_rows), n_cols(other.n_cols) {}

DirectDataset& DirectDataset::operator=(const DirectDataset& other) {
    this->data = other.data;
    this->n_rows = other.n_rows;
    this->n_cols = other.n_cols;
    return *this;
}

data_t DirectDataset::operator()(size_t i, size_t j) {
    return this->data[i * this->n_cols + j];
}

std::shared_ptr<void> DirectDataset::_operator_ev(const size_t j) {
    return std::shared_ptr<void>(
        new DirectDataset::Iterator<data_t>(data, n_cols));
}

void VirtualTargets::allocateFromSampleMask(
    size_t* sample_mask, size_t node_id, size_t n_items, size_t n_instances) {
    if (contiguous_labels != nullptr) {
        delete[] contiguous_labels;
    }
    contiguous_labels = new label_t[n_items];
    uint k = 0;
    for (uint i = 0; i < n_instances; i++) {
        if (sample_mask[i] == node_id) {
            contiguous_labels[k++] = operator[](i);
        }
    }
}

label_t* VirtualTargets::toLabels() {
    if (labels == nullptr) {
        labels = new label_t[getNumInstances()];
        for (unsigned int i = 0; i < getNumInstances(); i++) {
            labels[i] = static_cast<int>(operator[](i));
        }
    }
    return labels;
}

DirectTargets::DirectTargets(target_t* data, size_t n_instances) :
    data(data), n_rows(n_instances) {}

DirectTargets::DirectTargets(const DirectTargets& other) :
    data(other.data), n_rows(other.n_rows) {}

DirectTargets& DirectTargets::operator=(const DirectTargets& other) {
    this->data = other.data;
    this->n_rows = other.n_rows;
    return *this;
}

target_t DirectTargets::operator[](const size_t i) {
    return this->data[i];
}

}