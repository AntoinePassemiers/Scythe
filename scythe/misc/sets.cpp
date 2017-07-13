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
    /**
        Allocate memory for storing temporary values of a single feature,
        for the data samples belonging to the current node.
        This method is called right before the inner loop of the CART algorithm,
        and its purpose is to avoid calling virtual functions inside the vectorized
        inner loop.

        @param sample_mask
            Pointer indicating for each data sample the id of the node it belongs to
        @param node_id
            Id of the current node
        @param feature_id
            Id of the attribute whose values are going to be stored
        @param n_items
            Number of data samples belonging to the current node
        @param n_instances
            Number of data samples in the whole dataset
    */
    if (n_items != this.n_contiguous_items) {
        if (contiguous_data != nullptr) {
            delete[] contiguous_data;
        }
        contiguous_data = new data_t[n_items];
    }
    uint k = 0;
    for (uint i = 0; i < n_instances; i++) {
        if (sample_mask[i] == node_id) {
            contiguous_data[k++] = operator()(i, feature_id);
        }
    }
    this.n_contiguous_items = n_items;
}

DirectDataset::DirectDataset(Dataset dataset) :
    data(dataset.data), n_rows(dataset.n_rows), n_cols(dataset.n_cols) {}

DirectDataset::DirectDataset(data_t* data, size_t n_instances, size_t n_features) :
    data(data), n_rows(n_instances), n_cols(n_features) {}

data_t DirectDataset::operator()(size_t i, size_t j) {
    /**
        Returns the value of attribute j for sample i

        @param i
            Row id
        @param j
            Column id
    */
    return this->data[i * this->n_cols + j];
}

std::shared_ptr<void> DirectDataset::_operator_ev(const size_t j) {
    /**
        Type erasure of DirectDataset::Iterator<T>. This method returns
        an object that iterates over the attribute j.

        @param j
            Column id
    */
    return std::shared_ptr<void>(
        new DirectDataset::Iterator<data_t>(data, n_cols));
}

void VirtualTargets::allocateFromSampleMask(
    size_t* sample_mask, size_t node_id, size_t n_items, size_t n_instances) {
    /**
        Allocate memory for storing temporary values of the labels,
        for the data samples belonging to the current node.
        This method is called right before the inner loop of the CART algorithm,
        and its purpose is to avoid calling virtual functions inside the vectorized
        inner loop.

        @param sample_mask
            Pointer indicating for each data sample the id of the node it belongs to
        @param node_id
            Id of the current node
        @param n_items
            Number of data samples belonging to the current node
        @param n_instances
            Number of data samples in the whole dataset
    */
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

DirectTargets::DirectTargets(target_t* data, size_t n_instances) :
    data(data), n_rows(n_instances) {}

target_t DirectTargets::operator[](const size_t i) {
    /**
        Returns label of the data sample i

        @param i
            Row id
    */
    return this->data[i];
}

}