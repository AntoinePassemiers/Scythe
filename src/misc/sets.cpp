/**
    sets.cpp
    Datasets' structures
    
    @author Antoine Passemiers
    @version 1.0 09/06/2017
*/

#include "sets.hpp"


namespace scythe {

size_t VirtualDataset::getItemStride() {
    switch (getDataType()) {
        case NPY_BOOL_NUM:
            return sizeof(npy_bool);
        case NPY_INT8_NUM:
            return sizeof(npy_int8);
        case NPY_UINT8_NUM:
            return sizeof(npy_uint8);
        case NPY_INT16_NUM:
            return sizeof(npy_int16);
        case NPY_UINT16_NUM:
            return sizeof(npy_uint16);
        case NPY_INT32_NUM:
            return sizeof(npy_int32);
        case NPY_UINT32_NUM:
            return sizeof(npy_uint32);
        case NPY_INT64_NUM:
            return sizeof(npy_int64);
        case NPY_UINT64_NUM:
            return sizeof(npy_uint64);
        case NPY_FLOAT32_NUM:
            return sizeof(npy_float32);
        case NPY_FLOAT64_NUM:
            return sizeof(npy_float64);
        case NPY_FLOAT16_NUM:
            return sizeof(npy_float16);
        default:
            throw UnhandledDtypeException();
    }
}

VirtualDataset* VirtualDataset::shuffleAndCreateView(std::vector<size_t>& indexes) {
    size_t n_samples = getNumRows();
    assert(indexes.size() <= n_samples);
    size_t n_bytes_per_sample = getItemStride() * getRowStride();
    BYTE* temp = static_cast<BYTE*>(std::malloc(n_bytes_per_sample));
    BYTE* data = static_cast<BYTE*>(getData());
    size_t start = n_samples - indexes.size();
    for (size_t i = start; i < n_samples; i++) {
        BYTE* a = data + (i * n_bytes_per_sample);
        BYTE* b = data + (indexes.at(i - start) * n_bytes_per_sample);
        std::memcpy(temp, b, n_bytes_per_sample);
        std::memcpy(b, a, n_bytes_per_sample);
        std::memcpy(a, temp, n_bytes_per_sample);
    }
    std::free(temp);
    void* view = static_cast<void*>(data + (start * n_bytes_per_sample));
    return createView(view, indexes.size());
}

DirectDataset::DirectDataset(Dataset dataset) :
    data(dataset.data), n_rows(dataset.n_rows), n_cols(dataset.n_cols), dtype(dataset.dtype)     {}

DirectDataset::DirectDataset(void* data, size_t n_instances, size_t n_features) :
    data(data), n_rows(n_instances), n_cols(n_features) {}

VirtualDataset* DirectDataset::deepcopy() {
    throw WrongVirtualDatasetException();
}

VirtualDataset* DirectDataset::createView(void* view, size_t n_rows) {
    throw WrongVirtualDatasetException();
}

void DirectDataset::allocateFromSampleMask(
    size_t* const sample_mask, size_t node_id, size_t feature_id, size_t n_items, size_t n_instances) {
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
    fast_data_t* t_contiguous_data = static_cast<fast_data_t*>(contiguous_data);
    if (n_items != this->n_contiguous_items) { // TODO
        if (contiguous_data != nullptr) {
            delete[] contiguous_data;
        }
        t_contiguous_data = new fast_data_t[n_items];
        this->n_contiguous_items = n_items;
    }

    uint k = 0;
    iterator_cursor = feature_id;
    for (uint i = 0; i < n_instances; i++) {
        if (sample_mask[i] == node_id) {
            t_contiguous_data[k++] = static_cast<fast_data_t>(static_cast<data_t*>(data)[iterator_cursor]);
        }
        iterator_cursor += n_cols;
    }
    this->contiguous_data = static_cast<void*>(t_contiguous_data);
    assert(k == n_items);
}

data_t DirectDataset::operator()(size_t i, size_t j) {
    /**
        Returns the value of attribute j for sample i

        @param i
            Row id
        @param j
            Column id
    */
    return static_cast<data_t*>(this->data)[i * this->n_cols + j];
}

void DirectDataset::_iterator_begin(const size_t j) {
    iterator_cursor = j;
}

void DirectDataset::_iterator_inc() {
    iterator_cursor += n_cols;
}

data_t DirectDataset::_iterator_deref() {
    return static_cast<data_t*>(data)[iterator_cursor];
}

VirtualTargets* VirtualTargets::shuffleAndCreateView(std::vector<size_t>& indexes) {
    size_t n_samples = getNumInstances();
    assert(indexes.size() <= n_samples);
    target_t temp;
    target_t* data = getValues();
    size_t start = n_samples - indexes.size();
    for (size_t i = start; i < n_samples; i++) {
        temp = data[indexes.at(i - start)];
        data[indexes.at(i - start)] = data[i];
        data[i] = temp;
    }
    void* view = static_cast<void*>(&data[start]);
    return createView(view, indexes.size());
}

DirectTargets::DirectTargets(target_t* data, size_t n_instances) :
    data(data), n_rows(n_instances) {}

VirtualTargets* DirectTargets::deepcopy() {
    size_t n_required_bytes = n_rows * sizeof(target_t);
    target_t* new_data = static_cast<target_t*>(malloc(n_required_bytes));
    std::memcpy(new_data, data, n_required_bytes);
   return new DirectTargets(new_data, n_rows);
}

VirtualTargets* DirectTargets::createView(void* view, size_t n_rows) {
    return new DirectTargets(static_cast<target_t*>(view), n_rows);
}

void DirectTargets::allocateFromSampleMask(
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
    if (n_items != this->n_contiguous_items) { // TODO
        if (contiguous_labels != nullptr) {
            delete[] contiguous_labels;
        }
        contiguous_labels = new label_t[n_items];
        this->n_contiguous_items = n_items;
    }
    uint k = 0;
    iterator_cursor = 0;
    for (uint i = 0; i < n_instances; i++) {
        if (sample_mask[i] == node_id) {
            contiguous_labels[k++] = data[iterator_cursor];
        }
        iterator_cursor++;
    }
    assert(k == n_items);
}

void DirectTargets::_iterator_begin() {
    iterator_cursor = 0;
}

void DirectTargets::_iterator_inc() {
    iterator_cursor++;
}

data_t DirectTargets::_iterator_deref() {
    return data[iterator_cursor];
}

target_t DirectTargets::operator[](const size_t i) {
    /**
        Returns label of the data sample i

        @param i
            Row id
    */
    return this->data[i];
}

} // namespace