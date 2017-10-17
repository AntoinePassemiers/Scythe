/**
    concatenation_layer.cpp
    Base layer with concatenation of two feature matrices

    @author Antoine Passemiers
    @version 1.0 23/06/2017
*/

#include "concatenation_layer.hpp"


namespace scythe {

ConcatenationDataset::ConcatenationDataset(
    proba_t* data, size_t n_instances, size_t n_virtual_cols, size_t stride, int dtype) :
    data(data), n_instances(n_instances), n_virtual_cols(n_virtual_cols), stride(stride), dtype(dtype) {}

ConcatenationDataset::ConcatenationDataset(size_t n_instances, size_t n_virtual_features) :
    data(new proba_t[n_instances * n_virtual_features]),
    n_instances(n_instances),
    n_virtual_cols(n_virtual_features),
    stride(0),
    dtype(NPY_FLOAT32_NUM) {} // TODO : dtype in case of regresion task

VirtualDataset* ConcatenationDataset::deepcopy() {
    size_t n_required_bytes = getNumRows() * getRowStride() * getItemStride();
    void* new_data = malloc(n_required_bytes);
    std::memcpy(new_data, data, n_required_bytes);
    return new ConcatenationDataset(static_cast<proba_t*>(new_data), n_instances, n_virtual_cols, stride, dtype);
}

VirtualDataset* ConcatenationDataset::createView(void* view, size_t n_rows) {
    return new ConcatenationDataset(static_cast<proba_t*>(view), n_rows, n_virtual_cols, stride, dtype);
}

void ConcatenationDataset::concatenate(float* new_data, size_t width) {
    // TODO: parallel computing
    std::cout << "Concatenation : " << n_instances << ", ";
    std::cout << n_virtual_cols << ", " << width << ", " << this->stride << std::endl;
    size_t k = this->stride;
    assert(this->stride + width <= n_virtual_cols);
    for (unsigned int i = 0; i < this->n_instances; i++) {
        for (unsigned int j = 0; j < width; j++) {
            this->data[i * n_virtual_cols + k + j] = static_cast<proba_t>(new_data[i * width + j]);
            //std::cout << new_data[i * width + j] << ", ";
        }
        //std::cout << std::endl;
    }
    this->stride += width;
}

void ConcatenationDataset::allocateFromSampleMask(
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
            t_contiguous_data[k++] = static_cast<fast_data_t>(data[iterator_cursor]);
        }
        iterator_cursor += n_virtual_cols;
    }
    contiguous_data = static_cast<void*>(t_contiguous_data);
    assert(k == n_items);
}

data_t ConcatenationDataset::operator()(const size_t i, const size_t j) {
    return static_cast<data_t>(data[i * n_virtual_cols + j]);
}

void ConcatenationDataset::_iterator_begin(const size_t j) {
    iterator_cursor = j;
}

void ConcatenationDataset::_iterator_inc() {
    iterator_cursor += n_virtual_cols;
}

data_t ConcatenationDataset::_iterator_deref() {
    return data[iterator_cursor];
}

CascadeLayer::CascadeLayer(LayerConfig lconfig) : 
    Layer(lconfig) {
}

vdataset_p CascadeLayer::virtualize(MDDataset dataset) {
    // TODO : throw exception
    return nullptr;
}

vtargets_p CascadeLayer::virtualizeTargets(Labels* targets) {
    DirectTargets* direct_targets = new DirectTargets(
        targets->data, targets->n_rows);
    return std::shared_ptr<VirtualTargets>(direct_targets);
}

size_t CascadeLayer::getNumVirtualFeatures() {
    if (lconfig.fconfig.task == CLASSIFICATION_TASK) {
        return lconfig.n_forests * lconfig.fconfig.n_classes;
    }
    else {
        return lconfig.n_forests;
    }
}

} // namespace