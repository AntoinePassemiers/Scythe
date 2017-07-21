/**
    concatenation_layer.cpp
    Base layer with concatenation of two feature matrices

    @author Antoine Passemiers
    @version 1.0 23/06/2017
*/

#include "concatenation_layer.hpp"


namespace scythe {

ConcatenationDataset::ConcatenationDataset(size_t n_instances, size_t n_virtual_features) :
    data(new proba_t[n_instances * n_virtual_features]),
    n_instances(n_instances),
    n_virtual_cols(n_virtual_features),
    stride(0),
    dtype(DTYPE_PROBA) {} // TODO : dtype in case of regresion task

void ConcatenationDataset::concatenate(float* new_data, size_t width) {
    // TODO: parallel computing
    std::cout << "Concatenation : " << n_instances << ", ";
    std::cout << n_virtual_cols << ", " << width << ", " << this->stride << std::endl;
    size_t k = this->stride;
    assert(this->stride + width <= n_virtual_cols);
    for (unsigned int i = 0; i < this->n_instances; i++) {
        for (unsigned int j = 0; j < width; j++) {
            this->data[i * n_virtual_cols + k + j] = static_cast<proba_t>(new_data[i * width + j]);
        }
    }
    this->stride += width;
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

}