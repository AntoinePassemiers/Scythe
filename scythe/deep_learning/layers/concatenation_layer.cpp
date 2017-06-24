/**
    concatenation_layer.cpp
    Base layer with concatenation of two feature matrices

    @author Antoine Passemiers
    @version 1.0 23/06/2017
*/

#include "concatenation_layer.hpp"


ConcatenationDataset::ConcatenationDataset(size_t n_instances, size_t n_virtual_features) {
    size_t n_elements = n_instances * n_virtual_features;
    this->data = new proba_t[n_elements];
    this->n_instances = n_instances;
    this->n_virtual_cols = n_virtual_features;
    this->stride = 0;
    this->dtype = gbdf::DTYPE_PROBA; // TODO : dtype in case of a regression
}

ConcatenationDataset::ConcatenationDataset(const ConcatenationDataset& other) :
    data(other.data),
    n_instances(other.n_instances),
    n_virtual_cols(other.n_virtual_cols),
    stride(other.stride),
    dtype(other.stride) {}

ConcatenationDataset& ConcatenationDataset::operator=(const ConcatenationDataset& other) {
    this->data = other.data;
    this->n_instances = other.n_instances;
    this->n_virtual_cols = other.n_virtual_cols;
    this->stride = other.stride;
    this->dtype = other.dtype;
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
    return static_cast<data_t>(data[i * n_virtual_cols + j]);
}

CascadeLayer::CascadeLayer(LayerConfig lconfig) : 
    Layer(lconfig) {
}

vdataset_p CascadeLayer::virtualize(MDDataset dataset) {
    // TODO : throw exception
    return nullptr;
}

vtargets_p CascadeLayer::virtualizeTargets(Labels<target_t>* targets) {
    DirectTargets* direct_targets = new DirectTargets(
        targets->data, targets->n_rows);
    return std::shared_ptr<VirtualTargets>(direct_targets);
}

size_t CascadeLayer::getRequiredMemorySize() {
    /*
    size_t memory_size = vdataset->getNumInstances();
    assert(memory_size > 0);
    if (lconfig.fconfig.task == gbdf::CLASSIFICATION_TASK) {
        memory_size *= lconfig.fconfig.n_classes;
    }
    return memory_size * lconfig.n_forests;
    */
    return 0;
}

size_t CascadeLayer::getNumVirtualFeatures() {
    /*
    size_t n_vfeatures = dynamic_cast<ScannedDataset1D*>(vdataset.get())->getSc();
    assert(n_vfeatures > 0);
    if (lconfig.fconfig.task == gbdf::CLASSIFICATION_TASK) {
        n_vfeatures *= lconfig.fconfig.n_classes;
    }
    return n_vfeatures * lconfig.n_forests;
    */
    return 0;
}