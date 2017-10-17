/**
    scanner1D.cpp
    Multi-grained scanning

    @author Antoine Passemiers
    @version 1.0 10/06/2017
*/

#include "scanner1D.hpp"

namespace scythe {

ScannedDataset1D::ScannedDataset1D(
    void* data, size_t N, size_t M, size_t kc, int dtype) : 
    N(N),                // Number of instances
    M(M),                // Instance height
    kc(kc),              // Kernel width
    sc(M - kc + 1),      // Number of kernel positions per column
    Nprime(N * sc),      // Number of instances after scanning
    Mprime(kc),          // Number of features after scanning
    data(data),          // Pointer to the raw data
    dtype(dtype) {}      // Raw data type

VirtualDataset* ScannedDataset1D::deepcopy() {
    size_t n_required_bytes = getNumRows() * getRowStride() * getItemStride();
    void* new_data = malloc(n_required_bytes);
    std::memcpy(new_data, data, n_required_bytes);
    return new ScannedDataset1D(new_data, N, M, kc, dtype);
}

VirtualDataset* ScannedDataset1D::createView(void* view, size_t n_rows) {
    return new ScannedDataset1D(view, n_rows, M, kc, dtype);
}

void ScannedDataset1D::allocateFromSampleMask(
    size_t* const sample_mask, size_t node_id, size_t feature_id, size_t n_items, size_t n_instances) {
    // TODO
}

data_t ScannedDataset1D::operator()(size_t i, size_t j) {
    return 0; // TODO
}

void ScannedDataset1D::_iterator_begin(const size_t j) {
    // TODO
}

void ScannedDataset1D::_iterator_inc() {
    // TODO
}

data_t ScannedDataset1D::_iterator_deref() {
    return 0.0; // TODO
}

ScannedTargets1D::ScannedTargets1D(target_t* data, size_t n_instances, size_t sc) :
    data(data), n_rows(n_instances), s(sc) {}

VirtualTargets* ScannedTargets1D::deepcopy() {
    size_t n_required_bytes = getNumInstances() * sizeof(target_t);
    target_t* new_data = static_cast<target_t*>(malloc(n_required_bytes));
    std::memcpy(new_data, data, n_required_bytes);
   return new ScannedTargets1D(new_data, n_rows, s);
}

VirtualTargets* ScannedTargets1D::createView(void* view, size_t new_n_rows) {
    return new ScannedTargets1D(static_cast<target_t*>(view), new_n_rows, s);
}

void ScannedTargets1D::allocateFromSampleMask(
    size_t* sample_mask, size_t node_id, size_t n_items, size_t n_instances) {
    // TODO
}

void ScannedTargets1D::_iterator_begin() {
    // TODO
}

void ScannedTargets1D::_iterator_inc() {
    // TODO
}

data_t ScannedTargets1D::_iterator_deref() {
    return 0.0; // TODO
}

target_t ScannedTargets1D::operator[](const size_t i) {
    return data[i / s];
}

MultiGrainedScanner1D::MultiGrainedScanner1D(LayerConfig lconfig, size_t kc) : 
    Layer(lconfig), kc(kc) {
}

vdataset_p MultiGrainedScanner1D::virtualize(MDDataset dataset) {
    assert(dataset.n_dims == 2);
    assert(dataset.dims[0] > 0);
    assert(dataset.dims[1] > 0);
    Layer::vdataset = std::shared_ptr<ScannedDataset1D>(
        new ScannedDataset1D(
            dataset.data,
            dataset.dims[0],
            dataset.dims[1],
            this->kc,
            dataset.dtype));
    return Layer::vdataset;
}

vtargets_p MultiGrainedScanner1D::virtualizeTargets(Labels* targets) {
    ScannedDataset1D* vdataset = dynamic_cast<ScannedDataset1D*>((this->vdataset).get());
    size_t sc = vdataset->getSc();
    size_t n_rows = vdataset->getNumRows();
    assert(sc > 0);
    return std::shared_ptr<ScannedTargets1D>(new ScannedTargets1D(targets->data, n_rows, sc));
}

size_t MultiGrainedScanner1D::getNumVirtualFeatures() {
    size_t n_vfeatures = dynamic_cast<ScannedDataset1D*>(vdataset.get())->getSc();
    assert(n_vfeatures > 0);
    if (lconfig.fconfig.task == CLASSIFICATION_TASK) {
        n_vfeatures *= lconfig.fconfig.n_classes;
    }
    return n_vfeatures * lconfig.n_forests;
}

} // namespace