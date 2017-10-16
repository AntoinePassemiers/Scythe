/**
    scanner3D.cpp
    Multi-grained scanning

    @author Antoine Passemiers
    @version 1.0 10/06/2017
*/

#include "scanner3D.hpp"

namespace scythe {

ScannedDataset3D::ScannedDataset3D(
    void* data, size_t N, size_t M, size_t P, size_t Q, size_t kc, size_t kr, size_t kd, int dtype) : 
    N(N),       // Number of instances
    M(M),       // Instance height
    P(P),       // Instance width
    Q(Q),       // Instance depth
    kc(kc),     // Kernel width
    kr(kr),     // Kernel height
    kd(kd),     // Kernel depth
    sc(0),      // Number of kernel positions per column // TODO
    sr(0),      // Number of kernel positions per row    // TODO
    sd(0),      // Number of kernel positions per depth index // TODO
    Nprime(0),  // Number of instances after scanning // TODO
    Mprime(0),  // Number of features after scanning // TODO
    data(data), // Pointer to the raw data
    dtype(dtype) {} // Raw data type

VirtualDataset* ScannedDataset3D::deepcopy() {
    size_t n_required_bytes = getNumRows() * getRowStride() * getItemStride();
    void* new_data = malloc(n_required_bytes);
    std::memcpy(new_data, data, n_required_bytes);
    return new ScannedDataset3D(new_data, N, M, P, Q, kc, kr, kd, dtype);
}

void ScannedDataset3D::allocateFromSampleMask(
    size_t* const sample_mask, size_t node_id, size_t feature_id, size_t n_items, size_t n_instances) {
    // TODO
}

data_t ScannedDataset3D::operator()(size_t i, size_t j) {
    return 0; // TODO
}

void ScannedDataset3D::_iterator_begin(const size_t j) {
    // TODO
}

void ScannedDataset3D::_iterator_inc() {
    // TODO
}

data_t ScannedDataset3D::_iterator_deref() {
    return 0.0; // TODO
}

ScannedTargets3D::ScannedTargets3D(target_t* data, size_t n_instances, size_t sc, size_t sr, size_t sd) :
    data(data), n_rows(n_instances), sc(sc), sr(sr), sd(sd), s(sc * sr * sd) {}

VirtualTargets* ScannedTargets3D::deepcopy() {
    size_t n_required_bytes = getNumInstances() * sizeof(target_t);
    target_t* new_data = static_cast<target_t*>(malloc(n_required_bytes));
    std::memcpy(new_data, data, n_required_bytes);
   return new ScannedTargets3D(new_data, n_rows, sc, sr, sd);
}

void ScannedTargets3D::allocateFromSampleMask(
    size_t* sample_mask, size_t node_id, size_t n_items, size_t n_instances) {
    // TODO
}

void ScannedTargets3D::_iterator_begin() {
    // TODO
}

void ScannedTargets3D::_iterator_inc() {
    // TODO
}

target_t ScannedTargets3D::_iterator_deref() {
    return 0.0; // TODO
}

target_t ScannedTargets3D::operator[](const size_t i) {
    return data[i / s];
}

MultiGrainedScanner3D::MultiGrainedScanner3D(LayerConfig lconfig, size_t kc, size_t kr, size_t kd) : 
    Layer(lconfig), kc(kc), kr(kr), kd(kd) {}

vdataset_p MultiGrainedScanner3D::virtualize(MDDataset dataset) {
    return nullptr; // TODO
}

vtargets_p MultiGrainedScanner3D::virtualizeTargets(Labels* targets) {
    ScannedDataset3D* vdataset = dynamic_cast<ScannedDataset3D*>((this->vdataset).get());
    size_t sc = vdataset->getSc();
    size_t sr = vdataset->getSr();
    size_t sd = vdataset->getSd();
    size_t n_rows = vdataset->getNumRows();
    assert(sc > 0);
    assert(sr > 0);
    assert(sd > 0);
    return std::shared_ptr<ScannedTargets3D>(
        new ScannedTargets3D(targets->data, n_rows, sc, sr, sd));
}

size_t MultiGrainedScanner3D::getNumVirtualFeatures() {
    ScannedDataset3D* sdataset = dynamic_cast<ScannedDataset3D*>(vdataset.get());
    size_t n_vfeatures = sdataset->getSc() * sdataset->getSr() * sdataset->getSd();
    assert(n_vfeatures > 0);
    if (lconfig.fconfig.task == CLASSIFICATION_TASK) {
        n_vfeatures *= lconfig.fconfig.n_classes;
    }
    return n_vfeatures * lconfig.n_forests;
}

}