/**
    scanner2D.cpp
    Multi-grained scanning

    @author Antoine Passemiers
    @version 1.0 10/06/2017
*/

#include "scanner2D.hpp"


ScannedDataset2D::ScannedDataset2D(
    data_t* data, size_t N, size_t M, size_t P, size_t kc, size_t kr) : 
    N(N),                // Number of instances
    M(M),                // Instance height
    P(P),                // Instance width
    kc(kc),              // Kernel width
    kr(kr),              // Kernel height
    sc(P - kc + 1),      // Number of kernel positions per column
    sr(M - kr + 1),      // Number of kernel positions per row
    Nprime(N * sr * sc), // Number of instances after scanning
    Mprime(kc * kr),     // Number of features after scanning
    data(data) {}

data_t ScannedDataset2D::operator()(size_t i, size_t j) {
    return 0; // TODO
}

size_t ScannedDataset2D::getNumInstances() {
    return this->Nprime;
}

size_t ScannedDataset2D::getNumFeatures() {
    return this->Mprime;
}

size_t ScannedDataset2D::getRequiredMemorySize() {
    return this->Nprime * this->Mprime;
}

MultiGrainedScanner2D::MultiGrainedScanner2D(
    LayerConfig lconfig, size_t kc, size_t kr) : Layer(lconfig), kc(kc), kr(kr) {}

vdataset_p MultiGrainedScanner2D::virtualize(MDDataset dataset) {
    assert(dataset.n_dims == 3);
    assert(dataset.dims[0] > 0);
    assert(dataset.dims[1] > 0);
    assert(dataset.dims[2] > 0);
    Layer::vdataset = std::shared_ptr<ScannedDataset2D>(
        new ScannedDataset2D(
            dataset.data,
            dataset.dims[0],
            dataset.dims[1],
            dataset.dims[2],
            this->kc,
            this->kr));
    return Layer::vdataset;
}

size_t MultiGrainedScanner2D::getRequiredMemorySize() {
    size_t memory_size = this->vdataset.get()->getNumInstances();
    assert(memory_size > 0);
    if (lconfig.fconfig.task == gbdf::CLASSIFICATION_TASK) {
        memory_size *= lconfig.fconfig.n_classes;
    }
    return memory_size;
}