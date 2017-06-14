/**
    scanner1D.cpp
    Multi-grained scanning

    @author Antoine Passemiers
    @version 1.0 10/06/2017
*/

#include "scanner1D.hpp"


ScannedDataset1D::ScannedDataset1D(data_t* data, size_t N, size_t M, size_t kc) : 
    N(N),                // Number of instances
    M(M),                // Instance height
    kc(kc),              // Kernel width
    sc(M - kc + 1),      // Number of kernel positions per column
    Nprime(0), // TODO
    Mprime(0), // TODO
    data(data) {}

data_t ScannedDataset1D::operator()(size_t i, size_t j) {
    return 0; // TODO
}

size_t ScannedDataset1D::getNumInstances() {
    return this->Nprime;
}

size_t ScannedDataset1D::getNumFeatures() {
    return this->Mprime;
}

size_t ScannedDataset1D::getRequiredMemorySize() {
    return this->Nprime * this->Mprime;
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
            this->kc));
    return Layer::vdataset;
}

size_t MultiGrainedScanner1D::getRequiredMemorySize() {
    size_t memory_size = this->vdataset.get()->getNumInstances();
    assert(memory_size > 0);
    if (lconfig.fconfig.task == gbdf::CLASSIFICATION_TASK) {
        memory_size *= lconfig.fconfig.n_classes;
    }
    return memory_size;
}