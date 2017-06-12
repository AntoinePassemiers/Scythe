/**
    scanner1D.cpp
    Multi-grained scanning

    @author Antoine Passemiers
    @version 1.0 10/06/2017
*/

#include "scanner1D.hpp"


ScannedDataset1D::ScannedDataset1D(size_t kc) : kc(kc) {}

data_t ScannedDataset1D::operator()(size_t i, size_t j) {
    return 0; // TODO
}

size_t ScannedDataset1D::getNumInstances() {
    return this->Nprime;
}

size_t ScannedDataset1D::getNumFeatures() {
    return this->Mprime;
}

MultiGrainedScanner1D::MultiGrainedScanner1D(LayerConfig lconfig, size_t kc) : Layer(lconfig) {
    Layer::vdataset = std::shared_ptr<VirtualDataset>(new ScannedDataset1D(kc));
}