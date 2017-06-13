/**
    scanner2D.cpp
    Multi-grained scanning

    @author Antoine Passemiers
    @version 1.0 10/06/2017
*/

#include "scanner2D.hpp"


ScannedDataset2D::ScannedDataset2D(size_t kc, size_t kr) : kc(kc), kr(kr) {}

data_t ScannedDataset2D::operator()(size_t i, size_t j) {
    return 0; // TODO
}

size_t ScannedDataset2D::getNumInstances() {
    return this->Nprime;
}

size_t ScannedDataset2D::getNumFeatures() {
    return this->Mprime;
}

MultiGrainedScanner2D::MultiGrainedScanner2D(
        LayerConfig lconfig, size_t kc, size_t kr) : Layer(lconfig) {
    Layer::vdataset = std::shared_ptr<VirtualDataset>(new ScannedDataset2D(kc, kr));
}