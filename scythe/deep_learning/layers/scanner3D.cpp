/**
    scanner3D.cpp
    Multi-grained scanning

    @author Antoine Passemiers
    @version 1.0 10/06/2017
*/

#include "scanner3D.hpp"


ScannedDataset3D::ScannedDataset3D(size_t kc, size_t kr, size_t kd) : 
    N(0), M(0), P(0), Q(0), kc(kc), kr(kr), kd(kd), sc(0), sr(0), sd(0), Nprime(0), Mprime(0) {}

data_t ScannedDataset3D::operator()(size_t i, size_t j) {
    return 0; // TODO
}

size_t ScannedDataset3D::getNumInstances() {
    return this->Nprime;
}

size_t ScannedDataset3D::getNumFeatures() {
    return this->Mprime;
}

MultiGrainedScanner3D::MultiGrainedScanner3D(
        LayerConfig lconfig, size_t kc, size_t kr, size_t kd) : Layer(lconfig) {
    Layer::vdataset = std::shared_ptr<VirtualDataset>(new ScannedDataset3D(kc, kr, kd));
}