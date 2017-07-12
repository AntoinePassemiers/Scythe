/**
    scanner3D.cpp
    Multi-grained scanning

    @author Antoine Passemiers
    @version 1.0 10/06/2017
*/

#include "scanner3D.hpp"

namespace scythe {

ScannedDataset3D::ScannedDataset3D(
    data_t* data, size_t kc, size_t kr, size_t kd, int dtype) : 
    N(0), 
    M(0), 
    P(0), 
    Q(0), 
    kc(kc), 
    kr(kr), 
    kd(kd), 
    sc(0), 
    sr(0), 
    sd(0), 
    Nprime(0), 
    Mprime(0),
    data(data),
    dtype(dtype) {}

data_t ScannedDataset3D::operator()(size_t i, size_t j) {
    return 0; // TODO
}

std::shared_ptr<void> ScannedDataset3D::_operator_ev(const size_t j) {
    return nullptr; // TODO
}

size_t ScannedDataset3D::getSc() {
    return this->sc;
}

size_t ScannedDataset3D::getSr() {
    return this->sr;
}

size_t ScannedDataset3D::getSd() {
    return this->sd;
}

size_t ScannedDataset3D::getNumInstances() {
    return this->Nprime;
}

size_t ScannedDataset3D::getNumFeatures() {
    return this->Mprime;
}

size_t ScannedDataset3D::getRequiredMemorySize() {
    return this->Nprime * this->Mprime;
}

size_t ScannedDataset3D::getNumVirtualInstancesPerInstance() {
    return sr * sr * sd;
}

ScannedTargets3D::ScannedTargets3D(target_t* data, size_t n_instances, size_t sc, size_t sr, size_t sd) :
    data(data), n_rows(n_instances), s(sc * sr * sd) {}

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
    size_t n_rows = vdataset->getNumInstances();
    assert(sc > 0);
    assert(sr > 0);
    assert(sd > 0);
    return std::shared_ptr<ScannedTargets3D>(
        new ScannedTargets3D(targets->data, n_rows, sc, sr, sd));
}

size_t MultiGrainedScanner3D::getRequiredMemorySize() {
    size_t memory_size = this->vdataset->getNumInstances();
    assert(memory_size > 0);
    if (lconfig.fconfig.task == CLASSIFICATION_TASK) {
        memory_size *= lconfig.fconfig.n_classes;
    }
    return memory_size * lconfig.n_forests;
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