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

ScannedTargets3D::ScannedTargets3D(data_t* data, size_t n_instances, size_t sc, size_t sr, size_t sd) :
    data(data), n_rows(n_instances), s(sc * sr * sd) {}

ScannedTargets3D::ScannedTargets3D(const ScannedTargets3D& other) :
    data(other.data), n_rows(other.n_rows), s(other.s) {}

ScannedTargets3D& ScannedTargets3D::operator=(const ScannedTargets3D& other) {
    this->data = data;
    this->n_rows = n_rows;
    this->s = s;
}

data_t ScannedTargets3D::operator[](const size_t i) {
    return data[i / s];
}

MultiGrainedScanner3D::MultiGrainedScanner3D(
        LayerConfig lconfig, size_t kc, size_t kr, size_t kd) : Layer(lconfig) {
    Layer::vdataset = std::shared_ptr<VirtualDataset>(
        new ScannedDataset3D(kc, kr, kd));
}

vdataset_p MultiGrainedScanner3D::virtualize(MDDataset dataset) {
    return nullptr; // TODO
}

vtargets_p MultiGrainedScanner3D::virtualize(Labels<target_t>* targets) {
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
    size_t memory_size = this->vdataset.get()->getNumInstances();
    assert(memory_size > 0);
    if (lconfig.fconfig.task == gbdf::CLASSIFICATION_TASK) {
        memory_size *= lconfig.fconfig.n_classes;
    }
    return memory_size;
}