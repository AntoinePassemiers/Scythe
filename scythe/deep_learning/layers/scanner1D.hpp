/**
    scanner1D.hpp
    Multi-grained scanning

    @author Antoine Passemiers
    @version 1.0 10/06/2017
*/

#ifndef SCANNER1D_HPP_
#define SCANNER1D_HPP_

#include "../../misc/sets.hpp"
#include "layer.hpp"


class ScannedDataset1D : public VirtualDataset {
private:
    size_t N;  // Number of instances
    size_t M;  // Number of features
    size_t kc; // Kernel width
    size_t sc; // Number of kernel positions per column

    size_t Nprime; // Number of instances after scanning
    size_t Mprime; // Number of features after scanning

    data_t* data; // Pointer to the raw data
public:
    ScannedDataset1D(data_t* data, size_t N, size_t M, size_t kc);
    ~ScannedDataset1D() = default;
    data_t operator()(size_t i, size_t j);
    size_t getSc();
    size_t getNumInstances();
    size_t getNumFeatures();
    size_t getRequiredMemorySize();
};


class ScannedTargets1D : public VirtualTargets {
private:
    target_t* data;
    size_t n_rows;
    size_t s;
public:
    ScannedTargets1D(data_t* data, size_t n_instances, size_t sc);
    ScannedTargets1D(const ScannedTargets1D& other);
    ScannedTargets1D& operator=(const ScannedTargets1D& other);
    ~ScannedTargets1D() = default;
    data_t operator[](const size_t i);
    size_t getNumInstances() { return n_rows; }
    target_t* getValues() { return data; }
};


class MultiGrainedScanner1D : public Layer {
private:
    size_t kc; // Kernel width
public:
    MultiGrainedScanner1D(LayerConfig lconfig, size_t kc);
    ~MultiGrainedScanner1D() = default;
    vdataset_p virtualize(MDDataset dataset);
    vtargets_p virtualize(Labels<target_t>* targets);
    size_t getRequiredMemorySize();
};

#endif // SCANNER1D_HPP_